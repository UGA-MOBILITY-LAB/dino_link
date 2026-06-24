"""
DinoLink: DINOv2 token extraction -> Top-K selection -> Quantizer -> TokenDecoder.
Entry point: loads config, trains/evaluates, saves to logs/run_xx.

Pipeline:
  image -> DINOv2(frozen) -> top-k -> projector -> Quantizer(z_q_st, indices)
  -> TokenDecoder(z_q_st -> reconstructed DINO token)
Loss = token_loss + commit_loss (+ optional entropy regularization)
"""
import argparse
import copy
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# Reduce CUDA fragmentation (helps when reserved but unallocated memory is large)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.dinov2_extractor import DINOv2Extractor
from models.token_selector import TokenSelector
from models.quantizer import VQ, RVQ, vq_effectiveness  # noqa: F401
from models.projector import Projector
from models.decoder import TokenDecoder
from utils.image_loader import get_dataloader
from utils.visualizer import Visualizer
from losses import compute_losses


# ---------------------------------------------------------------------------
# Config / param helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_run_config(cfg: dict, run_dir: Path, filename: str = "config_used.yaml") -> Path:
    """Save the effective config for this run into log dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / filename
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def count_parameters(modules: dict) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for m in modules.values() if isinstance(m, nn.Module)
                for p in m.parameters())
    trainable = sum(p.numel() for m in modules.values() if isinstance(m, nn.Module)
                    for p in m.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device):
    model_cfg = cfg["model"]
    quant_cfg = cfg["quantizer"]
    dec_cfg = cfg.get("decoder", {})
    top_k = model_cfg["top_k"]
    selection = model_cfg.get("selection", "attention")

    data_image_size = cfg["data"].get("image_size")
    no_resize = data_image_size is None
    dinov2_image_size = None if no_resize else model_cfg.get("dinov2_image_size", 224)
    extractor = DINOv2Extractor(
        model_cfg["dinov2_name"],
        image_size=dinov2_image_size,
    ).to(device)
    nh = extractor.num_patches_h if extractor.num_patches_h is not None else 16
    nw = extractor.num_patches_w if extractor.num_patches_w is not None else 16
    num_patches = nh * nw
    if no_resize:
        # Variable-size mode: actual token count is decided per-image at runtime.
        k_use = top_k
    else:
        k_use = min(top_k, num_patches)
    hidden_size = extractor.hidden_size
    print(f"[Model] patch grid: {nh}x{nw} = {num_patches} patches, "
          f"top_k={top_k}, k_use={k_use}, hidden_size={hidden_size}")

    selector = TokenSelector(
        top_k=top_k,
        mode=selection,
        num_patches_h=nh,
        num_patches_w=nw,
    )
    quant_type = quant_cfg["type"]
    if quant_type == "rvq":
        quantizer = RVQ(
            quant_cfg["codebook_size"],
            quant_cfg["z_dim"],
            quant_cfg.get("num_quantizers", 2),
        ).to(device)
    elif quant_type == "vq":
        quantizer = VQ(quant_cfg["codebook_size"], quant_cfg["z_dim"]).to(device)
    else:
        raise ValueError(f"Unsupported quantizer.type: {quant_type}")

    z_dim = quant_cfg["z_dim"]
    projector = Projector(hidden_size, z_dim).to(device)

    # TokenDecoder: each VQ token -> DINO hidden token
    token_decoder = TokenDecoder(
        z_dim=z_dim,
        out_dim=hidden_size,
        hidden_dim=dec_cfg.get("hidden_dim", 512),
        use_pos=dec_cfg.get("use_pos", True),
    ).to(device)

    n_dec_params = sum(p.numel() for p in token_decoder.parameters())
    print(
        f"[Model] TokenDecoder: z_dim={z_dim} -> hidden_size={hidden_size}, "
        f"hidden_dim={dec_cfg.get('hidden_dim', 512)}, params={n_dec_params:,}"
    )

    return {
        "extractor": extractor,
        "selector": selector,
        "projector": projector,
        "quantizer": quantizer,
        "token_decoder": token_decoder,
        "k_use": k_use,
        "num_patches_h": nh,
        "num_patches_w": nw,
        "z_dim": z_dim,
        "hidden_size": hidden_size,
    }


# ---------------------------------------------------------------------------
# Single-step forward (train or eval)
# ---------------------------------------------------------------------------

def run_step(
    batch: torch.Tensor,
    modules: dict,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    device: torch.device,
    training: bool,
):
    """
    Per-token pipeline:
        image -> DINOv2(frozen) -> top-k -> projector -> VQ(z_q_st, indices)
        -> TokenDecoder: z_q_st -> (B, k, hidden_size)
    Loss = token MSE + commit_loss (+ optional entropy)
    """
    extractor = modules["extractor"]
    selector = modules["selector"]
    projector = modules["projector"]
    quantizer = modules["quantizer"]
    token_decoder = modules["token_decoder"]
    k_use = modules["k_use"]

    x = batch.to(device)
    if x.dim() == 5:
        # Multi-view input: (B, V, C, H, W) -> treat each camera view as one sample.
        b, v, c, h, w = x.shape
        x = x.reshape(b * v, c, h, w)
    B = x.shape[0]
    z_dim = modules["z_dim"]

    # 1. DINOv2: extract all patch tokens (frozen)
    with torch.no_grad():
        patch_tokens, attention_scores, num_patches_h, num_patches_w = extractor(x)

    # 2. Attention-based top-k selection
    selected, selected_indices = selector(
        patch_tokens, attention_scores,
        num_patches_h=num_patches_h, num_patches_w=num_patches_w,
    )
    k_cur = selected.shape[1]

    # 3. projector -> quantizer (VQ / RVQ)
    z_enc = projector(selected)  # (B, k, z_dim)
    z_q_st_flat, indices, quant_stats = quantizer(z_enc.reshape(-1, z_dim))
    z_q_st = z_q_st_flat.reshape(B, k_cur, z_dim)

    # 4. TokenDecoder: quantized latent tokens -> reconstructed selected DINO token
    with torch.no_grad():
        rows = selected_indices // num_patches_w
        cols = selected_indices % num_patches_w
        row_norm = (rows.float() / max(num_patches_h - 1, 1)) * 2.0 - 1.0
        col_norm = (cols.float() / max(num_patches_w - 1, 1)) * 2.0 - 1.0
        patch_pos = torch.stack([row_norm, col_norm], dim=-1)  # (B, k, 2)
    decoded_tokens = token_decoder(z_q_st, patch_pos=patch_pos)  # (B, k, hidden_size)

    # 5. Unified loss computation (token / commit)
    losses, loss, loss_extras = compute_losses(
        decoded_tokens=decoded_tokens,
        selected=selected,
        quantizer=quantizer,
        z_enc=z_enc,
        indices=indices,
        z_dim=z_dim,
        cfg=cfg,
        quant_stats=quant_stats,
    )
    z_enc_flat = loss_extras["z_enc_flat"]
    z_q_raw = loss_extras["z_q_raw"]
    token_mse_per_selected = loss_extras["token_mse_per_selected"]

    if training and optimizer is not None:
        if isinstance(optimizer, dict):
            active_optimizers = [opt for opt in optimizer.values() if opt is not None]
        else:
            active_optimizers = [optimizer]
        for opt in active_optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in active_optimizers:
            opt.step()

    # --- Metrics ---
    codebook_size = quantizer.codebook_size
    quant_metrics = vq_effectiveness(indices, z_enc_flat.detach(),
                                     z_q_raw.detach(), codebook_size)

    # VQ per-token L2 error (selected positions)
    N = num_patches_h * num_patches_w
    vq_err_flat = (z_enc_flat - z_q_raw).pow(2).sum(dim=-1).detach()
    vq_error_per_token = torch.zeros(B, N, device=device, dtype=vq_err_flat.dtype)
    vq_error_per_token.scatter_(1, selected_indices, vq_err_flat.reshape(B, k_cur))

    # Per-token token-MSE on selected indices (unselected = 0)
    with torch.no_grad():
        per_token_mse_all = torch.zeros(B, N, device=device, dtype=token_mse_per_selected.dtype)
        per_token_mse_all.scatter_(1, selected_indices, token_mse_per_selected.detach())

    token_mse_sel = token_mse_per_selected.mean().item()
    token_cos_sim = F.cosine_similarity(
        decoded_tokens.detach(), selected.detach(), dim=-1
    ).mean().item()
    quant_error_sel = vq_err_flat.mean().item()
    bits_per_index = int(math.ceil(math.log2(max(codebook_size, 2))))
    num_quantizers = int(getattr(quantizer, "num_quantizers", 1))
    bits_per_token = bits_per_index * num_quantizers
    indices_payload_bytes_per_image = float(k_cur * bits_per_token) / 8.0

    metrics = {
        "token_loss": losses["token"].item(),
        "token_mse_sel": token_mse_sel,
        "token_cos_sim": token_cos_sim,
        "commit_loss": losses["commit"].item(),
        "quant_error_sel": quant_error_sel,
        "indices_bits_per_token": float(bits_per_token),
        "indices_payload_bytes_per_image": indices_payload_bytes_per_image,
        "loss": loss.item(),
        "num_tokens": k_cur,
        **quant_metrics,
    }
    extra = {
        "per_token_mse_all": per_token_mse_all,           # (B, N)
        "vq_error_per_token": vq_error_per_token,
        "num_patches_h": num_patches_h,
        "num_patches_w": num_patches_w,
        "z_enc_flat": z_enc_flat.detach(),
    }
    return attention_scores, selected_indices, token_mse_per_selected.detach(), metrics, extra


# ---------------------------------------------------------------------------
# Checkpoint load
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path_or_dict, modules: dict,
                    optimizer: torch.optim.Optimizer | dict | None = None,
                    device: torch.device = None):
    if isinstance(ckpt_path_or_dict, (Path, str)):
        ckpt = torch.load(ckpt_path_or_dict, map_location=device or "cpu")
    else:
        ckpt = ckpt_path_or_dict
    epoch = ckpt.get("epoch", 0)
    if "projector" in ckpt:
        modules["projector"].load_state_dict(ckpt["projector"])
    else:
        # Compatible with old ckpt (proj + proj_norm)
        if "proj" in ckpt:
            modules["projector"].linear.load_state_dict(ckpt["proj"])
        if "proj_norm" in ckpt:
            modules["projector"].norm.load_state_dict(ckpt["proj_norm"])
    modules["quantizer"].load_state_dict(ckpt["quantizer"])
    if "token_decoder" in ckpt:
        modules["token_decoder"].load_state_dict(ckpt["token_decoder"])
    else:
        raise KeyError("No decoder state found in checkpoint.")
    if optimizer is not None:
        if isinstance(optimizer, dict):
            # New format
            if "optimizer_adam" in ckpt and optimizer.get("adam") is not None:
                optimizer["adam"].load_state_dict(ckpt["optimizer_adam"])
            if "optimizer_codebook" in ckpt and optimizer.get("codebook") is not None:
                optimizer["codebook"].load_state_dict(ckpt["optimizer_codebook"])
            # Backward compatible format
            if "optimizer" in ckpt and optimizer.get("adam") is not None:
                optimizer["adam"].load_state_dict(ckpt["optimizer"])
        else:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
    return epoch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_train(cfg: dict, args, device: torch.device):
    # When --run_name is not specified, default to the run time, e.g. run_20250203_143052
    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = cfg["run"].get("log_dir", "logs")
    run_dir = Path(log_dir) / run_name
    print(f"[Run] log dir: {run_dir}")
    visualizer = Visualizer(log_dir, run_name)

    data_root = cfg["data"]["data_root"]
    if getattr(args, "data_root", None):
        data_root = args.data_root
    # Persist effective train config for reproducibility.
    cfg_for_run = copy.deepcopy(cfg)
    cfg_for_run.setdefault("data", {})
    cfg_for_run["data"]["data_root"] = data_root
    saved_cfg_path = save_run_config(cfg_for_run, run_dir)
    print(f"[Run] saved config: {saved_cfg_path}")
    try:
        loader, _ = get_dataloader(
            data_root,
            cfg["data"]["dataset"],
            cfg["data"]["batch_size"],
            cfg["data"].get("num_workers", 4),
            cfg["data"].get("image_size", 224),
            shuffle=True,
            multi_view=cfg["data"].get("multi_view", False),
        )
    except Exception as e:
        print(f"[WARN] DataLoader failed: {e}. Using dummy data for demo.")
        from torch.utils.data import TensorDataset, DataLoader
        dummy = torch.rand(32, 3, 224, 224)
        loader = DataLoader(TensorDataset(dummy),
                            batch_size=cfg["data"]["batch_size"], shuffle=True)

    modules = build_model(cfg, device)
    total_params, trainable_params = count_parameters(modules)
    print(f"[Model] Parameters: total={total_params:,} trainable={trainable_params:,}")

    # Optimizer split:
    # - Adam for non-codebook parameters
    # - SGD for codebook embedding
    adam_params = []
    for m in [modules["projector"], modules["token_decoder"]]:
        adam_params.extend(list(m.parameters()))
    codebook_params = []
    if hasattr(modules["quantizer"], "codebook_parameters"):
        codebook_params = list(modules["quantizer"].codebook_parameters())
    codebook_param_ids = {id(p) for p in codebook_params}
    for p in modules["quantizer"].parameters():
        if id(p) not in codebook_param_ids:
            adam_params.append(p)

    optimizer = {
        "adam": torch.optim.Adam(
            adam_params,
            lr=cfg["run"].get("lr", 2e-4),
        )
    }
    codebook_lr = cfg["run"].get("codebook_lr", cfg["run"].get("lr", 2e-4))
    codebook_momentum = cfg["run"].get("codebook_momentum", 0.9)
    codebook_kmeans_every = int(cfg["run"].get("codebook_kmeans_every", 0))
    codebook_kmeans_iters = int(cfg["run"].get("codebook_kmeans_iters", 5))
    codebook_kmeans_samples = int(cfg["run"].get("codebook_kmeans_samples", 0))
    if isinstance(modules["quantizer"], (VQ, RVQ)):
        optimizer["codebook"] = torch.optim.SGD(
            codebook_params,
            lr=codebook_lr,
            momentum=codebook_momentum,
        )
        if codebook_kmeans_every > 0:
            print(
                f"[KMeans] Enabled for codebook: every={codebook_kmeans_every} steps, "
                f"iters={codebook_kmeans_iters}, sample_size={codebook_kmeans_samples}"
            )
    else:
        optimizer["codebook"] = None
        print("[Opt] Quantizer type without learned embedding codebook; SGD codebook optimizer disabled.")

    trainable = [modules["projector"], modules["quantizer"], modules["token_decoder"]]
    epochs = cfg["run"].get("epochs", 50)
    save_every = cfg["run"].get("save_every", 5)
    save_visuals = cfg["run"].get("save_visuals", True)
    global_step = 0

    for epoch in range(epochs):
        modules["extractor"].eval()
        for m in trainable:
            m.train()
        modules["quantizer"].train()

        epoch_metrics = []
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            global_step += 1
            attn, sel_idx, _, metrics, step_extra = run_step(
                batch, modules, optimizer, cfg, device, training=True
            )
            if (
                isinstance(modules["quantizer"], (VQ, RVQ))
                and codebook_kmeans_every > 0
                and global_step % codebook_kmeans_every == 0
            ):
                modules["quantizer"].kmeans_update_(
                    step_extra["z_enc_flat"],
                    num_iters=codebook_kmeans_iters,
                    sample_size=codebook_kmeans_samples,
                )
            epoch_metrics.append(metrics)
            if global_step % 10 == 0:
                torch.cuda.empty_cache()

        avg = {k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
               for k in epoch_metrics[0]}
        print(
            f"[Epoch {epoch+1}/{epochs}] loss={avg['loss']:.4f} "
            f"token={avg['token_loss']:.4f} token_mse_sel={avg['token_mse_sel']:.4f} "
            f"cos={avg['token_cos_sim']:.4f} commit={avg['commit_loss']:.4f} "
            f"tokens={avg['num_tokens']:.0f} "
            f"| IDX: {avg['indices_bits_per_token']:.0f}b/token {avg['indices_payload_bytes_per_image']:.1f}B/img "
            f"| Q: err_sel={avg['quant_error_sel']:.4f} util={avg['quant_utilization']:.3f} active={avg['quant_active_codes']:.0f} "
            f"perp={avg['quant_perplexity']:.1f} quant_err={avg['quant_error']:.4f}"
        )
        visualizer.log_metrics(global_step, avg)

        if (epoch + 1) % save_every == 0:
            if save_visuals:
                with torch.no_grad():
                    if batch.dim() in (4, 5):
                        sample = batch[:4].to(device)
                    else:
                        sample = batch.unsqueeze(0).to(device)
                    if sample.dim() == 5:
                        vis_image = sample[0:1, 0]
                    else:
                        vis_image = sample[0:1]
                    if sample.shape[0] > 4:
                        sample = sample[:4]
                    attn, sel_idx, _, _, extra = run_step(
                        sample, modules, None, cfg, device, training=False
                    )
                    nph, npw = extra["num_patches_h"], extra["num_patches_w"]

                    visualizer.save_attention_heatmap(
                        attn[0].cpu().numpy(),
                        save_name=f"attention_epoch{epoch+1}.png",
                        original_image=vis_image,
                    )
                    visualizer.save_token_mask(
                        selected_indices=sel_idx[0].cpu().numpy(),
                        num_patches=nph * npw,
                        save_name=f"selection_epoch{epoch+1}.png",
                    )
                    visualizer.save_token_metric_matrices(
                        per_token_mse_all=extra["per_token_mse_all"][0].cpu().numpy(),
                        vq_error_per_token=extra["vq_error_per_token"][0].cpu().numpy(),
                        selected_indices=sel_idx[0].cpu().numpy(),
                        num_patches_h=nph,
                        num_patches_w=npw,
                        save_name=f"token_matrices_epoch{epoch+1}.png",
                    )

            ckpt_dir = Path(log_dir) / run_name / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_obj = {
                "epoch": epoch + 1,
                "config": cfg,
                "projector": modules["projector"].state_dict(),
                "quantizer": modules["quantizer"].state_dict(),
                "token_decoder": modules["token_decoder"].state_dict(),
                "optimizer_adam": optimizer["adam"].state_dict(),
            }
            if optimizer.get("codebook") is not None:
                ckpt_obj["optimizer_codebook"] = optimizer["codebook"].state_dict()
            torch.save(
                ckpt_obj,
                ckpt_dir / f"ckpt_epoch{epoch+1}.pt",
            )

    print(f"[*] Training finished. Logs: {log_dir}/{run_name}")


# ---------------------------------------------------------------------------
# Test / evaluation loop
# ---------------------------------------------------------------------------

def run_test(cfg: dict, args, device: torch.device):
    log_dir = cfg["run"].get("log_dir", "logs")
    save_visuals = cfg["run"].get("save_visuals", True)
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    else:
        if not args.run_name:
            raise SystemExit("Test mode requires either --ckpt <path> "
                             "or --run_name <name> (and optional --ckpt_epoch).")
        ckpt_dir = Path(log_dir) / args.run_name / "checkpoints"
        if args.ckpt_epoch is not None:
            ckpt_path = ckpt_dir / f"ckpt_epoch{args.ckpt_epoch}.pt"
        else:
            ckpts = list(ckpt_dir.glob("ckpt_epoch*.pt"))
            if not ckpts:
                raise SystemExit(f"No checkpoints found in {ckpt_dir}")
            ckpt_path = max(ckpts, key=lambda p: p.stat().st_mtime)
            print(f"Using latest checkpoint: {ckpt_path}")
        if not ckpt_path.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if "config" in ckpt:
        cfg = ckpt["config"]

    modules = build_model(cfg, device)
    load_checkpoint(ckpt, modules, optimizer=None, device=device)
    modules["extractor"].eval()
    modules["projector"].eval()
    modules["quantizer"].eval()
    modules["token_decoder"].eval()

    test_root = cfg["data"].get("test_data_root") or cfg["data"]["data_root"]
    if getattr(args, "data_root", None):
        test_root = args.data_root
    try:
        loader, _ = get_dataloader(
            test_root,
            cfg["data"]["dataset"],
            cfg["data"]["batch_size"],
            cfg["data"].get("num_workers", 4),
            cfg["data"].get("image_size", 224),
            shuffle=False,
            multi_view=cfg["data"].get("multi_view", False),
        )
    except Exception as e:
        print(f"[WARN] Test DataLoader failed: {e}. Skipping test loop.")
        return

    all_metrics = []
    first_vis = None
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        with torch.no_grad():
            attn, sel_idx, _, metrics, extra = run_step(
                batch, modules, None, cfg, device, training=False
            )
        if save_visuals and first_vis is None:
            vis_image = batch[0:1].to(device)
            if vis_image.dim() == 5:
                vis_image = vis_image[:, 0]
            first_vis = {
                "image": vis_image,
                "attn": attn[0].cpu().numpy(),
                "sel_idx": sel_idx[0].cpu().numpy(),
                "nph": extra["num_patches_h"],
                "npw": extra["num_patches_w"],
                "per_token_mse_all": extra["per_token_mse_all"][0].cpu().numpy(),
                "vq_error_per_token": extra["vq_error_per_token"][0].cpu().numpy(),
            }
        all_metrics.append(metrics)

    avg = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
    print("[Test]", " ".join(f"{k}={v:.6f}" for k, v in avg.items()))
    run_name = args.run_name or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    visualizer = Visualizer(log_dir, run_name)
    visualizer.log_metrics(0, avg, filename="test_metrics.txt")
    if save_visuals and first_vis is not None:
        visualizer.save_attention_heatmap(
            first_vis["attn"],
            save_name="test_attention.png",
            original_image=first_vis["image"],
        )
        visualizer.save_token_mask(
            selected_indices=first_vis["sel_idx"],
            num_patches=first_vis["nph"] * first_vis["npw"],
            save_name="test_selection.png",
        )
        visualizer.save_token_metric_matrices(
            per_token_mse_all=first_vis["per_token_mse_all"],
            vq_error_per_token=first_vis["vq_error_per_token"],
            selected_indices=first_vis["sel_idx"],
            num_patches_h=first_vis["nph"],
            num_patches_w=first_vis["npw"],
            save_name="test_token_matrices.png",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Log dir name. Default: run_YYYYMMDD_HHMMSS (train) or test_... (test)",
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt_epoch", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("run", {}).get("seed") is not None:
        torch.manual_seed(cfg["run"]["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if args.mode == "train":
        run_train(cfg, args, device)
    else:
        run_test(cfg, args, device)


if __name__ == "__main__":
    main()
