from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from util.misc import NestedTensor, nested_tensor_from_tensor_list

from .detr import build as build_detr


class DinoLinkTokenDETR(nn.Module):
    """
    Pipeline:
      image -> DinoLink decoder tokens -> DETR detection heads
    """

    def __init__(
        self,
        detr_model: nn.Module,
        *,
        dinolink_cfg: str,
        dinolink_ckpt: Optional[str] = None,
        freeze_dinolink: bool = True,
    ):
        super().__init__()
        self.detr = detr_model
        self.num_queries = int(getattr(detr_model, "num_queries", 100))
        self.aux_loss = bool(getattr(detr_model, "aux_loss", False))

        # Make project root importable so we can reuse DinoLink modules.
        project_root = Path(__file__).resolve().parents[3]
        import importlib.util

        def _load_symbol(module_rel_path: str, symbol: str):
            module_path = project_root / module_rel_path
            spec = importlib.util.spec_from_file_location(
                f"dinolink_{module_path.stem}", module_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load module from {module_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, symbol)

        DINOv2Extractor = _load_symbol("models/dinov2_extractor.py", "DINOv2Extractor")
        TokenSelector = _load_symbol("models/token_selector.py", "TokenSelector")
        Projector = _load_symbol("models/projector.py", "Projector")
        VQ = _load_symbol("models/quantizer.py", "VQ")
        RVQ = _load_symbol("models/quantizer.py", "RVQ")
        vq_effectiveness = _load_symbol("models/quantizer.py", "vq_effectiveness")
        TokenDecoder = _load_symbol("models/decoder.py", "TokenDecoder")
        self._vq_effectiveness_fn = vq_effectiveness

        import yaml
        with open(dinolink_cfg, "r") as f:
            cfg = yaml.safe_load(f)

        model_cfg = cfg["model"]
        quant_cfg = cfg["quantizer"]
        dec_cfg = cfg.get("decoder", {})

        top_k = int(model_cfg["top_k"])
        selection = model_cfg.get("selection", "attention")
        dinov2_name = model_cfg["dinov2_name"]
        dinov2_image_size = model_cfg.get("dinov2_image_size", 224)

        self.extractor = DINOv2Extractor(
            dinov2_name,
            image_size=dinov2_image_size,
        )
        nh = self.extractor.num_patches_h if self.extractor.num_patches_h is not None else 16
        nw = self.extractor.num_patches_w if self.extractor.num_patches_w is not None else 16
        hidden_size = int(self.extractor.hidden_size)

        self.selector = TokenSelector(
            top_k=top_k,
            mode=selection,
            num_patches_h=nh,
            num_patches_w=nw,
        )

        self.z_dim = int(quant_cfg["z_dim"])
        self.projector = Projector(hidden_size, self.z_dim)

        quant_type = str(quant_cfg["type"]).lower()
        if quant_type == "rvq":
            self.quantizer = RVQ(
                quant_cfg["codebook_size"],
                self.z_dim,
                quant_cfg.get("num_quantizers", 2),
            )
        elif quant_type == "vq":
            self.quantizer = VQ(quant_cfg["codebook_size"], self.z_dim)
        elif quant_type in {"none", "identity", "no_vq"}:
            self.quantizer = None
        else:
            raise ValueError(f"Unsupported quantizer.type: {quant_type}")

        self.token_decoder = TokenDecoder(
            z_dim=self.z_dim,
            out_dim=hidden_size,
            hidden_dim=dec_cfg.get("hidden_dim", 512),
            use_pos=dec_cfg.get("use_pos", True),
        )

        def _safe_load_module(module: nn.Module, state_dict: Dict[str, torch.Tensor], name: str) -> None:
            """Load only shape-matching tensors so mismatched ckpts can still run."""
            current = module.state_dict()
            matched: Dict[str, torch.Tensor] = {}
            skipped = []
            for k, v in state_dict.items():
                if k in current and current[k].shape == v.shape:
                    matched[k] = v
                else:
                    skipped.append(k)
            missing, unexpected = module.load_state_dict(matched, strict=False)
            if skipped:
                print(f"[DinoLinkTokenDETR] {name}: skipped {len(skipped)} mismatched keys.")
            if missing:
                print(f"[DinoLinkTokenDETR] {name}: missing {len(missing)} keys after partial load.")
            if unexpected:
                print(f"[DinoLinkTokenDETR] {name}: unexpected {len(unexpected)} keys in ckpt.")

        if dinolink_ckpt:
            ckpt = torch.load(dinolink_ckpt, map_location="cpu", weights_only=False)
            if "projector" in ckpt:
                _safe_load_module(self.projector, ckpt["projector"], "projector")
            if self.quantizer is not None and "quantizer" in ckpt:
                _safe_load_module(self.quantizer, ckpt["quantizer"], "quantizer")
            if "token_decoder" in ckpt:
                _safe_load_module(self.token_decoder, ckpt["token_decoder"], "token_decoder")

        hidden_dim = int(self.detr.class_embed.in_features)
        token_dim = int(self.token_decoder.norm.normalized_shape[0])
        if token_dim == hidden_dim:
            self.token_proj = nn.Identity()
        else:
            self.token_proj = nn.Linear(token_dim, hidden_dim)
        self._pos_num_feats = hidden_dim // 2
        self._pos_temperature = 10000

        if freeze_dinolink:
            modules_to_freeze = [self.extractor, self.projector, self.token_decoder]
            if self.quantizer is not None:
                modules_to_freeze.append(self.quantizer)
            for m in modules_to_freeze:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    @staticmethod
    def _to_images(samples: NestedTensor | torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(samples, NestedTensor):
            return samples.tensors
        if isinstance(samples, list):
            return nested_tensor_from_tensor_list(samples).tensors
        if torch.is_tensor(samples):
            return samples
        raise ValueError(f"Unsupported samples type: {type(samples)}")

    def _run_dinolink(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, Dict[str, float]]:
        # DETR data pipeline already applies ImageNet normalization.
        # DINOv2 extractor also normalizes internally → undo DETR's to avoid double-norm.
        img = img * self.extractor._std + self.extractor._mean
        patch_tokens, attention_scores, nph, npw = self.extractor(img)
        selected, selected_indices = self.selector(
            patch_tokens, attention_scores, num_patches_h=nph, num_patches_w=npw
        )
        z_enc = self.projector(selected)
        b, k, _ = z_enc.shape
        if self.quantizer is None:
            # No-VQ mode: pass projector output directly to TokenDecoder.
            z_q_st = z_enc
            vq_eval = {
                "enabled": 0.0,
                "quant_error": 0.0,
                "quant_utilization": 0.0,
                "quant_perplexity": 0.0,
                "quant_active_codes": 0.0,
            }
        else:
            z_q_st_flat, indices, quant_stats = self.quantizer(z_enc.reshape(-1, self.z_dim))
            z_q_st = z_q_st_flat.reshape(b, k, self.z_dim)
            codebook_size = int(getattr(self.quantizer, "codebook_size", 0))
            vq_eval = self._vq_effectiveness_fn(
                indices=indices.reshape(-1),
                z_enc=quant_stats["z_enc"].reshape(-1, self.z_dim).detach(),
                z_q=quant_stats["z_q_raw"].reshape(-1, self.z_dim).detach(),
                codebook_size=codebook_size,
            )
            vq_eval["enabled"] = 1.0

        num_patches_total = nph * npw
        bits_pos = math.ceil(math.log2(max(num_patches_total, 2)))
        if self.quantizer is not None:
            cb_size = int(getattr(self.quantizer, "codebook_size", 2))
            num_q = int(getattr(self.quantizer, "num_quantizers", 1))
            bits_code = math.ceil(math.log2(max(cb_size, 2))) * num_q
            vq_eval["transmission_bits"] = float(k * (bits_code + bits_pos))
        else:
            bits_token = self.z_dim * 32
            vq_eval["transmission_bits"] = float(k * (bits_token + bits_pos))

        # Extra fields for visualization/debugging (ignored by loss & metrics helpers).
        vq_eval["nph"] = float(nph)
        vq_eval["npw"] = float(npw)
        vq_eval["selected_indices"] = selected_indices.detach()

        return z_q_st, selected_indices, nph, npw, vq_eval

    @staticmethod
    def _now_s(device_like) -> float:
        if torch.is_tensor(device_like):
            device = device_like.device
        elif isinstance(device_like, torch.device):
            device = device_like
        else:
            device = torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    # NOTE: Dead code (kept commented for reference).
    # The current token-only path feeds ALL DINO tokens (K can be 1024) into the
    # encoder memory and uses num_queries only for decoder queries, so we no
    # longer trim/pad tokens to num_queries.
    #
    # def _fit_num_queries(self, tokens: torch.Tensor) -> torch.Tensor:
    #     # DETR criterion can work with any query count, but we keep the configured
    #     # query length for stable memory and checkpoint behavior.
    #     b, k, c = tokens.shape
    #     q = self.num_queries
    #     if k == q:
    #         return tokens
    #     if k > q:
    #         return tokens[:, :q, :]
    #     pad = tokens.new_zeros((b, q - k, c))
    #     return torch.cat([tokens, pad], dim=1)
    #
    # def _fit_num_queries_indices(
    #     self, selected_indices: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     b, k = selected_indices.shape
    #     q = self.num_queries
    #     selected_indices = selected_indices.long()
    #     if k == q:
    #         mask = torch.zeros((b, q), dtype=torch.bool, device=selected_indices.device)
    #         return selected_indices, mask
    #     if k > q:
    #         trimmed = selected_indices[:, :q]
    #         mask = torch.zeros((b, q), dtype=torch.bool, device=selected_indices.device)
    #         return trimmed, mask
    #     pad = torch.zeros((b, q - k), dtype=torch.long, device=selected_indices.device)
    #     fitted = torch.cat([selected_indices, pad], dim=1)
    #     valid = torch.cat(
    #         [
    #             torch.ones((b, k), dtype=torch.bool, device=selected_indices.device),
    #             torch.zeros((b, q - k), dtype=torch.bool, device=selected_indices.device),
    #         ],
    #         dim=1,
    #     )
    #     # Transformer key padding mask: True means ignore that token.
    #     padding_mask = ~valid
    #     return fitted, padding_mask


    # def _head_forward(self, tokens_3d: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     hs = self.token_proj(tokens_3d).unsqueeze(0)  # (1,B,Q,C)
    #     outputs_class = self.detr.class_embed(hs)
    #     outputs_coord = self.detr.bbox_embed(hs).sigmoid()
    #     out: Dict[str, torch.Tensor] = {
    #         "pred_logits": outputs_class[-1],
    #         "pred_boxes": outputs_coord[-1],
    #     }
    #     if self.aux_loss:
    #         out["aux_outputs"] = [
    #             {"pred_logits": a, "pred_boxes": b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
    #         ]
    #     return out

    # def forward(self, samples: NestedTensor | torch.Tensor | list[torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     img = self._to_images(samples)
    #     decoded_tokens, _ = self._run_dinolink(img)
    #     query_tokens = self._fit_num_queries(decoded_tokens)
    #     return self._head_forward(query_tokens)

    # def _head_forward(
    #     self,
    #     tokens_3d: torch.Tensor,
    #     selected_indices: torch.Tensor,
    #     nph: int,
    #     npw: int,
    # ) -> Dict[str, torch.Tensor]:
    #     # 1. Build the encoder's src/pos/mask from DINO tokens
    #     b = tokens_3d.shape[0]
    #     src_proj = self.token_proj(tokens_3d).transpose(0, 1)  # [Q, B, C]
    #     fitted_indices, mask_flattened = self._fit_num_queries_indices(selected_indices)

    #     rows = fitted_indices // npw
    #     cols = fitted_indices % npw
    #     row_norm = (rows.float() / max(nph - 1, 1)) * 2.0 - 1.0
    #     col_norm = (cols.float() / max(npw - 1, 1)) * 2.0 - 1.0
    #     pos_2d = torch.stack([row_norm, col_norm], dim=-1)  # [B, Q, 2]
    #     pos_flattened = self.memory_pos_proj(pos_2d).transpose(0, 1)  # [Q, B, C]

    #     # 2. Transformer encoder
    #     memory = self.detr.transformer.encoder(
    #         src_proj,
    #         src_key_padding_mask=mask_flattened,
    #         pos=pos_flattened,
    #     )

    #     # 3. Position queries: [Q, B, C]
    #     query_embed = self.detr.query_embed.weight.unsqueeze(1).repeat(1, b, 1)
    #     tgt = torch.zeros_like(query_embed)

    #     # 4. Transformer decoder (tgt starts from all zeros)
    #     hs = self.detr.transformer.decoder(
    #         tgt=tgt,
    #         memory=memory,
    #         memory_key_padding_mask=mask_flattened,
    #         pos=pos_flattened,
    #         query_pos=query_embed,
    #     )
    def _sinusoidal_pe(self, row_norm: torch.Tensor, col_norm: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal PE identical to DETR's PositionEmbeddingSine.

        Args:
            row_norm: [B, Q] in [-1, 1]
            col_norm: [B, Q] in [-1, 1]
        Returns:
            pos: [Q, B, C]
        """
        scale = 2 * math.pi
        y_embed = ((row_norm + 1.0) / 2.0) * scale  # [0, 2π]
        x_embed = ((col_norm + 1.0) / 2.0) * scale

        dim_t = torch.arange(self._pos_num_feats, dtype=torch.float32, device=row_norm.device)
        dim_t = self._pos_temperature ** (2 * (dim_t // 2) / self._pos_num_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t  # [B, Q, D/2]
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)

        return torch.cat([pos_y, pos_x], dim=-1).transpose(0, 1)  # [Q, B, C]

    def _head_forward(
        self,
        z_q_st: torch.Tensor,
        selected_indices: torch.Tensor,
        nph: int,
        npw: int,
        vq_eval: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_q_st: [B, K, z_dim]               — quantized (or identity) latent tokens
            selected_indices: [B, K]           — their 1-D patch indices
        """
        b, k, _ = z_q_st.shape

        # Server-side decode stage: latent tokens -> DINO token space.
        rows = selected_indices // npw
        cols = selected_indices % npw
        row_norm = (rows.float() / max(nph - 1, 1)) * 2.0 - 1.0
        col_norm = (cols.float() / max(npw - 1, 1)) * 2.0 - 1.0
        patch_pos = torch.stack([row_norm, col_norm], dim=-1)
        decoded_tokens = self.token_decoder(z_q_st, patch_pos=patch_pos)

        # 1. Encoder memory: ALL K DINO tokens → [K, B, hidden_dim]
        src_proj = self.token_proj(decoded_tokens).transpose(0, 1)

        # 2. Position encoding: sinusoidal PE for every token position
        selected_indices = selected_indices.long()
        if k > selected_indices.shape[1]:
            selected_indices = selected_indices[:, :k]
        elif k < selected_indices.shape[1]:
            selected_indices = selected_indices[:, :k]
        rows = selected_indices // npw
        cols = selected_indices % npw
        row_norm = (rows.float() / max(nph - 1, 1)) * 2.0 - 1.0
        col_norm = (cols.float() / max(npw - 1, 1)) * 2.0 - 1.0
        pos_flattened = self._sinusoidal_pe(row_norm, col_norm)  # [K, B, C]

        # No padding mask needed — all K tokens are real
        mask_flattened = torch.zeros((b, k), dtype=torch.bool, device=decoded_tokens.device)

        # 3. Transformer encoder: K tokens attend to each other
        memory = self.detr.transformer.encoder(
            src_proj,
            src_key_padding_mask=mask_flattened,
            pos=pos_flattened,
        )

        # 4. Decoder queries: num_queries (e.g. 30) independent of K
        query_embed = self.detr.query_embed.weight.unsqueeze(1).repeat(1, b, 1)
        tgt = torch.zeros_like(query_embed)

        # 5. Transformer decoder: 30 queries attend to K memory tokens
        hs = self.detr.transformer.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=mask_flattened,
            pos=pos_flattened,
            query_pos=query_embed,
        )

        hs = hs.transpose(1, 2)

        # 6. Detection heads
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if vq_eval is not None:
            out["vq_eval"] = vq_eval
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        return out

    def forward(self, samples: NestedTensor | torch.Tensor | list[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not getattr(self, "_dinolink_path_printed", False):
            print("[DinoLinkTokenDETR] Forward path: image -> DinoLink -> DETR encoder/decoder -> detection heads.")
            self._dinolink_path_printed = True
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        img = self._to_images(samples)
        timed = not self.training
        if timed:
            t0 = self._now_s(img)
        z_q_st, selected_indices, nph, npw, vq_eval = self._run_dinolink(img)
        if timed:
            t1 = self._now_s(img)
        out = self._head_forward(z_q_st, selected_indices, nph, npw, vq_eval=vq_eval)
        if timed:
            t2 = self._now_s(img)
            edge_compute_s = float(t1 - t0)
            server_compute_s = float(t2 - t1)
            # All-local compute is edge stage + server-stage compute on one device.
            vq_eval["edge_compute_s"] = edge_compute_s
            vq_eval["server_compute_s"] = server_compute_s
            vq_eval["all_local_compute_s"] = edge_compute_s + server_compute_s
        return out
        
def build_dinolink_token_model(args):
    if getattr(args, "masks", False):
        raise ValueError("DinoLink token path currently supports box detection only (no --masks).")
    if not getattr(args, "dinolink_cfg", None):
        raise ValueError("--dinolink_cfg is required when --use_dinolink_tokens is enabled.")

    detr_model, criterion, postprocessors = build_detr(args)
    model = DinoLinkTokenDETR(
        detr_model,
        dinolink_cfg=args.dinolink_cfg,
        dinolink_ckpt=getattr(args, "dinolink_ckpt", None),
        freeze_dinolink=bool(getattr(args, "freeze_dinolink", True)),
    )
    return model, criterion, postprocessors

