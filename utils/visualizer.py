"""
Token heatmap and similarity plot for DinoLink runs.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Saves token attention and token-level evaluation matrices."""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.log_dir / "figs"
        self.fig_dir.mkdir(exist_ok=True)

    def save_attention_heatmap(
        self,
        attention_scores: np.ndarray,
        save_name: str = "attention_heatmap.png",
        title: str = "CLS-to-token attention",
        normalize: bool = True,
        original_image=None,
    ):
        """
        attention_scores: (num_patches,) or (H, W). For DINOv2 256 -> 16x16.
        If normalize=True, min-max scale to [0,1] so contrast is visible (raw attn can be very flat).
        If original_image is provided (3, H, W) or (1, 3, H, W) in [0,1], draw it side-by-side with the heatmap.
        """
        if attention_scores.ndim == 1:
            n = len(attention_scores)
            side = int(np.sqrt(n))
            if side * side != n:
                side = int(np.ceil(np.sqrt(n)))
                pad = side * side - n
                attention_scores = np.pad(
                    attention_scores, (0, pad), mode="constant", constant_values=0
                )
            attn_2d = attention_scores[: side * side].reshape(side, side).astype(np.float64)
        else:
            attn_2d = np.asarray(attention_scores, dtype=np.float64)
        if normalize and attn_2d.size > 0:
            # Percentile-based scale: avoid an overall dark image and make relatively high-attention regions more visible (vmax=98th percentile)
            vmax = float(np.percentile(attn_2d, 98))
            vmin = float(attn_2d.min())
            if vmax > vmin:
                attn_2d = np.clip(attn_2d, vmin, vmax)
                attn_2d = (attn_2d - vmin) / (vmax - vmin)
            else:
                attn_2d = np.zeros_like(attn_2d)

        if original_image is not None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            img = original_image
            if img.dim() == 4:
                img = img[0]
            img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            axes[0].imshow(img)
            axes[0].set_title("Original")
            axes[0].axis("off")
            im = axes[1].imshow(attn_2d, cmap="viridis", vmin=0, vmax=1)
            axes[1].set_title(title)
            plt.colorbar(im, ax=axes[1])
            axes[1].axis("off")
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            im = ax.imshow(attn_2d, cmap="viridis", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_title(title)
        plt.tight_layout()
        path = self.fig_dir / save_name
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def save_token_mask(
        self,
        selected_indices: np.ndarray,
        num_patches: int = 256,
        save_name: str = "token_mask.png",
        title: str = "Selected tokens (top-k)",
    ):
        """
        selected_indices: (k,) — which patch indices are selected.
        Visualize as 16x16 grid with 1 where selected, 0 otherwise.
        """
        side = int(np.sqrt(num_patches))
        mask = np.zeros(num_patches, dtype=np.float32)
        mask[selected_indices] = 1.0
        mask_2d = mask.reshape(side, side)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(mask_2d, cmap="RdYlBu", vmin=0, vmax=1)
        ax.set_title(title)
        plt.tight_layout()
        path = self.fig_dir / save_name
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def save_token_metric_matrices(
        self,
        per_token_mse_all: np.ndarray,
        vq_error_per_token: np.ndarray,
        selected_indices: np.ndarray,
        num_patches_h: int,
        num_patches_w: int,
        save_name: str = "token_metric_matrices.png",
    ):
        """
        Save 2D token-level evaluation matrices:
          1) token reconstruction MSE
          2) quantization L2 error
          3) selected-token mask
        """
        total = num_patches_h * num_patches_w
        mse = np.asarray(per_token_mse_all[:total], dtype=np.float32).reshape(
            num_patches_h, num_patches_w
        )
        vq = np.asarray(vq_error_per_token[:total], dtype=np.float32).reshape(
            num_patches_h, num_patches_w
        )
        mask = np.zeros(total, dtype=np.float32)
        valid_sel = selected_indices[selected_indices < total]
        mask[valid_sel] = 1.0
        mask = mask.reshape(num_patches_h, num_patches_w)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        mse_vmax = max(float(np.max(mse)), 1e-6)
        im0 = axes[0].imshow(mse, cmap="hot", vmin=0, vmax=mse_vmax, interpolation="nearest")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_title(f"Token MSE (sel avg={float(mse.flat[valid_sel].mean()) if len(valid_sel) else 0.0:.4f})")
        axes[0].axis("off")

        vq_vmax = max(float(np.max(vq)), 1e-6)
        im1 = axes[1].imshow(vq, cmap="hot", vmin=0, vmax=vq_vmax, interpolation="nearest")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title(f"Quant L2 Error (sel avg={float(vq.flat[valid_sel].mean()) if len(valid_sel) else 0.0:.4f})")
        axes[1].axis("off")

        axes[2].imshow(mask, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        axes[2].set_title(f"Selection Mask ({len(valid_sel)}/{total})")
        axes[2].axis("off")

        plt.tight_layout()
        path = self.fig_dir / save_name
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def log_metrics(self, step: int, metrics: dict, filename: str = "metrics.txt"):
        """Append metrics to a log file."""
        path = self.log_dir / filename
        line = f"step={step} " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()) + "\n"
        with open(path, "a") as f:
            f.write(line)
