"""
Top-K or attention-based token selection. Arranges selected tokens in spatial order.
Grid size comes from extractor (224 -> 16x16, 448 -> 32x32).
"""
import torch


def topk_by_attention(
    attention_scores: torch.Tensor,
    k: int,
    num_patches_h: int,
    num_patches_w: int,
) -> torch.Tensor:
    """
    Get top-k patch indices by attention, then sort by spatial order (row-major).
    attention_scores: (B, num_patches)
    Returns: (B, k) indices in spatial order.
    """
    B, num_patches = attention_scores.shape
    topk_vals, topk_idx = attention_scores.topk(k, dim=1, largest=True, sorted=False)
    rows = topk_idx // num_patches_w
    cols = topk_idx % num_patches_w
    order = rows * num_patches_w + cols
    sorted_order = order.argsort(dim=1)
    topk_idx_spatial = torch.gather(topk_idx, 1, sorted_order)
    return topk_idx_spatial


def uniform_selection(
    num_patches: int,
    k: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Uniformly sample k patch indices (same for all batch). For ablation.
    Returns: (B, k) indices.
    """
    perm = torch.randperm(num_patches, device=device)[:k]
    perm = perm.sort()[0]
    return perm.unsqueeze(0).expand(batch_size, -1)


class TokenSelector:
    """
    Selects top-k tokens from DINOv2 patch sequence.
    mode: "attention" -> top-k by CLS-patch attention, sorted spatially
          "uniform" -> uniformly spaced / random k indices (ablation)
    num_patches_h, num_patches_w: from extractor (e.g. 16,16 for 224 or 32,32 for 448).
    """

    def __init__(
        self,
        top_k: int = 100,
        mode: str = "attention",
        num_patches_h: int = 16,
        num_patches_w: int = 16,
    ):
        self.top_k = top_k
        self.mode = mode
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.num_patches = self.num_patches_h * self.num_patches_w

    def __call__(
        self,
        patch_tokens: torch.Tensor,
        attention_scores: torch.Tensor,
        num_patches_h: int | None = None,
        num_patches_w: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        patch_tokens: (B, num_patches, hidden_size)
        attention_scores: (B, num_patches)
        num_patches_h, num_patches_w: optional override (e.g. for variable-size / no_resize).
        Returns:
            selected_tokens: (B, k, hidden_size)
            selected_indices: (B, k)
        """
        nh = num_patches_h if num_patches_h is not None else self.num_patches_h
        nw = num_patches_w if num_patches_w is not None else self.num_patches_w
        num_patches = nh * nw
        B = patch_tokens.shape[0]
        device = patch_tokens.device
        k = min(self.top_k, patch_tokens.shape[1])

        if self.mode == "attention":
            topk_idx = topk_by_attention(
                attention_scores, k, nh, nw
            )
        else:
            topk_idx = uniform_selection(
                num_patches, k, B, device
            )

        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(
            -1, -1, patch_tokens.size(-1)
        )
        selected = torch.gather(patch_tokens, 1, topk_idx_expanded)
        return selected, topk_idx
