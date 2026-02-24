"""
DINOv2 token extractor: frozen backbone, returns patch tokens and CLS-to-patch attention.
Image size (224 or 448) is configurable via dinov2_image_size in config.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

DINOV2_PATCH_SIZE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2Extractor(nn.Module):
    """
    Wraps a frozen DINOv2 backbone. Forward returns:
    - patch_tokens: (B, num_patches, hidden_size)
    - attention_scores: (B, num_patches) — CLS-to-patch attention (mean over heads)
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        image_size: int | None = 224,
    ):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size  # None = variable size, no resize in forward
        self.patch_size = DINOV2_PATCH_SIZE
        if image_size is not None:
            self.num_patches_h = image_size // DINOV2_PATCH_SIZE
            self.num_patches_w = self.num_patches_h
            self.num_patches = self.num_patches_h * self.num_patches_w
        else:
            self.num_patches_h = self.num_patches_w = self.num_patches = None

        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            # Some transformers builds do not support `dinov2_with_registers`.
            # Fall back to the compatible non-registers backbone so downstream
            # integration can still run.
            if "with-registers" not in model_name:
                raise
            import warnings
            warnings.warn(
                f"transformers cannot load {model_name}; "
                "falling back to facebook/dinov2-base."
            )
            model_name = "facebook/dinov2-base"
            self.model_name = model_name
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_attentions = True
        self.backbone = AutoModel.from_pretrained(
            model_name, config=config, trust_remote_code=True
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.hidden_size = self.backbone.config.hidden_size
        # DINOv2 with Registers: 序列为 [CLS, reg1..regN, patch1..patchK]，只取 patch 部分
        self.num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W), range [0, 1]. If image_size is set, resized to that; else no resize (crop to multiple of 14).
        Returns:
            patch_tokens: (B, num_patches, hidden_size)
            attention_scores: (B, num_patches)
            num_patches_h, num_patches_w: int (for variable-size; when fixed, from self)
        """
        B, _, H, W = x.shape
        if self.image_size is not None:
            x_in = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            num_patches_h = self.num_patches_h
            num_patches_w = self.num_patches_w
        else:
            # No resize: crop to multiple of patch_size for backbone
            h_ = (H // self.patch_size) * self.patch_size
            w_ = (W // self.patch_size) * self.patch_size
            if h_ < self.patch_size or w_ < self.patch_size:
                x_in = F.interpolate(x, size=(max(h_, self.patch_size), max(w_, self.patch_size)), mode="bilinear", align_corners=False)
                h_, w_ = x_in.shape[2], x_in.shape[3]
            else:
                x_in = x[:, :, :h_, :w_].contiguous()
            num_patches_h = h_ // self.patch_size
            num_patches_w = w_ // self.patch_size
        x_in = (x_in - self._mean) / self._std
        out = self.backbone(
            pixel_values=x_in,
            output_attentions=True,
            return_dict=True,
        )
        last_hidden = out.last_hidden_state
        num_patches = num_patches_h * num_patches_w
        start = 1 + self.num_register_tokens
        end = start + num_patches
        patch_tokens = last_hidden[:, start:end, :]
        attentions = out.attentions
        if attentions is not None:
            # 最后 4 层: CLS(0) -> patch 部分 [start:end]，对层和 head 取平均
            n_layers = min(4, len(attentions))
            layer_scores = []
            for attn in attentions[-n_layers:]:
                cls_to_patch = attn[:, :, 0, start:end]
                layer_scores.append(cls_to_patch.mean(dim=1))
            attention_scores = torch.stack(layer_scores, dim=0).mean(dim=0)
            if not hasattr(self, "_warned_attn"):
                self._warned_attn = True
                print(f"[DINOv2] Using CLS-to-patch attention (last {n_layers} layers avg, mean over heads).")
        else:
            # Fallback: use L2 norm of patch tokens (no real attention from backbone)
            attention_scores = patch_tokens.norm(dim=2)
            if not hasattr(self, "_warned_attn"):
                self._warned_attn = True
                print("[DINOv2] attentions is None: using patch L2 norm (fallback).")
        return patch_tokens, attention_scores, num_patches_h, num_patches_w
