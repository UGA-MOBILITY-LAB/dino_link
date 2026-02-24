from __future__ import annotations

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
        import sys
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from main import build_model as build_dinolink_model  # type: ignore

        import yaml
        with open(dinolink_cfg, "r") as f:
            cfg = yaml.safe_load(f)

        modules = build_dinolink_model(cfg, device=torch.device("cpu"))
        self.extractor = modules["extractor"]
        self.selector = modules["selector"]
        self.projector = modules["projector"]
        self.quantizer = modules["quantizer"]
        self.token_decoder = modules["token_decoder"]
        self.z_dim = int(modules["z_dim"])

        if dinolink_ckpt:
            ckpt = torch.load(dinolink_ckpt, map_location="cpu")
            if "projector" in ckpt:
                self.projector.load_state_dict(ckpt["projector"])
            if "quantizer" in ckpt:
                self.quantizer.load_state_dict(ckpt["quantizer"])
            if "token_decoder" in ckpt:
                self.token_decoder.load_state_dict(ckpt["token_decoder"])

        hidden_dim = int(self.detr.class_embed.in_features)
        token_dim = int(self.token_decoder.norm.normalized_shape[0])
        if token_dim == hidden_dim:
            self.token_proj = nn.Identity()
        else:
            self.token_proj = nn.Linear(token_dim, hidden_dim)

        if freeze_dinolink:
            for m in [self.extractor, self.projector, self.quantizer, self.token_decoder]:
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

    def _run_dinolink(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_tokens, attention_scores, nph, npw = self.extractor(img)
        selected, selected_indices = self.selector(
            patch_tokens, attention_scores, num_patches_h=nph, num_patches_w=npw
        )
        z_enc = self.projector(selected)
        z_q_st_flat, _, _ = self.quantizer(z_enc.reshape(-1, self.z_dim))
        b, k, _ = z_enc.shape
        z_q_st = z_q_st_flat.reshape(b, k, self.z_dim)

        rows = selected_indices // npw
        cols = selected_indices % npw
        row_norm = (rows.float() / max(nph - 1, 1)) * 2.0 - 1.0
        col_norm = (cols.float() / max(npw - 1, 1)) * 2.0 - 1.0
        patch_pos = torch.stack([row_norm, col_norm], dim=-1)
        decoded_tokens = self.token_decoder(z_q_st, patch_pos=patch_pos)
        return decoded_tokens, attention_scores

    def _fit_num_queries(self, tokens: torch.Tensor) -> torch.Tensor:
        # DETR criterion can work with any query count, but we keep the configured
        # query length for stable memory and checkpoint behavior.
        b, k, c = tokens.shape
        q = self.num_queries
        if k == q:
            return tokens
        if k > q:
            return tokens[:, :q, :]
        pad = tokens.new_zeros((b, q - k, c))
        return torch.cat([tokens, pad], dim=1)

    def _head_forward(self, tokens_3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        hs = self.token_proj(tokens_3d).unsqueeze(0)  # (1,B,Q,C)
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        out: Dict[str, torch.Tensor] = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        return out

    def forward(self, samples: NestedTensor | torch.Tensor | list[torch.Tensor]) -> Dict[str, torch.Tensor]:
        img = self._to_images(samples)
        decoded_tokens, _ = self._run_dinolink(img)
        query_tokens = self._fit_num_queries(decoded_tokens)
        return self._head_forward(query_tokens)


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

