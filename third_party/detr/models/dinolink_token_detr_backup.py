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
        TokenDecoder = _load_symbol("models/decoder.py", "TokenDecoder")

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

        quant_type = quant_cfg["type"]
        if quant_type == "rvq":
            self.quantizer = RVQ(
                quant_cfg["codebook_size"],
                self.z_dim,
                quant_cfg.get("num_quantizers", 2),
            )
        elif quant_type == "vq":
            self.quantizer = VQ(quant_cfg["codebook_size"], self.z_dim)
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
            if "quantizer" in ckpt:
                _safe_load_module(self.quantizer, ckpt["quantizer"], "quantizer")
            if "token_decoder" in ckpt:
                _safe_load_module(self.token_decoder, ckpt["token_decoder"], "token_decoder")

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

    def _head_forward(self, tokens_3d: torch.Tensor, samples: NestedTensor) -> Dict[str, torch.Tensor]:
        # 1. 准备 DinoLink Content Queries: [B, Q, C] -> [Q, B, C]
        content_queries = self.token_proj(tokens_3d).transpose(0, 1)
        
        # 2. 获取图像特征
        features, pos = self.detr.backbone(samples)
        src, mask = features[-1].decompose()
        
        # --- 关键修复：展平 2D 特征图为序列格式 ---
        # src: [B, C, H, W] -> [H*W, B, C]
        # pos: [B, C, H, W] -> [H*W, B, C]
        # mask: [B, H, W] -> [B, H*W]
        
        b, c, h, w = src.shape
        # input_proj 是 Conv2d，必须先在 4D 特征图上做通道映射
        src_proj_2d = self.detr.input_proj(src)
        src_proj = src_proj_2d.flatten(2).permute(2, 0, 1)
        pos_flattened = pos[-1].flatten(2).permute(2, 0, 1)
        mask_flattened = mask.flatten(1)
        # ---------------------------------------

        # 3. 调用 Transformer Encoder
        memory = self.detr.transformer.encoder(
            src_proj, 
            src_key_padding_mask=mask_flattened, 
            pos=pos_flattened
        )
        
        # 4. 准备 Position Queries: [Q, B, C]
        query_embed = self.detr.query_embed.weight.unsqueeze(1).repeat(1, b, 1)
        
        # 5. 调用 Transformer Decoder
        hs = self.detr.transformer.decoder(
            tgt=content_queries, 
            memory=memory,
            memory_key_padding_mask=mask_flattened,
            pos=pos_flattened,
            query_pos=query_embed
        )
        
        # hs: [Layers, Q, B, C] -> [Layers, B, Q, C]
        hs = hs.transpose(1, 2)

        # 6. 检测头输出
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = [{"pred_logits": a, "pred_boxes": b} 
                                 for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return out

    def forward(self, samples: NestedTensor | torch.Tensor | list[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
            
        img = self._to_images(samples)
        decoded_tokens, _ = self._run_dinolink(img)
        query_tokens = self._fit_num_queries(decoded_tokens)
        
        # 传入 samples 以获取 backbone 特征
        return self._head_forward(query_tokens, samples)
        
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

