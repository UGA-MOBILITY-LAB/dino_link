# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Run DETR on a COCO-format dataset and save visualization images with predicted boxes.
Usage (same coco_path as eval):
  python vis_predictions.py --batch_size 2 --no_aux_loss --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/tianle/dinolink_project/datasets/nuscenes/export_2d_detr \
    --vis_out_dir /home/tianle/dinolink_project/datasets/nuscenes/export_2d/vis_detr
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

import datasets
import util.misc as utils
from datasets import build_dataset
from models import build_model
from main import get_args_parser  # 复用官方的参数定义，避免缺少字段

# High-contrast nuScenes-style palette (RGB tuples).
NUSC_COLOR_PALETTE = [
    (255, 99, 71),    # tomato
    (30, 144, 255),   # dodger blue
    (50, 205, 50),    # lime green
    (255, 165, 0),    # orange
    (147, 112, 219),  # medium purple
    (0, 206, 209),    # dark turquoise
    (255, 20, 147),   # deep pink
    (255, 215, 0),    # gold
    (0, 191, 255),    # deep sky blue
    (124, 252, 0),    # lawn green
    (255, 69, 0),     # orange red
    (138, 43, 226),   # blue violet
]


def class_color(class_idx: int):
    return NUSC_COLOR_PALETTE[class_idx % len(NUSC_COLOR_PALETTE)]


def main():
    # 复用 main.py 里的 get_args_parser，确保和官方脚本参数完全一致
    parser = argparse.ArgumentParser('DETR prediction visualization', parents=[get_args_parser()])
    parser.add_argument('--vis_out_dir', type=str, default=None,
                        help='Directory to save visualization images (default: coco_path/vis_detr)')
    parser.add_argument('--score_thr', type=float, default=0.5, help='Min score to draw box')
    parser.add_argument('--box_width', type=int, default=5, help='Bounding box line width')
    parser.add_argument('--font_size', type=int, default=30, help='Label font size')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval()

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        checkpoint_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

        # When using DinoLink wrapper, model keys are prefixed with "detr." for the inner DETR.
        # Only add "detr." prefix when loading a *plain* DETR checkpoint (no wrapper-specific keys),
        # to avoid corrupting checkpoints that were saved from DinoLinkTokenDETR.
        if getattr(args, "use_dinolink_tokens", False):
            model_keys = list(model.state_dict().keys())
            has_wrapped_keys = any(k.startswith("detr.") for k in model_keys)
            ckpt_keys = list(checkpoint_state.keys())
            has_plain_ckpt_keys = any(not k.startswith("detr.") for k in ckpt_keys)
            wrapper_prefixes = (
                "extractor.",
                "selector.",
                "projector.",
                "token_proj.",
                "token_decoder.",
                "quantizer.",
            )
            ckpt_has_wrapper_keys = any(k.startswith(wrapper_prefixes) for k in ckpt_keys)
            if has_wrapped_keys and has_plain_ckpt_keys and not ckpt_has_wrapper_keys:
                checkpoint_state = {f"detr.{k}": v for k, v in checkpoint_state.items()}
                print("Adjusted resume checkpoint keys by adding 'detr.' prefix for DinoLink wrapper.")

        try:
            model.load_state_dict(checkpoint_state)
        except RuntimeError as e:
            # Allow loading checkpoints with slightly different heads/backbones by
            # only loading shape-matched tensors (same strategy as training script).
            print(f"Strict checkpoint loading failed, falling back to shape-matched loading.\nError: {e}")
            model_state = model.state_dict()
            filtered_state = {
                k: v for k, v in checkpoint_state.items()
                if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape
            }
            skipped = sorted(set(checkpoint_state.keys()) - set(filtered_state.keys()))
            print(f"Loading shape-matched keys: {len(filtered_state)} / {len(model_state)}")
            print(f"Skipped checkpoint keys: {len(skipped)}")
            model.load_state_dict(filtered_state, strict=False)

    dataset_val = build_dataset(image_set='val', args=args)
    data_loader = DataLoader(
        dataset_val, args.batch_size, shuffle=False,
        collate_fn=utils.collate_fn, num_workers=args.num_workers,
    )

    # Build a mapping from contiguous label id (what DETR uses) to category name
    # using the SAME category id ordering as the training dataset.
    # Many DETR-style datasets expose `cat_ids` that define this order.
    if hasattr(dataset_val, "cat_ids"):
        cat_ids = list(dataset_val.cat_ids)
    else:
        # Fallback: derive from COCO cat ids (may differ if a custom dataset changes the order)
        cat_ids = sorted(dataset_val.coco.cats.keys())
    id2name = {
        idx: dataset_val.coco.cats[cat_id]["name"]
        for idx, cat_id in enumerate(cat_ids)
    }

    root = Path(args.coco_path)
    img_folder = root / "val2017"
    vis_dir = Path(args.vis_out_dir) if args.vis_out_dir else root / "vis_detr"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(data_loader, desc="Vis"):
        samples, targets = batch
        samples = samples.to(device)
        # 将 target 也移到同一 device，保持与 engine.evaluate 一致，避免 device mismatch
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        with torch.no_grad():
            outputs = model(samples)
        results = postprocessors['bbox'](outputs, orig_sizes)

        for i, (target, res) in enumerate(zip(targets, results)):
            image_id = target["image_id"].item()
            boxes = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            labels = res["labels"].cpu().numpy()

            info = dataset_val.coco.loadImgs(image_id)[0]
            file_name = info["file_name"]
            img_path = img_folder / file_name
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")

            # Filter by score and build label strings
            keep = scores >= args.score_thr
            boxes_keep = boxes[keep]
            scores_keep = scores[keep]
            labels_keep = labels[keep]

            if len(boxes_keep) == 0:
                img.save(vis_dir / Path(file_name).name)
                continue

            # Official torchvision drawing: image (C,H,W) uint8, boxes (N,4) xyxy
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)
            boxes_tensor = torch.as_tensor(boxes_keep, dtype=torch.float)
            label_strs = []
            for score, label in zip(scores_keep, labels_keep):
                lbl_idx = max(int(label) - 1, 0)
                cls_name = id2name.get(lbl_idx, f"c{lbl_idx}")
                label_strs.append(f"{cls_name} {score:.2f}")

            out_tensor = draw_bounding_boxes(
                img_tensor,
                boxes_tensor,
                labels=label_strs,
                colors="red",
                width=4,
                font_size=22,
            )
            out_pil = Image.fromarray(out_tensor.permute(1, 2, 0).numpy())
            out_pil.save(vis_dir / Path(file_name).name)

    print(f"Saved visualizations to {vis_dir}")


if __name__ == "__main__":
    main()
