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

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import util.misc as utils
from datasets import build_dataset
from models import build_model
from main import get_args_parser  # 复用官方的参数定义，避免缺少字段


# COCO 80 class names (index 0 = person, ...)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


def main():
    # 复用 main.py 里的 get_args_parser，确保和官方脚本参数完全一致
    parser = argparse.ArgumentParser('DETR prediction visualization', parents=[get_args_parser()])
    parser.add_argument('--vis_out_dir', type=str, default=None,
                        help='Directory to save visualization images (default: coco_path/vis_detr)')
    parser.add_argument('--score_thr', type=float, default=0.5, help='Min score to draw box')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval()

    if args.resume:
        if args.resume.startswith('https'):
            ckpt = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])

    dataset_val = build_dataset(image_set='val', args=args)
    data_loader = DataLoader(
        dataset_val, args.batch_size, shuffle=False,
        collate_fn=utils.collate_fn, num_workers=args.num_workers,
    )

    root = Path(args.coco_path)
    img_folder = root / "val2017"
    vis_dir = Path(args.vis_out_dir) if args.vis_out_dir else root / "vis_detr"
    vis_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = None

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
            draw = ImageDraw.Draw(img)

            for box, score, label in zip(boxes, scores, labels):
                if score < args.score_thr:
                    continue
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                cls_name = COCO_CLASSES[int(label)] if int(label) < len(COCO_CLASSES) else f"c{label}"
                text = f"{cls_name} {score:.2f}"
                if font:
                    draw.text((x1, y1 - 16), text, fill="red", font=font)
                else:
                    draw.text((x1, y1 - 14), text, fill="red")

            out_name = Path(file_name).name
            img.save(vis_dir / out_name)

    print(f"Saved visualizations to {vis_dir}")


if __name__ == "__main__":
    main()
