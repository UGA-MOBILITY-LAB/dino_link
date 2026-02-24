#!/usr/bin/env python3
"""
可视化 nuScenes devkit 导出的 2D 标注（image_annotations.json）。

只依赖：
- image_annotations.json（含 filename, bbox_corners, category_name）
- 一个实际存放图片的目录（按 basename 匹配），例如 datasets/nuscenes/export_2d/images
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict

from PIL import Image, ImageDraw, ImageFont


def index_images_by_basename(root: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    root = os.path.abspath(root)
    for r, _dirs, files in os.walk(root):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            mapping[f] = os.path.join(r, f)
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 2D annotations from image_annotations.json by matching basenames in an images dir."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="/home/tianle/dinolink_project/datasets/nuscenes/export_2d/image_annotations.json",
        help="Path to image_annotations.json.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory where images actually live (will be searched recursively, match by basename).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/tianle/dinolink_project/datasets/nuscenes/export_2d/vis_2d",
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Max number of images to visualize (-1 for all).",
    )
    args = parser.parse_args()

    with open(args.json, "r") as f:
        ann_list = json.load(f)

    # 按 basename 分组，避免依赖原始路径（samples/CAM_FRONT/...）
    anns_by_base = defaultdict(list)
    for ann in ann_list:
        base = os.path.basename(ann["filename"])
        bbox = ann["bbox_corners"]
        cat = ann.get("category_name", "")
        anns_by_base[base].append((bbox, cat))

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Indexing images under {args.images_dir} ...")
    basename_to_path = index_images_by_basename(args.images_dir)
    print(f"Found {len(basename_to_path)} images in source dir.")

    items = sorted(anns_by_base.items())
    if args.max_images > 0:
        items = items[: args.max_images]

    print(f"Have annotations for {len(anns_by_base)} images, will visualize {len(items)} images.")

    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        pass

    for idx, (base, boxes) in enumerate(items, 1):
        img_path = basename_to_path.get(base)
        if img_path is None or not os.path.exists(img_path):
            print(f"[{idx}] Skip missing image: {base}")
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for bbox, cat in boxes:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            if cat:
                text = str(cat)
                if font:
                    draw.text((x1 + 2, y1 + 2), text, fill="yellow", font=font)
                else:
                    draw.text((x1 + 2, y1 + 2), text, fill="yellow")

        out_path = os.path.join(args.out_dir, base)
        img.save(out_path)
        print(f"[{idx}] Saved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
可视化 nuScenes devkit 导出的 2D 标注（export_2d_annotations_as_json.py 的输出）。

读取 image_annotations.json，按照其中的 filename + bbox_corners 画框，
并将可视化结果保存到指定输出目录。
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


def _index_images_by_basename(images_dir: str):
    """Build a map from basename -> full path under images_dir (one level or recursive)."""
    out = {}
    images_dir = os.path.abspath(images_dir)
    for root, _dirs, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                path = os.path.join(root, f)
                # 重名时后者覆盖，通常 basename 在 nuScenes 里唯一
                out[f] = path
    return out


def visualize_2d_annotations(
    json_path: str,
    dataroot: str,
    out_dir: str,
    max_images: int = -1,
    images_dir: Optional[str] = None,
) -> None:
    with open(json_path, "r") as f:
        ann_list = json.load(f)

    # 按图片分组
    anns_by_file = defaultdict(list)
    for ann in ann_list:
        filename = ann["filename"]  # 例如 samples/CAM_FRONT/xxx.jpg
        bbox = ann["bbox_corners"]  # [x1, y1, x2, y2]
        cat = ann.get("category_name", "")
        anns_by_file[filename].append((bbox, cat))

    os.makedirs(out_dir, exist_ok=True)

    # 若指定了 images_dir，建立 basename -> 绝对路径 的索引
    basename_to_path = None
    if images_dir and os.path.isdir(images_dir):
        basename_to_path = _index_images_by_basename(images_dir)
        print(f"Images-dir: found {len(basename_to_path)} images under {images_dir} (lookup by basename).")

    # 排序保证可重复
    items = sorted(anns_by_file.items())
    if max_images > 0:
        items = items[:max_images]

    print(f"Found {len(anns_by_file)} images in JSON, will visualize {len(items)} images.")

    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        pass

    for idx, (rel_path, boxes) in enumerate(items, 1):
        img_path = os.path.join(dataroot, rel_path)
        if not os.path.exists(img_path) and basename_to_path is not None:
            name = os.path.basename(rel_path)
            img_path = basename_to_path.get(name, img_path)
        if not os.path.exists(img_path):
            print(f"[{idx}] Skip missing image: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for bbox, cat in boxes:
            x1, y1, x2, y2 = bbox
            # 画框
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            # 画类别文本
            if cat:
                text = str(cat)
                if font:
                    draw.text((x1 + 2, y1 + 2), text, fill="yellow", font=font)
                else:
                    draw.text((x1 + 2, y1 + 2), text, fill="yellow")

        # 输出文件名沿用原始文件名结构，放到 out_dir 中
        out_path = os.path.join(out_dir, os.path.basename(rel_path))
        img.save(out_path)
        print(f"[{idx}] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 2D annotations from export_2d_annotations_as_json.py output."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="/home/tianle/dinolink_project/datasets/nuscenes/export_2d/image_annotations.json",
        help="Path to image_annotations.json exported by nuScenes devkit.",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default="/mnt/datasets/nuScenes",
        help="nuScenes dataroot (same as --dataroot used for export_2d_annotations_as_json.py).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/tianle/dinolink_project/datasets/nuscenes/export_2d/vis_2d",
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Max number of images to visualize (-1 for all).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional: directory where images actually live (e.g. val_500/images). Lookup by filename when dataroot path is missing.",
    )
    args = parser.parse_args()

    visualize_2d_annotations(
        args.json,
        args.dataroot,
        args.out_dir,
        args.max_images,
        args.images_dir,
    )


if __name__ == "__main__":
    main()

