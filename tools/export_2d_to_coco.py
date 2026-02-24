#!/usr/bin/env python3
"""
将 nuScenes devkit 导出的 2D 标注（image_annotations.json）转为 COCO 格式。

输入：image_annotations.json（每条含 filename, bbox_corners [x1,y1,x2,y2], category_name）
输出：COCO 格式 JSON（images, annotations, categories），bbox 为 [x, y, w, h]。
"""

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert export_2d image_annotations.json to COCO format."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="/home/tianle/dinolink_project/datasets/nuscenes/export_2d/image_annotations.json",
        help="Path to image_annotations.json.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output COCO JSON path (default: same dir as --json, name export_2d_coco.json).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1600,
        help="Image width for COCO images (nuScenes default).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="Image height for COCO images (nuScenes default).",
    )
    parser.add_argument(
        "--file-name-style",
        type=str,
        choices=("original", "basename"),
        default="original",
        help="original: file_name 使用原始路径 samples/CAM_FRONT/xxx.jpg; basename: prefix+文件名.",
    )
    parser.add_argument(
        "--file-name-prefix",
        type=str,
        default="images/",
        help="仅当 --file-name-style=basename 时有效; file_name = prefix + basename.",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {json_path}")

    out_path = Path(args.out) if args.out else json_path.parent / "export_2d_coco.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        ann_list = json.load(f)

    # 收集唯一 filename -> image_id（按出现顺序稳定）
    filename_to_id = OrderedDict()
    for ann in ann_list:
        fn = ann["filename"]
        if fn not in filename_to_id:
            filename_to_id[fn] = len(filename_to_id) + 1

    # 收集唯一 category_name -> category_id
    name_to_cat_id = OrderedDict()
    for ann in ann_list:
        name = ann.get("category_name", "unknown")
        if name not in name_to_cat_id:
            name_to_cat_id[name] = len(name_to_cat_id) + 1

    # COCO images
    images = []
    for fn, im_id in filename_to_id.items():
        if args.file_name_style == "original":
            file_name = fn
        else:
            base = os.path.basename(fn)
            file_name = (args.file_name_prefix + base) if args.file_name_prefix else base
        images.append({
            "id": im_id,
            "file_name": file_name,
            "width": args.width,
            "height": args.height,
        })

    # COCO categories
    categories = [
        {"id": cid, "name": name, "supercategory": "object"}
        for name, cid in name_to_cat_id.items()
    ]

    # COCO annotations: bbox_corners [x1,y1,x2,y2] -> bbox [x, y, w, h]
    annotations = []
    for ann_id, ann in enumerate(ann_list, 1):
        x1, y1, x2, y2 = ann["bbox_corners"]
        x = float(x1)
        y = float(y1)
        w = float(x2 - x1)
        h = float(y2 - y1)
        area = w * h
        image_id = filename_to_id[ann["filename"]]
        category_id = name_to_cat_id[ann.get("category_name", "unknown")]
        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": area,
            "iscrowd": 0,
        })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Converted: {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
