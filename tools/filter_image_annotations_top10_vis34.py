#!/usr/bin/env python3
"""
从 image_annotations.json 中只保留：
  - visibility_token 为 "3" 或 "4"
  - category 属于 10 类（car, pedestrian, barrier, truck, trafficcone, construction, motorcycle, bus, bicycle, trailer），
    并将 category_name 统一为上述短名。

输出：
  1) 过滤后的列表 JSON（与输入格式相同）。
  2) 若指定 --coco-dir：在该目录下生成 COCO 格式 train/val 划分（annotations/instances_*.json），
     并可选的 train2017/、val2017/ 图片目录（从 --images-dir 按 basename 软链或复制，见 --copy-images）。
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

# 固定 10 类顺序（与 export_2d_*_top10_vis34 一致），id 1~10
COCO_TOP10_CATEGORIES = [
    {"id": 1, "name": "car", "supercategory": "object"},
    {"id": 2, "name": "pedestrian", "supercategory": "object"},
    {"id": 3, "name": "barrier", "supercategory": "object"},
    {"id": 4, "name": "truck", "supercategory": "object"},
    {"id": 5, "name": "trafficcone", "supercategory": "object"},
    {"id": 6, "name": "construction", "supercategory": "object"},
    {"id": 7, "name": "motorcycle", "supercategory": "object"},
    {"id": 8, "name": "bus", "supercategory": "object"},
    {"id": 9, "name": "bicycle", "supercategory": "object"},
    {"id": 10, "name": "trailer", "supercategory": "object"},
]
SHORT_NAME_TO_CID = {c["name"]: c["id"] for c in COCO_TOP10_CATEGORIES}

# nuScenes category_name 前缀 -> 10 类短名（与 export_2d_train_coco_second_level_top10_vis34 一致）
NUSCENES_TO_TOP10 = [
    ("vehicle.car", "car"),
    ("human.pedestrian", "pedestrian"),
    ("static_object.barrier", "barrier"),
    ("vehicle.truck", "truck"),
    ("movable_object.traffic_cone", "trafficcone"),
    ("static_object.construction", "construction"),
    ("vehicle.motorcycle", "motorcycle"),
    ("vehicle.bus", "bus"),
    ("vehicle.bicycle", "bicycle"),
    ("vehicle.trailer", "trailer"),
]


def map_category(name: str) -> Optional[str]:
    """若属于 10 类之一则返回短名，否则返回 None（应丢弃）。
    支持：1) 已是短名（car, bus, ...）；2) nuScenes 长名（vehicle.car, human.pedestrian, ...）。
    """
    if not name:
        return None
    if name in SHORT_NAME_TO_CID:
        return name
    for prefix, short in NUSCENES_TO_TOP10:
        if name.startswith(prefix):
            return short
    return None


def _build_coco_split(
    filtered: List[dict],
    filename_order: List[str],
    width: int,
    height: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """从过滤后的列表和本 split 的 filename 顺序构建 COCO images/annotations。"""
    fn_to_im_id = {fn: (i + 1) for i, fn in enumerate(filename_order)}
    images = []
    for fn, im_id in fn_to_im_id.items():
        base = os.path.basename(fn)
        images.append({
            "id": im_id,
            "file_name": base,
            "width": width,
            "height": height,
        })
    annotations = []
    ann_id = 0
    for ann in filtered:
        fn = ann["filename"]
        if fn not in fn_to_im_id:
            continue
        ann_id += 1
        x1, y1, x2, y2 = ann["bbox_corners"]
        x, y = float(x1), float(y1)
        w, h = float(x2 - x1), float(y2 - y1)
        annotations.append({
            "id": ann_id,
            "image_id": fn_to_im_id[fn],
            "category_id": SHORT_NAME_TO_CID[ann["category_name"]],
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
        })
    return images, annotations, COCO_TOP10_CATEGORIES


def _write_coco_and_link(
    filtered: List[dict],
    coco_dir: Path,
    train_ratio: float,
    seed: Optional[int],
    width: int,
    height: int,
    images_dir: Optional[Path],
    copy_images: bool = False,
) -> None:
    """划分 train/val，写 annotations/instances_*.json，可选建 train2017/val2017（软链或复制）。"""
    unique_fns = list({ann["filename"] for ann in filtered})
    if seed is not None:
        random.seed(seed)
    random.shuffle(unique_fns)
    n = len(unique_fns)
    n_train = max(1, int(n * train_ratio))
    train_fns = unique_fns[:n_train]
    val_fns = unique_fns[n_train:]

    ann_dir = coco_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for split_name, fns in [("train2017", train_fns), ("val2017", val_fns)]:
        images, annotations, categories = _build_coco_split(
            filtered, fns, width, height
        )
        out_json = ann_dir / f"instances_{split_name}.json"
        coco = {"images": images, "annotations": annotations, "categories": categories}
        with open(out_json, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"  {out_json}: {len(images)} images, {len(annotations)} annotations.")

        if images_dir is not None and images_dir.is_dir():
            split_img_dir = coco_dir / split_name
            split_img_dir.mkdir(parents=True, exist_ok=True)
            index = index_images_by_basename(images_dir)
            copied_or_linked = 0
            for fn in fns:
                base = os.path.basename(fn)
                src = index.get(base)
                if src is None:
                    continue
                src_resolved = Path(os.path.realpath(src))
                if not src_resolved.exists():
                    continue
                dst = split_img_dir / base
                if copy_images:
                    if dst.exists():
                        dst.unlink()
                    shutil.copy2(src_resolved, dst)
                    copied_or_linked += 1
                else:
                    if not dst.exists():
                        os.symlink(src_resolved, dst)
                        copied_or_linked += 1
            mode = "copied" if copy_images else "symlinks"
            print(f"  {split_img_dir}: {copied_or_linked} new {mode} (total images {len(fns)}).")


def index_images_by_basename(root: Path) -> dict:
    """Recursively index image basename -> Path."""
    out = {}
    for r, _dirs, files in os.walk(root):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            out[f] = Path(r) / f
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter image_annotations.json to top-10 categories and visibility 3/4 only."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="datasets/nuscenes/nuSenes2d/image_annotations.json",
        help="Input image_annotations.json path.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: overwrite --json).",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="If set, require --out; do not overwrite input.",
    )
    parser.add_argument(
        "--coco-dir",
        type=str,
        default=None,
        help="Output COCO dataset root (e.g. datasets/nuscenes/export_2d_5000_coco_top10_vis34). "
             "Writes annotations/instances_train2017.json and instances_val2017.json.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio for train/val split (default 0.8).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default 42).",
    )
    parser.add_argument(
        "--width", type=int, default=1600, help="Image width for COCO (nuScenes default)."
    )
    parser.add_argument(
        "--height", type=int, default=900, help="Image height for COCO (nuScenes default)."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Source images dir (e.g. nuSenes2d/images). If set with --coco-dir, creates train2017/ and val2017/ by basename.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy image files into train2017/val2017 instead of symlinking (for portable archive).",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    out_path = Path(args.out) if args.out else json_path
    if args.no_overwrite and out_path.resolve() == json_path.resolve():
        raise SystemExit("Use --out when --no-overwrite to avoid overwriting input.")

    with open(json_path, "r") as f:
        ann_list = json.load(f)

    allowed_vis = {"3", "4"}
    filtered = []
    for ann in ann_list:
        vis = ann.get("visibility_token")
        if str(vis) not in allowed_vis:
            continue
        short = map_category(ann.get("category_name", ""))
        if short is None:
            continue
        # 保留原记录结构，只改 category_name 为短名
        rec = dict(ann)
        rec["category_name"] = short
        filtered.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(filtered, f, indent=4, sort_keys=True)

    print(f"Kept {len(filtered)} annotations (vis 3/4, top-10 only). Wrote: {out_path}")

    if getattr(args, "coco_dir", None):
        coco_dir = Path(args.coco_dir)
        images_dir = Path(args.images_dir) if getattr(args, "images_dir", None) else None
        print(f"Writing COCO split to {coco_dir} ...")
        _write_coco_and_link(
            filtered,
            coco_dir,
            train_ratio=args.train_ratio,
            seed=args.split_seed,
            width=args.width,
            height=args.height,
            images_dir=images_dir,
            copy_images=getattr(args, "copy_images", False),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
