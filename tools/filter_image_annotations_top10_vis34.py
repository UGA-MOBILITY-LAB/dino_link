#!/usr/bin/env python3
"""
From image_annotations.json, keep only:
  - visibility_token equal to "3" or "4"
  - category belonging to the 10 classes (car, pedestrian, barrier, truck, trafficcone, construction, motorcycle, bus, bicycle, trailer),
    and normalize category_name to the short names above.

Output:
  1) The filtered list JSON (same format as the input).
  2) If --coco-dir is specified: generate a COCO-format train/val split under that directory (annotations/instances_*.json),
     and optionally train2017/, val2017/ image directories (symlinked or copied from --images-dir by basename, see --copy-images).
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

# Fixed order of the 10 classes (consistent with export_2d_*_top10_vis34), id 1~10
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

# nuScenes category_name prefix -> short names of the 10 classes (consistent with export_2d_train_coco_second_level_top10_vis34)
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
    """Return the short name if it belongs to one of the 10 classes, otherwise return None (should be discarded).
    Supports: 1) already a short name (car, bus, ...); 2) nuScenes long name (vehicle.car, human.pedestrian, ...).
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
    """Build COCO images/annotations from the filtered list and the filename order of this split."""
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
    """Split into train/val, write annotations/instances_*.json, optionally create train2017/val2017 (symlink or copy)."""
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
        # Keep the original record structure, only change category_name to the short name
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
