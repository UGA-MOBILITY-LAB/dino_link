#!/usr/bin/env python3
"""
Based on export_2d/image_annotations.json, copy/symlink the corresponding images
by basename from the image directory you specify into export_2d/images.

Use case:
- image_annotations.json contains filename (with path), but the actual images are
  stored together in some directory, e.g. you symlinked
  `/mnt/datasets/nuScenes/samples/CAM_FRONT/*.jpg` to somewhere else.
- This script only matches by file name (basename), not relying on the original path.
- With --random-sample, randomly take a portion of the images required by the JSON to fill (count or ratio).

Examples:
  # Fill all
  python tools/fill_export_2d_images_from_dir.py \\
    --json datasets/nuscenes/export_2d/image_annotations.json \\
    --images-src /real/path/to/all_cam_images \\
    --out-dir datasets/nuscenes/export_2d/images \\
    --symlink

  # Randomly take 500 images
  python tools/fill_export_2d_images_from_dir.py ... --random-sample 500 --symlink

  # Randomly take 20% ratio
  python tools/fill_export_2d_images_from_dir.py ... --random-sample 0.2 --symlink
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List


def index_images_by_basename(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    root = root.resolve()
    for r, _dirs, files in os.walk(root):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            p = Path(r) / f
            # If there are duplicate names, later ones overwrite earlier ones; generally basename is unique in nuScenes
            mapping[f] = p
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill export_2d/images by matching basenames from a source image directory."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="/home/tianle/dinolink_project/datasets/nuscenes/export_2d/image_annotations.json",
        help="Path to export_2d image_annotations.json.",
    )
    parser.add_argument(
        "--images-src",
        type=str,
        required=True,
        help="Directory where images are actually stored (searched recursively for jpg/png etc.). Matched by file name (basename).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/tianle/dinolink_project/datasets/nuscenes/export_2d/images",
        help="Directory to place the images into (created if it does not exist).",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying (saves disk space, recommended).",
    )
    parser.add_argument(
        "--random-sample",
        type=float,
        default=None,
        metavar="N_or_FRAC",
        help="Random sample: an integer is a count (e.g. 500), a decimal in (0,1] is a ratio (e.g. 0.2 means 20%%). If not given, fill all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed, for reproducibility.",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    src_root = Path(args.images_src)
    if not src_root.is_dir():
        raise FileNotFoundError(src_root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        anns = json.load(f)

    # image_annotations.json is a list, each entry has a filename
    basenames: List[str] = list({os.path.basename(rec["filename"]) for rec in anns})

    # Random sample: by count or ratio
    if args.random_sample is not None:
        if args.seed is not None:
            random.seed(args.seed)
        n_total = len(basenames)
        if args.random_sample > 1:
            k = min(int(args.random_sample), n_total)
        elif 0 < args.random_sample <= 1:
            k = max(1, int(n_total * args.random_sample))
        else:
            raise ValueError("--random-sample should be a positive integer (count) or a decimal in (0,1] (ratio).")
        basenames = random.sample(basenames, k)
        print(f"Random sample: using {k} of {n_total} images (seed={args.seed}).")
    else:
        print(f"Need {len(basenames)} unique images according to JSON.")

    print(f"Indexing source images under {src_root} ...")
    mapping = index_images_by_basename(src_root)
    print(f"Found {len(mapping)} images in source dir (by basename).")

    copied = 0
    skipped = 0
    # Process in random order (if --random-sample is not specified, shuffle first before processing to avoid a fixed order)
    order = list(basenames)
    if args.random_sample is None and args.seed is not None:
        random.seed(args.seed)
    random.shuffle(order)

    for base in order:
        src = mapping.get(base)
        if src is None:
            print(f"Skip missing in src: {base}")
            skipped += 1
            continue

        dst = out_dir / base
        if dst.exists():
            # Skip if it already exists (do not overwrite)
            continue

        if args.symlink:
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)
        copied += 1

    print(f"Done. Copied/linked {copied} images into {out_dir}, skipped {skipped} not found in source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
