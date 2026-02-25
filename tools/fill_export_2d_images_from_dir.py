#!/usr/bin/env python3
"""
基于 export_2d/image_annotations.json，把对应图片按 basename
从你指定的图片目录里拷贝/软链接到 export_2d/images 下。

使用场景：
- image_annotations.json 里有 filename（含路径），但实际图片集中放在某个目录，
  比如你把 `/mnt/datasets/nuScenes/samples/CAM_FRONT/*.jpg` 软链接到了别的地方。
- 这个脚本只按文件名（basename）匹配，不依赖原始路径。
- 加 --random-sample 时，从 JSON 要求的图片里随机取一部分填充（数量或比例）。

示例：
  # 填充全部
  python tools/fill_export_2d_images_from_dir.py \\
    --json datasets/nuscenes/export_2d/image_annotations.json \\
    --images-src /real/path/to/all_cam_images \\
    --out-dir datasets/nuscenes/export_2d/images \\
    --symlink

  # 随机取 500 张
  python tools/fill_export_2d_images_from_dir.py ... --random-sample 500 --symlink

  # 随机取 20% 比例
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
            # 若有重名，后出现的覆盖前面的，一般 basename 在 nuScenes 里是唯一的
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
        help="export_2d image_annotations.json 路径。",
    )
    parser.add_argument(
        "--images-src",
        type=str,
        required=True,
        help="实际存放图片的目录（会递归搜索 jpg/png 等）。按文件名（basename）匹配。",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/tianle/dinolink_project/datasets/nuscenes/export_2d/images",
        help="要把图片放到的目录（不存在会创建）。",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="使用软链接而不是复制（节省磁盘，推荐）。",
    )
    parser.add_argument(
        "--random-sample",
        type=float,
        default=None,
        metavar="N_or_FRAC",
        help="随机取：填整数为数量（如 500），填 (0,1] 小数为比例（如 0.2 表示 20%%）。不传则填充全部。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，便于复现。",
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

    # image_annotations.json 是列表，每条有 filename
    basenames: List[str] = list({os.path.basename(rec["filename"]) for rec in anns})

    # 随机取：按数量或比例
    if args.random_sample is not None:
        if args.seed is not None:
            random.seed(args.seed)
        n_total = len(basenames)
        if args.random_sample > 1:
            k = min(int(args.random_sample), n_total)
        elif 0 < args.random_sample <= 1:
            k = max(1, int(n_total * args.random_sample))
        else:
            raise ValueError("--random-sample 应为正整数（数量）或 (0,1] 小数（比例）。")
        basenames = random.sample(basenames, k)
        print(f"Random sample: using {k} of {n_total} images (seed={args.seed}).")
    else:
        print(f"Need {len(basenames)} unique images according to JSON.")

    print(f"Indexing source images under {src_root} ...")
    mapping = index_images_by_basename(src_root)
    print(f"Found {len(mapping)} images in source dir (by basename).")

    copied = 0
    skipped = 0
    # 随机顺序处理（若未指定 --random-sample 则先 shuffle 再处理，避免固定顺序）
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
            # 已存在就跳过（不覆盖）
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
