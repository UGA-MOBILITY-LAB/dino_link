#!/usr/bin/env bash
# 为官方 DETR 的 --coco_path 创建适配目录，使 nuScenes COCO 格式 val 可被正确找到。
# DETR 要求: coco_path/annotations/instances_val2017.json, coco_path/val2017/<file_name>
# 你的 val.json 里 file_name 为 samples/CAM_FRONT/xxx.jpg，故 val2017 指到 nuScenes 根目录即可。

set -e
COCO_FOR_DETR="${1:-/home/tianle/dinolink_project/datasets/nuscenes/coco_for_detr}"
NUSCENES_ROOT="${2:-/mnt/datasets/nuScenes}"
COCO_ANNOTATIONS="${3:-/home/tianle/dinolink_project/datasets/nuscenes/coco_cam_front}"

mkdir -p "$COCO_FOR_DETR/annotations"
ln -sf "$COCO_ANNOTATIONS/train.json" "$COCO_FOR_DETR/annotations/instances_train2017.json"
ln -sf "$COCO_ANNOTATIONS/val.json" "$COCO_FOR_DETR/annotations/instances_val2017.json"
ln -sf "$NUSCENES_ROOT" "$COCO_FOR_DETR/train2017"
ln -sf "$NUSCENES_ROOT" "$COCO_FOR_DETR/val2017"
echo "Created: $COCO_FOR_DETR"
echo "  annotations/instances_train2017.json -> $COCO_ANNOTATIONS/train.json"
echo "  annotations/instances_val2017.json -> $COCO_ANNOTATIONS/val.json"
echo "  train2017, val2017 -> $NUSCENES_ROOT"
echo ""
echo "Eval with:"
echo "  cd $(dirname "$0")"
echo "  python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path $COCO_FOR_DETR"
