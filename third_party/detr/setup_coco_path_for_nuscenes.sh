#!/usr/bin/env bash
# Create an adapter directory for the official DETR --coco_path so the nuScenes COCO-format val can be found correctly.
# DETR requires: coco_path/annotations/instances_val2017.json, coco_path/val2017/<file_name>
# In your val.json the file_name is samples/CAM_FRONT/xxx.jpg, so val2017 just needs to point to the nuScenes root directory.

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
