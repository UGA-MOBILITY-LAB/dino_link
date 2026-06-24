#!/usr/bin/env bash
# Build DETR's --coco_path directory using export_2d (export_2d_coco.json + images/).
# In COCO the file_name is images/xxx.jpg, so val2017 points to export_2d, making val2017/images/xxx.jpg exist.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPORT_2D="${1:-$PROJECT_ROOT/datasets/nuscenes/export_2d}"
COCO_DETR="${2:-$PROJECT_ROOT/datasets/nuscenes/export_2d_detr}"

EXPORT_2D="$(realpath "$EXPORT_2D")"
COCO_DETR="$(realpath "$COCO_DETR")"

mkdir -p "$COCO_DETR/annotations"
ln -sf "$EXPORT_2D/export_2d_coco.json" "$COCO_DETR/annotations/instances_val2017.json"
ln -sf "$EXPORT_2D/export_2d_coco.json" "$COCO_DETR/annotations/instances_train2017.json"
ln -sfn "$EXPORT_2D" "$COCO_DETR/val2017"
ln -sfn "$EXPORT_2D" "$COCO_DETR/train2017"

echo "Created: $COCO_DETR"
echo "  annotations/instances_*2017.json -> $EXPORT_2D/export_2d_coco.json"
echo "  val2017, train2017 -> $EXPORT_2D"
echo ""
echo "Eval: cd $SCRIPT_DIR && python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path $COCO_DETR"
