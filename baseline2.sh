#!/usr/bin/env bash
set -euo pipefail

# Danh sách các ResNet muốn test (không có dấu phẩy)
MODELS=(
  "resnet18"
  "resnet34"
  "resnet50"
  "resnet101"
  "inception_v3"
)

# Danh sách CAM methods muốn test
# Ví dụ: polyp, polym, polypm, gradcam, scorecam, cluster…
BASELINE_CAM_METHODS=(
  "polypm"
)

# Cấu hình chung
DATASET="datasets/imagenet"
BASE_EXCEL_DIR="results/imagenet"
TOP_N=1000

# Tạo thư mục chung nếu chưa có
mkdir -p "$BASE_EXCEL_DIR"

for MODEL in "${MODELS[@]}"; do
  for CAM in "${BASELINE_CAM_METHODS[@]}"; do
    echo "==== Testing $MODEL with CAM_METHOD=$CAM ===="

    EXCEL_PATH="${BASE_EXCEL_DIR}/${MODEL}_${CAM}.xlsx"

    python3 test_batch.py \
      --mode batch \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --excel-path "$EXCEL_PATH" \
      --cam-method "$CAM" \
      --top-n "$TOP_N" \
      --batch-size 256 \
      --num-workers 4

    echo "Results saved to $EXCEL_PATH"
    echo
  done
done
