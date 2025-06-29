#!/usr/bin/env bash
set -euo pipefail

# Danh sách các model muốn test
MODELS=(
  "resnet34"
  "resnet50"
  "resnet101"
  "resnet152"
)

# Danh sách các CAM methods baseline (không bao gồm cluster)
BASELINE_CAM_METHODS=(
  "gradcam"
  "gradcamplusplus"
  "layercam"
  "scorecam"
  "ablationcam"
  "shapleycam"
)

# Cấu hình chung
LAYER="layer4"
DATASET="datasets/imagenet"
BASE_EXCEL_DIR="results/imagenet"
TOP_N=1000

# Tạo thư mục chung nếu chưa có
mkdir -p "$BASE_EXCEL_DIR"

for MODEL in "${MODELS[@]}"; do
  for CAM in "${BASELINE_CAM_METHODS[@]}"; do
    echo "==== Testing $MODEL with CAM_METHOD=$CAM ===="

    # Lưu file kết quả riêng biệt cho mỗi cặp model+CAM
    EXCEL_PATH="${BASE_EXCEL_DIR}/${MODEL}_${CAM}.xlsx"

    python3 test.py \
      --mode batch \
      --model "$MODEL" \
      --layer-name "$LAYER" \
      --dataset "$DATASET" \
      --excel-path "$EXCEL_PATH" \
      --cam-method "$CAM" \
      --top-n "$TOP_N"

    echo "Results saved to $EXCEL_PATH"
    echo
  done
done
