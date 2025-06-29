#!/usr/bin/env bash
set -euo pipefail

# Danh sách các ResNet muốn test
MODELS=(
  "resnet34"
  "resnet50"
  "resnet101"
  "resnet152"
)

# Cấu hình chung
LAYER="layer4"
DATASET="datasets/imagenet"
BASE_EXCEL_DIR="results/imagenet"
K_VALUES=(30 40 50 60 70 75 80 85 90 95 100)
CAM_METHOD="cluster"
TOP_N=1000
ZERO_RATIO=0.5
TEMPERATURE=0.5

for MODEL in "${MODELS[@]}"; do
  echo "==== Testing $MODEL ===="

  # Đường dẫn lưu Excel riêng cho từng model
  EXCEL_PATH="${BASE_EXCEL_DIR}/${MODEL}_${CAM_METHOD}_zr${ZERO_RATIO}_temp${TEMPERATURE}.xlsx"
  
  # Tạo thư mục nếu chưa có
  mkdir -p "$(dirname "$EXCEL_PATH")"
  
  python3 test.py \
    --mode batch \
    --model "${MODEL}" \
    --layer-name "${LAYER}" \
    --dataset "${DATASET}" \
    --excel-path "${EXCEL_PATH}" \
    --k-values "${K_VALUES[@]}" \
    --cam-method "${CAM_METHOD}" \
    --top-n "${TOP_N}" \
    --zero-ratio "${ZERO_RATIO}" \
    --temperature "${TEMPERATURE}"

  echo "---- Finished $MODEL, results in $EXCEL_PATH ----"
  echo
done
