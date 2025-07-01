#!/usr/bin/env bash
set -euo pipefail

MODELS=("inception_v3")

# Cấu hình chung
DATASET="datasets/imagenet"
BASE_EXCEL_DIR="results/imagenet"
K_VALUES=(60 70 80 90 100)
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
