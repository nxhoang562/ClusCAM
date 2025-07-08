#!/usr/bin/env bash
set -euo pipefail

# Danh sách các ResNet muốn test
MODELS=(
  "resnet18"
)
# Cấu hình chung
DATASET="datasets/ILSVRC2012_img_val"
BASE_EXCEL_DIR="results/imagenet_val"
K_VALUES=(2 5 10 15 20 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200)
CAM_METHOD="cluster"
TOP_N=3000
ZERO_RATIO=0.5
TEMPERATURE=0.5
MODE_TYPE="validation"
BATCH_SIZE=64

for MODEL in "${MODELS[@]}"; do
  echo "==== $MODE_TYPE with $MODEL ===="

  # Đường dẫn lưu Excel riêng cho từng model
  EXCEL_PATH="${BASE_EXCEL_DIR}/${MODEL}_${MODE_TYPE}_${CAM_METHOD}_batch${BATCH_SIZE}_zr${ZERO_RATIO}_temp${TEMPERATURE}_${TOP_N}imgs.xlsx"
  
  # Tạo thư mục nếu chưa có
  mkdir -p "$(dirname "$EXCEL_PATH")"
  
  python3 test_batchplus.py \
    --mode batch \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --excel-path "${EXCEL_PATH}" \
    --k-values "${K_VALUES[@]}" \
    --cam-method "${CAM_METHOD}" \
    --top-n "${TOP_N}" \
    --zero-ratio "${ZERO_RATIO}" \
    --temperature "${TEMPERATURE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 4 \
    --mode-type "${MODE_TYPE}" \

  echo "---- Finished $MODEL, results in $EXCEL_PATH ----"
  echo
done
