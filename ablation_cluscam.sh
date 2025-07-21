#!/usr/bin/env bash
set -euo pipefail

# Danh sách các ResNet muốn test
MODELS=(
  "resnet18"
)

# Cấu hình chung
CAM_METHOD="randomcam"
DATASET="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened"
BASE_EXCEL_DIR="results/abalation_resenet18"
K_VALUES=(10 30 50 100)
START_IDX=1001
END_IDX=2001
MODE_TYPE="test"


ZERO_RATIOS=(0 0.5)
TEMPERATURES=(1 0.5)

for MODEL in "${MODELS[@]}"; do
  for ZERO_RATIO in "${ZERO_RATIOS[@]}"; do
    for TEMPERATURE in "${TEMPERATURES[@]}"; do

      echo "==== $MODE_TYPE with $MODEL (zero_ratio=${ZERO_RATIO}, temperature=${TEMPERATURE}) ===="

      # Đường dẫn lưu Excel riêng cho từng cấu hình
      EXCEL_PATH="${BASE_EXCEL_DIR}/${CAM_METHOD}_${K_VALUES}_${MODEL}_zr${ZERO_RATIO}_temp${TEMPERATURE}_1001-2001imgs.xlsx"
      mkdir -p "$(dirname "$EXCEL_PATH")"

      python3 test_ablation.py \
        --mode batch \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --excel-path "${EXCEL_PATH}" \
        --k-values "${K_VALUES[@]}" \
        --cam-method "${CAM_METHOD}" \
        --zero-ratio "${ZERO_RATIO}" \
        --temperature "${TEMPERATURE}" \
        --start-idx "${START_IDX}" \
        --end-idx "${END_IDX}"

      echo "---- Finished $MODEL zr=${ZERO_RATIO} temp=${TEMPERATURE}, results in $EXCEL_PATH ----"
      echo

    done
  done
done
