#!/usr/bin/env bash
set -euo pipefail

# Danh sách các ResNet muốn test
MODELS=(
  "resnet18"
)

# Cấu hình chung
CAM_METHOD="diffcam"
DATASET="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened"
BASE_EXCEL_DIR="results/abalation2-for-temperature_softmax"
K_VALUES=(10 50)
START_IDX=1001
END_IDX=2001

for MODEL in "${MODELS[@]}"; do
  echo "==== $CAM_METHOD with $MODEL ===="

  # Đường dẫn lưu Excel riêng cho từng cấu hình
  EXCEL_PATH="${BASE_EXCEL_DIR}/logi_diff_no-zerout-softmax$_${MODEL}_1001-2001imgs.xlsx"
  mkdir -p "$(dirname "$EXCEL_PATH")"

  python3 test_ablation.py \
    --mode batch \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --excel-path "${EXCEL_PATH}" \
    --k-values "${K_VALUES[@]}" \
    --cam-method "${CAM_METHOD}" \
    --start-idx "${START_IDX}" \
    --end-idx "${END_IDX}"

  echo "---- Finished $MODEL, results in $EXCEL_PATH ----"
  echo
done
