#!/usr/bin/env bash
set -euo pipefail

# Danh sách các ResNet muốn test
MODELS=(
  # 'resnet18'
  # 'efficientNet'
  # 'resnet34'
  'resnet50'
  # 'inception_v3'
  'resnet101'
)

# Cấu hình chung
CAM_METHOD="cluster"
DATASET="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened"
BBOX="/home/infres/ltvo/ClusCAM/datasets/LOC_val_solution.csv"
BASE_EXCEL_DIR="results/chaylai_localization_results"
# K_VALUES=(2 5 10 15 20 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200)
K_VALUES=(120 150)
START_IDX=1001
END_IDX=1501
ZERO_RATIO=0.5
TEMPERATURE=0.5
MODE_TYPE="test"  


for MODEL in "${MODELS[@]}"; do
  echo "==== $MODE_TYPE with $MODEL ===="

  # Đường dẫn lưu Excel riêng cho từng model
  EXCEL_PATH="${BASE_EXCEL_DIR}/${MODEL}_${MODE_TYPE}_${CAM_METHOD}_zr${ZERO_RATIO}_temp${TEMPERATURE}_${START_IDX}-${END_IDX}imgs.xlsx"
  
  # Tạo thư mục nếu chưa có
  mkdir -p "$(dirname "$EXCEL_PATH")"
  
  python3 /home/infres/ltvo/ClusCAM/test_localization.py \
    --mode batch \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --excel-path "${EXCEL_PATH}" \
    --k-values "${K_VALUES[@]}" \
    --cam-method "${CAM_METHOD}" \
    --zero-ratio "${ZERO_RATIO}" \
    --temperature "${TEMPERATURE}" \
    --start-idx "${START_IDX}" \
    --end-idx "${END_IDX}" \
    --bbox-csv "${BBOX}" 

  echo "---- Finished $MODEL, results in $EXCEL_PATH ----"
  echo
done
