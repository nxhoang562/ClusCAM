#!/usr/bin/env bash
set -euo pipefail

# Danh sách các ResNet hoặc model bạn muốn test
MODELS=(
  "resnet50"
  "resnet101"
  "inception_v3"
)

# Danh sách các CAM baseline bạn muốn so sánh
CAM_METHODS=(
  "scorecam"
)

# Cấu hình c
DATASET="datasets/ILSVRC2012_img_val"
BASE_EXCEL_DIR="results/imagenet_val"
START_IDX=1001
END_IDX=4001
MODE_TYPE="test"

for MODEL in "${MODELS[@]}"; do
  for CAM_METHOD in "${CAM_METHODS[@]}"; do
    echo "==== $MODE_TYPE with $MODEL + $CAM_METHOD ===="

    # Đường dẫn lưu Excel riêng cho từng model + cam
    EXCEL_PATH="${BASE_EXCEL_DIR}/${MODE_TYPE}_${MODEL}_cam-baselines_${START_IDX}_${END_IDX}-imgs.xlsx"

    # Tạo thư mục nếu chưa có
    mkdir -p "$(dirname "$EXCEL_PATH")"

    python3 test_batch2.py \
      --mode batch \
      --model "${MODEL}" \
      --dataset "${DATASET}" \
      --excel-path "${EXCEL_PATH}" \
      --cam-method "${CAM_METHOD}" \
      --mode-type "${MODE_TYPE}" \
      --start-idx "${START_IDX}" \
      --end-idx "${END_IDX}"

    echo "---- Finished $MODEL + $CAM_METHOD, results in $EXCEL_PATH ----"
    echo
  done
done
