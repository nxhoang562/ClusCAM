#!/usr/bin/env bash
set -euo pipefail

# Danh sách các model muốn test
MODELS=(
  resnet34
  resnet50
  inception_v3
  efficientNet
  resnet101
  resnet152
  vgg16
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

  # "cluster", "gradcam", "gradcamplusplus",
  #           "layercam", "scorecam", "ablationcam", "shapleycam"

# Cấu hình chung
DATASET="datasets/ILSVRC2012_img_val"
BASE_EXCEL_DIR="results/imagenet_val"
START_IDX=1001
END_IDX=4001
# Tạo thư mục chung nếu chưa có
mkdir -p "$BASE_EXCEL_DIR"

for MODEL in "${MODELS[@]}"; do
  for CAM in "${BASELINE_CAM_METHODS[@]}"; do
    echo "==== Testing $MODEL with CAM_METHOD=$CAM ===="

    # Lưu file kết quả riêng biệt cho mỗi cặp model+CAM
    EXCEL_PATH="${BASE_EXCEL_DIR}/_${MODEL}_cam-baselines_${START_IDX}_${END_IDX}-imgs.xlsx"

    mkdir -p "$(dirname "$EXCEL_PATH")"

    python3 test_baselines.py \
      --mode batch \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --excel-path "$EXCEL_PATH" \
      --cam-method "$CAM" \
      --start-idx "$START_IDX" \
      --end-idx "$END_IDX" 

    echo "Results saved to $EXCEL_PATH"
    echo
  done
done
