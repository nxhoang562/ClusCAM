#!/usr/bin/env bash
set -euo pipefail

# Cấu hình tham số
MODEL="resnet18"
LAYER="layer4"
DATASET="datasets/imagenet"  # Thư mục chứa ảnh, có thể là đường dẫn đến tập dữ liệu ImageNet hoặc một tập dữ liệu khác
EXCEL_PATH="results/imagenet"
K_VALUES=(30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)    # danh sách các K bạn muốn thử
# CAM_METHOD="gradcam"  # hoặc grad, score, v.v.
TOP_N=1000

# Tạo thư mục kết quả (theo MODEL) nếu chưa có
mkdir -p "$(dirname "${EXCEL_PATH}")"


# python3 test.py --mode batch --model "${MODEL}" --layer-name "${LAYER}" --dataset "${DATASET}" --excel-path "${EXCEL_PATH}" --cam-method "${CAM_METHOD}" --top-n "${TOP_N}" 
# python3 test.py --mode batch --model "${MODEL}" --layer-name "${LAYER}" --dataset "${DATASET}" --excel-path "${EXCEL_PATH}" --cam-method "gradcamplusplus" --top-n "${TOP_N}" 
# python3 test.py --mode batch --model "${MODEL}" --layer-name "${LAYER}" --dataset "${DATASET}" --excel-path "${EXCEL_PATH}" --cam-method "scorecam" --top-n "${TOP_N}"
# python3 test.py --mode batch --model "${MODEL}" --layer-name "${LAYER}" --dataset "${DATASET}" --excel-path "${EXCEL_PATH}" --cam-method "ablationcam" --top-n "${TOP_N}"
# python3 test.py --mode batch --model "${MODEL}" --layer-name "${LAYER}" --dataset "${DATASET}" --excel-path "${EXCEL_PATH}" --cam-method "layercam" --top-n "${TOP_N}"
python3 test.py --mode batch --model "${MODEL}" --layer-name "${LAYER}" --dataset "${DATASET}" --excel-path "${EXCEL_PATH}" --cam-method "shapleycam" --top-n "${TOP_N}"
