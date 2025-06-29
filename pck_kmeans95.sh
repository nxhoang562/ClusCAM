#!/usr/bin/env bash
set -euo pipefail

# Cấu hình tham số
MODEL="resnet18"
LAYER="layer4"
DATASET="datasets/imagenet"  # Thư mục chứa ảnh, có thể là đường dẫn đến tập dữ liệu ImageNet hoặc một tập dữ liệu khác
EXCEL_PATH="results/imagenet"
K_VALUES=(30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)    # danh sách các K bạn muốn thử
CAM_METHOD="cluster"  # hoặc grad, score, v.v.
TOP_N=1000

# Tạo thư mục kết quả (theo MODEL) nếu chưa có
mkdir -p "$(dirname "${EXCEL_PATH}")"


python3 test2.py --mode batch --model "${MODEL}" --layer-name "${LAYER}" --dataset "${DATASET}" --excel-path "${EXCEL_PATH}" --k-values "${K_VALUES[@]}" --cam-method "${CAM_METHOD}" --top-n "${TOP_N}" --zero-ratio 0.5 --temperature 0.5