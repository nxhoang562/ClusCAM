#!/usr/bin/env bash
set -euo pipefail

# Cấu hình tham số
MODEL="alzheimer_resnet18"
LAYER="layer4"
DATASET="datasets/alzheimer/test_imgs"       
EXCEL_PATH="results"  
K_VALUES=(30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
# CAM_METHOD="shapleycam"
TOP_N=""   # để trống nghĩa là load toàn bộ ảnh

# Tạo thư mục kết quả nếu chưa có
mkdir -p "$(dirname "${EXCEL_PATH}")"

# Chuẩn bị mảng đối số cho --top-n (nếu có)
if [ -n "${TOP_N}" ]; then
  TOPN_ARGS=(--top-n "${TOP_N}")
else
  TOPN_ARGS=()
fi

# Chạy batch test với model alzheimer_resnet18


python3 test.py --mode batch --model "${MODEL}" --layer-name "${LAYER}" --dataset "${DATASET}" --excel-path "${EXCEL_PATH}" --k-values "${K_VALUES[@]}" --cam-method cluster --zero-ratio 0.5 --temperature 0.5