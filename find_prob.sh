#!/usr/bin/env bash
set -euo pipefail


MODELS=(
   'resnet18'
  'resnet34'
  'resnet50'
  'resnet101'
  'efficientNet'
  'vit_b_16'
  'swin_b'
  'inception_v3'
)

# Cấu hình chung
DATASET="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened"
BASE_EXCEL_DIR="results/prediction_probs"
EXCEL_FILENAME="all_models_probs.xlsx"
EXCEL_PATH="${BASE_EXCEL_DIR}/${EXCEL_FILENAME}"
START_IDX=1001
END_IDX=3001

# Tạo thư mục kết quả nếu chưa có
mkdir -p "${BASE_EXCEL_DIR}"

# Nếu file đã tồn tại, xóa để viết lại từ đầu
if [[ -f "${EXCEL_PATH}" ]]; then
  rm "${EXCEL_PATH}"
fi

echo "=> Writing all model predictions into ${EXCEL_PATH}"
for MODEL in "${MODELS[@]}"; do
  echo "---- Processing model: ${MODEL} ----"
  python3 test_probs.py \
    --mode batch \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --excel-path "${EXCEL_PATH}" \
    --start-idx "${START_IDX}" \
    --end-idx "${END_IDX}"
  echo "   → Sheet '${MODEL}' written."
done

echo "✅ All done. Results in ${EXCEL_PATH}"
