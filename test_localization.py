import os
import json
import torch
import torchvision.models as models
import pandas as pd

from torchvision.models import (
    inception_v3, Inception_V3_Weights,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)
from args import get_args
from utils_localization import batch_test


def reshape_transform(tensor, height=14, width=14):
    # ViT-specific reshape (năm sau nếu cần)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def main():
    args = get_args()

    # --- Load bbox CSV và chỉ giữ ảnh có đúng 1 box ---
    if args.mode == 'batch':
        if not args.bbox_csv:
            raise RuntimeError("--bbox-csv là bắt buộc khi mode='batch'")
        df = pd.read_csv(args.bbox_csv)

        # split PredictionString thành tokens
        df['tokens'] = df['PredictionString'].str.split()
        # chỉ lấy những dòng có đúng 5 token: [class, x1, y1, x2, y2]
        df1 = df[df['tokens'].apply(len) == 5].copy()
        if df1.empty:
            raise RuntimeError("Không tìm thấy ảnh nào có đúng 1 box trong CSV.")

        # parse bbox thành list[int]
        df1['bbox'] = df1['tokens'].apply(lambda t: list(map(int, t[1:])))
        # cho dễ lookup:
        df1.set_index('ImageId', inplace=True)
    else:
        df1 = None
    # ---------------------------------------------------

    # --- Load model và target_layer ---
    resnet_confs = {
        'resnet18': (models.resnet18, ResNet18_Weights),
        'resnet34': (models.resnet34, ResNet34_Weights),
        'resnet50': (models.resnet50, ResNet50_Weights),
        'resnet101': (models.resnet101, ResNet101_Weights),
        'resnet152': (models.resnet152, ResNet152_Weights),
    }

    if args.model in resnet_confs:
        ctor, W = resnet_confs[args.model]
        model = ctor(weights=W.IMAGENET1K_V1)
        model.eval()
        input_size   = (224, 224)
        target_layer = model.layer4

    elif args.model == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.eval()
        target_layer = model.features[28]
        input_size   = (224, 224)

    elif args.model == 'inception_v3':
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.eval()
        target_layer = model.Mixed_7c
        input_size   = (299, 299)

    elif args.model == 'efficientNet':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.eval()
        target_layer = model.features[-1]
        input_size   = (224, 224)

    else:
        raise ValueError(f"Model {args.model} không hỗ trợ")

    model_dict = {
        'type':         args.model,
        'arch':         model,
        'target_layer': target_layer,
        'input_size':   input_size,
        'cam_method':   args.cam_method,
        'zero_ratio':   args.zero_ratio,
        'temperature':  args.temperature,
    }
    # ------------------------------------

    # --- Single-image mode ---
    if args.mode == 'single':
        if not args.img_path:
            raise RuntimeError("--img-path là bắt buộc khi mode='single'")
        drop, inc = test_single_image(
            model,
            model_dict,
            args.img_path,
            args.save_prefix,
            cam_method=args.cam_method,
            num_clusters=args.num_clusters
        )
        print(f"Average Drop: {drop:.4f}, Increase Confidence: {inc:.4f}")
        return

    # --- Batch mode ---
    # chuẩn bị danh sách bboxes song song với list_image_paths
    image_paths = None
    if args.mode == 'batch':
        # list tất cả ảnh trong folder
        all_paths = sorted(os.listdir(args.dataset))
        # chỉ chọn những file .jpg/.png và whose stem in df1.index
        image_paths = [
            os.path.join(args.dataset, fn)
            for fn in all_paths
            if os.path.splitext(fn)[0] in df1.index
        ]
        if not image_paths:
            raise RuntimeError("Không tìm thấy ảnh nào khớp với ImageId trong CSV.")
        # build bboxes list đúng thứ tự
        bboxes = [
            df1.loc[os.path.splitext(os.path.basename(p))[0], 'bbox']
            for p in image_paths
        ]
    # ----------------------------------------------------------------

    # --- Xuất Excel ---
    excel = args.excel_path
    if not excel.lower().endswith(('.xls', '.xlsx')):
        os.makedirs(excel, exist_ok=True)
        if args.cam_method == "cluster":
            excel = os.path.join(
                excel,
                f"{args.model}_{args.cam_method}_zero-out-{args.zero_ratio}_temperature-{args.temperature}.xlsx"
            )
        else:
            excel = os.path.join(excel, f"{args.model}__{args.cam_method}.xlsx")

    batch_test(
        model,
        model_dict,
        args.dataset,
        excel,
        args.k_values,
        bbox_csv=args.bbox_csv,
        cam_method=args.cam_method,
        top_n=args.top_n,
        model_name=args.model,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )


if __name__ == '__main__':
    main()
