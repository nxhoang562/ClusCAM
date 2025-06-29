import os
import torchvision.models as models
from args import get_args
from test_utils_pca import test_single_image, batch_test
import torch
from models.alzheimer_resnet18.alzheimer_resnet18 import load_model

def main():
    args = get_args()

    # Load model
    if args.model == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()

        model_dict = {
            'type': args.model,
            'arch': model,
            'layer_name': args.layer_name,
            'input_size': (224, 224),
            'cam_method': args.cam_method,
            'zero_ratio': args.zero_ratio,
            'temperature': args.temperature,
        }
    elif args.model == 'alzheimer_resnet18':
        model = load_model(checkpoint_path = "/home/infres/xnguyen-24/cluster_cam/models/alzheimer_resnet18/alzheimer_resnet18.pth", device='cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        model_dict = {
            'type': args.model,
            'arch': model,
            'layer_name': args.layer_name,
            'input_size': (128, 128),  # Kích thước đầu vào của mô hình MRI
            'cam_method': args.cam_method,
            'zero_ratio': args.zero_ratio,
            'temperature': args.temperature,
        }
        
    else:
        raise ValueError(f"Model {args.model} không hỗ trợ")

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

    else:
        if not args.dataset or not args.excel_path:
            raise RuntimeError("--dataset và --excel-path là bắt buộc khi mode='batch'")

        # Nếu excel_path không phải file .xlsx, coi như thư mục và thêm filenames
        excel = args.excel_path
        if not excel.lower().endswith(('.xls', '.xlsx')):
            os.makedirs(excel, exist_ok=True)
            if args.cam_method == "cluster":
                excel = os.path.join(excel, f"pca2_{args.model}_{args.cam_method}_zero-out-{args.zero_ratio}_temperature-{args.temperature}.xlsx")
            else:
                excel = os.path.join(excel, f"{args.model}__{args.cam_method}.xlsx")

        batch_test(
            model,
            model_dict,
            args.dataset,
            excel,
            args.k_values,
            cam_method=args.cam_method,
            top_n=args.top_n,
            model_name=args.model
        )


if __name__ == '__main__':
    main()
