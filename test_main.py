import os
import torch
import torchvision.models as models
from torchvision.models import (
    inception_v3, Inception_V3_Weights,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights, efficientnet_b0, EfficientNet_B0_Weights,
    vit_b_16, ViT_B_16_Weights,
    swin_b,  Swin_B_Weights,
)

from args import get_args


from utils_main import batch_test
from models.alzheimer_resnet18.alzheimer_resnet18 import load_model


# def reshape_transform(tensor, height=14, width=14):
#     # ViT-specific reshape
#     result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
#     # bring channels to first dim
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result


def main():
    args = get_args()

    # Thiết lập mapping cho các ResNet pretrained
    resnet_confs = {
        'resnet18': (models.resnet18, ResNet18_Weights),
        'resnet34': (models.resnet34, ResNet34_Weights),
        'resnet50': (models.resnet50, ResNet50_Weights),
        'resnet101': (models.resnet101, ResNet101_Weights),
        'resnet152': (models.resnet152, ResNet152_Weights),
    }

    # Load model động
    if args.model in resnet_confs:
        constructor, WeightsEnum = resnet_confs[args.model]
        model = constructor(weights=WeightsEnum.IMAGENET1K_V1)
        model.eval()
        input_size = (224, 224)
        target_layer = model.layer4  # ResNet sử dụng layer4 cho layer cuối
     
    elif args.model == 'alzheimer_resnet18':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(
            checkpoint_path="/home/infres/xnguyen-24/cluster_cam/models/alzheimer_resnet18/alzheimer_resnet18.pth",
            device=device
        )
        model.eval()
        target_layer = model.layer4   # Alzheimer ResNet18 sử dụng layer4 cho layer cuối
        input_size = (128, 128)
        
    elif args.model == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.eval()
        input_size = (224, 224)
        target_layer = model.features[28]  # VGG16 sử dụng features.28 cho layer cuối
      
    elif args.model == 'inception_v3':
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.eval()
        input_size = (299, 299)
        target_layer = model.Mixed_7c  # Inception V3 sử dụng Mixed_   
     
    elif args.model == 'efficientNet':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.eval()
        input_size = (224, 224)
        target_layer = model.features[-1] # EfficientNet sử dụng features cuối cùng là c
  
    elif args.model == 'vit_b_16':
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.eval()
        input_size = (224, 224)
        # target_layer = model.encoder.layers[-1].ln_1  # activations sau attention nhưng trước MLP
        try:
            target_layer = model.conv_proj
        except AttributeError:
            target_layer = model.patch_embed.proj  
    elif args.model == 'swin_b': 
        model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        model.eval()
        input_size = (224, 224)
        target_layer = model.features[0][0]
    else:
        raise ValueError(f"Model {args.model} không hỗ trợ")

    model_dict = {
        'type': args.model,
        'arch': model,
        'target_layer': target_layer,
        'input_size': input_size,
        'cam_method': args.cam_method,
        'zero_ratio': args.zero_ratio,
        'temperature': args.temperature,
    }

    # Chạy chế độ single image
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

    # Chạy chế độ batch
    else:
        if not args.dataset or not args.excel_path:
            raise RuntimeError("--dataset và --excel-path là bắt buộc khi mode='batch'")

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
            cam_method=args.cam_method,
            top_n=args.top_n,
            model_name=args.model, 
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )


if __name__ == '__main__':
    main()
