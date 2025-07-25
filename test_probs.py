import os
import torch
import torchvision.models as models
from torchvision.models import (
    inception_v3, Inception_V3_Weights,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights, efficientnet_b0, EfficientNet_B0_Weights,
    vit_b_16, ViT_B_16_Weights,
    swin_b, Swin_B_Weights,
)

from args import get_args

from find_prob import batch_test
from models.alzheimer_resnet18.alzheimer_resnet18 import load_model


def reshape_transform(tensor, height=14, width=14):
    # ViT-specific reshape
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # bring channels to first dim
    result = result.transpose(2, 3).transpose(1, 2)
    return result


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
        target_layer = model.layer4
    elif args.model == 'alzheimer_resnet18':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(
            checkpoint_path="/home/infres/xnguyen-24/cluster_cam/models/alzheimer_resnet18/alzheimer_resnet18.pth",
            device=device
        )
        model.eval()
        input_size = (128, 128)
        target_layer = model.layer4
    elif args.model == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.eval()
        input_size = (224, 224)
        target_layer = model.features[28]
    elif args.model == 'inception_v3':
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.eval()
        input_size = (299, 299)
        target_layer = model.Mixed_7c
    elif args.model == 'efficientNet':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.eval()
        input_size = (224, 224)
        target_layer = model.features[-1]
    elif args.model == 'vit_b_16':
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.eval()
        input_size = (224, 224)
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
            raise RuntimeError("--img-path là bắt buộc khi mode='single'" )
        drop, inc = test_single_image(
            model,
            model_dict,
            args.img_path,
            args.save_prefix,
            cam_method=args.cam_method,
            num_clusters=args.num_clusters
        )
        print(f"Average Drop: {drop:.4f}, Increase Confidence: {inc:.4f}")

    # Chạy chế độ batch probabilities
    elif args.mode == 'batch':
        if not args.dataset or not args.excel_path:
            raise RuntimeError("--dataset và --excel-path là bắt buộc khi mode='batch_probs'")
        excel = args.excel_path
        # đảm bảo đường dẫn file
        if os.path.isdir(excel):
            os.makedirs(excel, exist_ok=True)
            excel = os.path.join(excel, f"all_models_probs.xlsx")
        batch_test(
            model,
            args.model,
            args.dataset,
            excel,
            top_n=args.top_n,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
    else:
        raise ValueError(f"Mode '{args.mode}' không hỗ trợ")


if __name__ == '__main__':
    main()
