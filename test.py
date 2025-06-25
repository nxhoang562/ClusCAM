import torchvision.models as models
from args import get_args
from utils import test_single_image, batch_test


def main():
    args = get_args()

    # Load model
    if args.model == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Model {args.model} không hỗ trợ")
    model.eval()

    model_dict = {
        'type': args.model,
        'arch': model,
        'layer_name': args.layer_name,
        'input_size': (224, 224)
    }

    if args.mode == 'single':
        if not args.img_path:
            raise RuntimeError("--img-path là bắt buộc khi mode='single'")
        drop, inc = test_single_image(
            model, model_dict,
            args.img_path, args.save_prefix,
            num_clusters=args.num_clusters
        )
        print(f"Average Drop: {drop:.4f}, Increase Confidence: {inc:.4f}")
    else:
        if not args.image_dir or not args.excel_path:
            raise RuntimeError("--image-dir và --excel-path là bắt buộc khi mode='batch'")
        batch_test(
            model, model_dict,
            args.image_dir,
            args.excel_path,
            args.k_values,
            top_n=args.top_n
        )
        
if __name__ == '__main__':
  main()

