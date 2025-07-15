import torch
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from PIL import Image
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# import CAM_FACTORY from your batch_test utility
from utils_main import CAM_FACTORY


def load_image(img_path, device, size=224):
    """Load and preprocess image."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    float_img = np.array(img.resize((size, size))) / 255.0
    return tensor, float_img


class PScore:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, attribution_map, pseudo_gt):
        a = attribution_map.flatten().astype(np.float32)
        g = pseudo_gt.flatten().astype(np.float32)
        na = (a - a.min()) / (a.max() - a.min() + self.eps)
        ng = (g - g.min()) / (g.max() - g.min() + self.eps)
        return float(np.dot(na, ng) /
                     (np.linalg.norm(na) * np.linalg.norm(ng) + self.eps))


def compute_pseudo_gt(models, input_tensor, target_category,
                      k, zero_ratio, temperature):
    """
    Compute pseudo‑ground‑truth by averaging ClusterScoreCAM maps
    across all models, without going through find_layer.
    """
    cams = []
    for m in models:
        # directly pick the module for the final layer block
        target_layer = m.layer4[-1]
        md = {
            'arch': m,
            'target_layer': target_layer,
            'zero_ratio': zero_ratio,
            'temperature': temperature
        }
        # instantiate via CAM_FACTORY to avoid find_layer
        cam = CAM_FACTORY['cluster'](md, num_clusters=k)
        out = cam(input_tensor, targets=[ClassifierOutputTarget(target_category)])[0]
        
        np_out = out.detach().cpu().numpy().astype(np.float32)
        cams.append(np_out)

        try:
            cam.activations_and_grads.release()
        except Exception:
            pass

    arr = np.stack(cams, 0)
    normed = [(x - x.min()) / (x.max() - x.min() + 1e-8) for x in arr]
    return np.mean(np.stack(normed, 0), axis=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', required=True,
                        help='Paths to input images or directories')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Ending index (exclusive)')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--k', type=int, default=10,
                        help='num_clusters for ClusterScoreCAM')
    parser.add_argument('--zero_ratio', type=float, default=0.5,
                        help='zero_ratio for ClusterScoreCAM')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='temperature for ClusterScoreCAM')
    parser.add_argument('--output', type=str, default='cluster_pscore.xlsx',
                        help='Excel output file')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # --- build list of images ---
    all_images = []
    for p in args.images:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                fp = os.path.join(p, f)
                if os.path.isfile(fp):
                    all_images.append(fp)
        elif os.path.isfile(p):
            all_images.append(p)
        else:
            print(f"Warning: '{p}' không tồn tại, bỏ qua.")
    img_list = all_images[args.start_idx:args.end_idx]
    if not img_list:
        raise ValueError("Không có ảnh để xử lý.")

    # --- load models ---
    model_specs = {
        'resnet18': (models.resnet18, ResNet18_Weights.DEFAULT),
        'resnet34': (models.resnet34, ResNet34_Weights.DEFAULT),
        'resnet50': (models.resnet50, ResNet50_Weights.DEFAULT),
    }
    models_dict = {}
    for name, (ctor, weights) in model_specs.items():
        m = ctor(weights=weights).to(device).eval()
        models_dict[name] = m

    # --- benchmark chỉ ClusterScoreCAM ---
    results = {name: [] for name in model_specs}
    img_names = []

    for img_path in tqdm(img_list, desc='Processing'):
        img_names.append(os.path.basename(img_path))
        inp, _ = load_image(img_path, device)

        # chọn target class từ 1 model bất kỳ
        with torch.no_grad():
            pred = next(iter(models_dict.values()))(inp)
        tgt = pred.argmax(dim=1).item()

        # pseudo‑GT
        pseudo_gt = compute_pseudo_gt(
            list(models_dict.values()), inp, tgt,
            args.k, args.zero_ratio, args.temperature
        )

        # đánh giá mỗi model
        for name, m in models_dict.items():
            img_layer = m.layer4[-1]
            md = {
                'arch': m,
                'target_layer': img_layer,
                'zero_ratio': args.zero_ratio,
                'temperature': args.temperature
            }
            cam = CAM_FACTORY['cluster'](md, num_clusters=args.k)
            
            tmap = cam(inp, targets=[ClassifierOutputTarget(tgt)])[0]
            gmap = tmap.detach().cpu().numpy().astype(np.float32)
            try:
                cam.activations_and_grads.release()
            except Exception:
                pass
            score = PScore()(gmap, pseudo_gt)
            results[name].append(score)

    # --- ghi Excel ---
    df = pd.DataFrame(results, index=img_names)
    avg = df.mean().to_frame().T
    avg.index = ['Average']
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        pd.concat([avg, df], axis=0).to_excel(writer, sheet_name='ClusterScoreCAM')

    print(f"Xong {len(img_list)} ảnh. Lưu kết quả -> {args.output}")
