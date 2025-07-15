import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from PIL import Image
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# Grad-CAM imports
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM, AblationCAM, ShapleyCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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
        return float(np.dot(na, ng) / (np.linalg.norm(na) * np.linalg.norm(ng) + self.eps))


def compute_pseudo_gt(models, input_tensor, target_category, CamClass):
    """
    Compute pseudo-ground-truth by averaging CAM maps across models for a specific CamClass.
    Each model uses its layer4[-1] as the target layer.
    """
    cams = []
    for m in models:
        target_layer = m.layer4[-1]
        cam = CamClass(model=m, target_layers=[target_layer])
        out = cam(input_tensor, targets=[ClassifierOutputTarget(target_category)])[0]
        cams.append(out.astype(np.float32))
        try:
            cam.activations_and_grads.release()
        except Exception:
            pass

    arr = np.stack(cams, 0)
    # Normalize each map to [0,1]
    normed = [(x - x.min()) / (x.max() - x.min() + 1e-8) for x in arr]
    return np.mean(np.stack(normed, 0), axis=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', required=True,
                        help='Paths to input images or directories')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index (inclusive) of images to process')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Ending index (exclusive) of images to process')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--output', type=str, default='pscore_results.xlsx',
                        help='Excel output file')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Build full list of image file paths
    all_images = []
    for path in args.images:
        if os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                full = os.path.join(path, f)
                if os.path.isfile(full):
                    all_images.append(full)
        elif os.path.isfile(path):
            all_images.append(path)
        else:
            print(f"Warning: path '{path}' not found, skipping.")

    img_list = all_images[args.start_idx:args.end_idx]
    if not img_list:
        raise ValueError(f"No images to process [{args.start_idx}:{args.end_idx}]")

    # Prepare models with modern weights API
    model_specs = {
        'resnet18': (models.resnet18, ResNet18_Weights.DEFAULT),
        'resnet34': (models.resnet34, ResNet34_Weights.DEFAULT),
        'resnet50': (models.resnet50, ResNet50_Weights.DEFAULT),
    }
    pseudo_models = {}
    for name, (ctor, weights) in model_specs.items():
        m = ctor(weights=weights).to(device).eval()
        pseudo_models[name] = m

    cam_methods = {
        'GradCAM': GradCAM,
        'ScoreCAM': ScoreCAM,
        'GradCAM++': GradCAMPlusPlus,
        'LayerCAM': LayerCAM,
        'AblationCAM': AblationCAM,
        'ShapleyCAM': ShapleyCAM,
    }

    results = {cam_name: {mdl: [] for mdl in model_specs} for cam_name in cam_methods}
    img_names = []

    for img_path in tqdm(img_list, desc='Processing Images'):
        img_names.append(os.path.basename(img_path))
        inp, _ = load_image(img_path, device)

        # determine target class using one reference model
        ref = next(iter(pseudo_models.values()))
        print(f"Processing image: {img_path}")
        with torch.no_grad():
            pred = ref(inp)
        tgt = pred.argmax(dim=1).item()

        # compute pseudo-ground-truth for each CAM method
        pseudo_gts = {}
        for cam_name, CamClass in cam_methods.items():
            pseudo_gts[cam_name] = compute_pseudo_gt(
                list(pseudo_models.values()), inp, tgt, CamClass
            )

        # score each CAM method on each model
        for m_name, (ctor, weights) in model_specs.items():
            print(f"Evaluating {m_name} with {len(pseudo_gts)} CAM methods")
            model = ctor(weights=weights).to(device).eval()
            lyr = model.layer4[-1]
            for cam_name, CamClass in cam_methods.items():
                print(f"  Using {cam_name}...")
                cam = CamClass(model=model, target_layers=[lyr])
                gmap = cam(inp, targets=[ClassifierOutputTarget(tgt)])[0]
                try:
                    cam.activations_and_grads.release()
                except Exception:
                    pass
                score = PScore()(gmap, pseudo_gts[cam_name])
                results[cam_name][m_name].append(score)

    # export Excel
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        for cam_name, data in results.items():
            df = pd.DataFrame(data, index=img_names)
            avg = df.mean().to_frame().T
            avg.index = ['Average']
            pd.concat([avg, df], axis=0).to_excel(writer, sheet_name=cam_name)

    print(f"Done [{len(img_list)} images]. Saved -> {args.output}")
