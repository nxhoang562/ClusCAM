import os
import torch
import pandas as pd
from torchvision import transforms
import torch.nn.functional as F

from utils_folder import load_image, list_image_paths
from torchvision.models import VGG
import torch.nn as nn

rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gray_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def get_transform_for_model(model):
    if hasattr(model, "conv1"):
        in_ch = model.conv1.in_channels
    elif isinstance(model, VGG):
        first_conv = next(
            (m for m in model.features if isinstance(m, nn.Conv2d)),
            None
        )
        in_ch = first_conv.in_channels if first_conv is not None else 3
    else:
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        in_ch = conv_layers[0].in_channels if conv_layers else 3
    return gray_transform if in_ch == 1 else rgb_transform


def batch_test(
    model,
    model_name: str,
    dataset: str,
    excel_path: str,
    top_n: int = None,
    start_idx: int = None,
    end_idx: int = None
):
    """
    Run the model over images in `dataset` and save top-1 probability per image
    into an Excel file at `excel_path`, using a sheet named `model_name`.
    """
    # Device and eval mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Gather image paths
    all_paths = list_image_paths(dataset)
    if not all_paths:
        raise RuntimeError(f"No images found in {dataset}")

    # Select subset if requested
    if start_idx is not None and end_idx is not None:
        image_paths = all_paths[start_idx:end_idx]
    else:
        image_paths = all_paths if top_n is None else all_paths[:top_n]

    # Prepare transform and storage
    transform = get_transform_for_model(model)
    top1_idxs = []
    pred_probs = []

    # Inference loop
    with torch.no_grad():
        total = len(image_paths)
        for idx, path in enumerate(image_paths, 1):
            print(f"Processing image {idx}/{total}: {os.path.basename(path)}")  # progress print
            img = load_image(path)
            inp = transform(img).unsqueeze(0).to(device)
            logits = model(inp)
            probs = F.softmax(logits, dim=1)
            cls_idx = int(probs.argmax(dim=1))
            p = float(probs[0, cls_idx])
            top1_idxs.append(cls_idx)
            pred_probs.append(p)

    # Build DataFrame
    df = pd.DataFrame({
        "image_path": image_paths,
        "top1_index": top1_idxs,
        "predicted_probability": pred_probs
    })

    # Write to Excel (append or create)
    mode = "a" if os.path.exists(excel_path) else "w"

    # Write to Excel with correct if_sheet_exists usage
    if mode == "a":
        with pd.ExcelWriter(excel_path,
                            engine="openpyxl",
                            mode=mode,
                            if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=model_name, index=False)
    else:
        with pd.ExcelWriter(excel_path,
                            engine="openpyxl",
                            mode=mode) as writer:
            df.to_excel(writer, sheet_name=model_name, index=False)

    print(f"Saved sheet '{model_name}' in {excel_path}")
