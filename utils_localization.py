# utils_main.py

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
    ClassifierOutputReST
)
from metrics import EnergyPointGame, EnergyPointGame_Threshold, Local_Error, Local_Error_Binary, Local_Error_EnergyThreshold

from cam_setup import CAM_FACTORY  # giả sử bạn import từ module riêng
from utils_folder import load_image
from torchvision import transforms
from torchvision.models import VGG
import torch.nn as nn
from torchvision.models import VisionTransformer

# Transforms
rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

gray_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def get_transform_for_model(model):
    """
    Tự động chọn pipeline transform:
    - Nếu conv đầu tiên chỉ nhận 1 channel (grayscale), dùng gray_transform
    - Ngược lại dùng rgb_transform
    """
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


def list_image_paths(dataset_dir):
    """
    Trả về tất cả các đường dẫn ảnh trong folder.
    """
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    return [
        os.path.join(dataset_dir, fn)
        for fn in sorted(os.listdir(dataset_dir))
        if fn.lower().endswith(exts)
    ]


def load_bboxes_from_csv(csv_path):
    """
    Đọc CSV có cột ImageId và PredictionString,
    chỉ giữ những dòng PredictionString đúng 5 token (1 box),
    trả về dict { ImageId : [x1,y1,x2,y2] }.
    """
    df = pd.read_csv(csv_path)
    df['tokens'] = df['PredictionString'].str.split()
    df1 = df[df['tokens'].apply(len) == 5].copy()
    if df1.empty:
        raise RuntimeError("No single-box entries in bbox CSV.")
    df1['bbox'] = df1['tokens'].apply(lambda t: list(map(int, t[1:])))
    return df1.set_index('ImageId')['bbox'].to_dict()


def batch_test(
    model,
    model_dict,
    dataset,
    excel_path,
    k_values,
    bbox_csv=None,
    cam_method="cluster",
    top_n=None,
    start_idx=None,
    end_idx=None,
    model_name=None
):
    # 1. Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name is None:
        raise ValueError("batch_test: cần truyền model_name")
    model = model.to(device).eval()
    # ensure CAM_FACTORY uses the correct device
    model_dict['device'] = device

    # 2. Load bounding boxes if provided
    if bbox_csv:
        bbox_dict = load_bboxes_from_csv(bbox_csv)
        allowed_ids = set(bbox_dict.keys())
    else:
        bbox_dict = {}
        allowed_ids = None

    # 3. Collect and optionally filter image paths
    all_paths = []
    for fn in sorted(os.listdir(dataset)):
        stem = os.path.splitext(fn)[0]
        if allowed_ids and stem not in allowed_ids:
            continue
        if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            all_paths.append(os.path.join(dataset, fn))

    if not all_paths:
        raise RuntimeError(f"No images found in {dataset} after filtering.")

    # 4. Apply start/end or top_n slicing
    if start_idx is not None and end_idx is not None:
        image_paths = all_paths[start_idx:end_idx]
    else:
        image_paths = all_paths if top_n is None else all_paths[:top_n]

    # 5. Prepare parallel bbox list
    if bbox_csv:
        bboxes = [
            bbox_dict[os.path.splitext(os.path.basename(p))[0]]
            for p in image_paths
        ]
    else:
        bboxes = [None] * len(image_paths)

    # 6. Predict top1 indices
    transform = get_transform_for_model(model)
    top1_idxs = []
    with torch.no_grad():
        for path in image_paths:
            img = load_image(path)
            inp = transform(img).unsqueeze(0).to(device)
            logits = model(inp)
            top1_idxs.append(int(logits.argmax(dim=1).item()))

    # 7. Prepare Excel output
    if os.path.isdir(excel_path):
        excel_dir = excel_path
        excel_fn = "results.xlsx"
    else:
        excel_dir = os.path.dirname(excel_path) or '.'
        excel_fn = os.path.basename(excel_path)
    os.makedirs(excel_dir, exist_ok=True)
    full_path = os.path.join(excel_dir, excel_fn)

    ks = k_values if cam_method == "cluster" else [None]
    for c in ks:
        print(f"\n=== Testing method={cam_method}" + (f", K={c}" if c else "") + " ===")

        energys = []
        energy_thrs = []
        local_erros = []
        local_binary_erros = []
        local_threshold_erros = []
        
        
        if cam_method in ("polyp","polym","polypm") \
        and model_dict.get("target_layer_list") is None \
        and isinstance(model, VisionTransformer):
            # hook sau self-attention của block cuối
            model_dict["target_layer_list"] = ["conv_proj"]
        
        if cam_method in ("reciprocam") \
        and model_dict.get("target_layer_list") is None \
        and isinstance(model, VisionTransformer):
            model_dict["target_layer_list"] = model.encoder.layers[-1].ln_2

        # 8. Initialize CAM and move it to the correct device
        if cam_method == "cluster":
            cam = CAM_FACTORY["cluster"](model_dict, num_clusters=c)
        elif cam_method == "reciprocam":
            cam = CAM_FACTORY["reciprocam"](model_dict)
            cam.model    = cam.model.to(device)       # weights lên GPU nếu device là 'cuda'
            cam.gaussian = cam.gaussian.to(device)    # filter lên GPU
            cam.device   = device   
        else:
            cam = CAM_FACTORY[cam_method](model_dict)
        cam.model = cam.model.to(device)

        # 9. Loop over images
        for idx, (path, cls) in enumerate(zip(image_paths, top1_idxs), start=1):
            print(f"[{idx}/{len(image_paths)}] {os.path.basename(path)} -> class {cls}")
            img = load_image(path)
            img_tensor = transform(img).unsqueeze(0).to(device)

            # compute saliency map
            if cam_method == "cluster":
                sal_map = cam(img_tensor, class_idx=cls).cpu().squeeze(0).squeeze(0)
                print(sal_map.shape)
            
            elif cam_method in ["polyp", "polym", "polypm"]:
                out = cam(img_tensor, class_idx=cls)   # out là list hoặc tensor
                # nếu là list thì lấy phần tử cuối
                if isinstance(out, (list, tuple)):
                    out = out[-1]
                # bây giờ out có thể là numpy array hoặc torch.Tensor
                if isinstance(out, np.ndarray):
                    sal_map = torch.from_numpy(out)
                else:
                    sal_map = out
                sal_map = sal_map.cpu().squeeze(0).squeeze(0)
            
            elif cam_method == "opticam":
                label_tensor = torch.tensor([cls], device=device)
                norm_map, _ = cam(img_tensor, label_tensor)
                sal_map = norm_map.cpu().squeeze(0).squeeze(0)
            
            elif cam_method == "reciprocam":
                out_cam, _ = cam(img_tensor, index=cls)
                # Nếu out_cam là numpy array thì convert, rồi đưa lên CPU
                sal_map = (
                    torch.from_numpy(out_cam)
                    if isinstance(out_cam, np.ndarray)
                    else out_cam
                ).cpu()
                
                tempt = sal_map.unsqueeze(0).unsqueeze(0)
                
                if tempt.shape[-2:] != img_tensor.shape[-2:]:
                    sal_map = F.interpolate(
                        tempt,
                        size=img_tensor.shape[-2:],      # (224, 224)
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                
            elif cam_method == "shapleycam":
                saliency_np = cam(input_tensor=img_tensor, targets=[ClassifierOutputReST(cls)])
                sal_map = torch.from_numpy(saliency_np).cpu().squeeze(0)
            else:
                saliency_np = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(cls)])
                sal_map = torch.from_numpy(saliency_np).cpu().squeeze(0)
                print(sal_map.shape)
            
                
            # compute metrics if bbox provided
            if bbox_csv:
                bbox = bboxes[idx - 1]
                
                energy = EnergyPointGame(bbox,  sal_map)
                energys.append(float(energy.item()))
                
                energy_thr = EnergyPointGame_Threshold(bbox,  sal_map, threshold=0.5)
                energy_thrs.append(float(energy_thr))

                local_erro = Local_Error(bbox,  sal_map)
                local_erros.append(float(local_erro))
                
                local_binary_erro = Local_Error_Binary(bbox, sal_map, thr = 0.5)
                local_binary_erros.append(float(local_binary_erro))
                
                local_threshold_erro = Local_Error_EnergyThreshold(bbox, sal_map, thr = 0.5)
                local_threshold_erros.append(float(local_threshold_erro))

        # 10. Build DataFrame and write to Excel
        data = {
            "image_path": image_paths,
            "top1_index": top1_idxs,
        }
        if bbox_csv:
            data["energy@bbox"] = energys
            data["energy@bbox_threshold"] =  energy_thrs
            data["local_erro"] = local_erros
            data["local_binary_erro"] = local_binary_erros
            data["local_threshold_erros"] = local_threshold_erros

        df = pd.DataFrame(data)
        # add average row
        avg_row = {
            "image_path": "AVERAGE",
            "top1_index": "",
        }
        if bbox_csv:
            avg_row["energy@bbox"] = np.mean(energys)
            avg_row["energy@bbox_threshold"] = np.mean(energy_thrs)
            avg_row["local_erro"] = np.mean(local_erros)
            avg_row["local_binary_erro"] = np.mean(local_binary_erros)
            avg_row["local_threshold_erros"] = np.mean(local_threshold_erros)
            

        df = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)

        sheet = cam_method if c is None else f"{cam_method}_K{c}"
        mode = "a" if os.path.exists(full_path) else "w"
        with pd.ExcelWriter(full_path, engine="openpyxl", mode=mode,
                            if_sheet_exists=("replace" if mode == "a" else None)) as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)

        print(f"Saved sheet {sheet} in {full_path}")
