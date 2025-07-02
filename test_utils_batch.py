import os
import torch
import pandas as pd
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torchvision.models import VGG
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import load_image, basic_visualize, list_image_paths
from metrics.average_drop import AverageDrop
from metrics.average_increase import AverageIncrease
from cam.metacam import ClusterScoreCAM
from cam.polycam import PCAMp, PCAMm, PCAMpm


from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM,
    AblationCAM, ShapleyCAM
)

# -- Transforms for different input channels --
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


def get_transform_for_model(model: nn.Module) -> transforms.Compose:
    if hasattr(model, "conv1"):
        in_ch = model.conv1.in_channels
    elif isinstance(model, VGG):
        first_conv = next((m for m in model.features if isinstance(m, nn.Conv2d)), None)
        in_ch = first_conv.in_channels if first_conv is not None else 3
    else:
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        in_ch = convs[0].in_channels if convs else 3
    return gray_transform if in_ch == 1 else rgb_transform


class ImageFolderDataset(Dataset):
    """
    Dataset load ảnh từ folder và apply transform.
    Trả về tuple (tensor, đường dẫn).
    """
    def __init__(self, image_paths: list[str], transform: transforms.Compose):
        self.paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.paths[idx]
        img = load_image(path)
        tensor = self.transform(img)
        return tensor, path


# Mapping method name -> CAM constructor
CAM_FACTORY = {
    "cluster": lambda md, num_clusters=None: ClusterScoreCAM(
        md,
        num_clusters=num_clusters,
        zero_ratio=md.get("zero_ratio", 0.5),
        temperature=md.get("temperature", 1.0)
    ),
    "gradcam": lambda md, **kw: GradCAM(model=md["arch"], target_layers=[md["target_layer"]], **kw),
    "gradcamplusplus": lambda md, **kw: GradCAMPlusPlus(model=md["arch"], target_layers=[md["target_layer"]], **kw),
    "layercam": lambda md, **kw: LayerCAM(model=md["arch"], target_layers=[md["target_layer"]], **kw),
    "scorecam": lambda md, **kw: ScoreCAM(model=md["arch"], target_layers=[md["target_layer"]], **kw),
    "ablationcam": lambda md, **kw: AblationCAM(model=md["arch"], target_layers=[md["target_layer"]], **kw),
    "shapleycam": lambda md, **kw: ShapleyCAM(model=md["arch"], target_layers=[md["target_layer"]], **kw),
    
     "polyp": lambda md, **kw: PCAMp(
        model=md["arch"],
        target_layer_list=md.get("target_layer_list"),
        batch_size=md.get("batch_size", 32),
        intermediate_maps=md.get("intermediate_maps", False),
        lnorm=md.get("lnorm", True)
    ),
    "polym": lambda md, **kw: PCAMm(
        model=md["arch"],
        target_layer_list=md.get("target_layer_list"),
        batch_size=md.get("batch_size", 32),
        intermediate_maps=md.get("intermediate_maps", False),
        lnorm=md.get("lnorm", True)
    ),
    "polypm": lambda md, **kw: PCAMpm(
        model=md["arch"],
        target_layer_list=md.get("target_layer_list"),
        batch_size=md.get("batch_size", 32),
        intermediate_maps=md.get("intermediate_maps", False),
        lnorm=md.get("lnorm", True)
    ),
    
}


def batch_test(
    model: nn.Module,
    model_dict: dict,
    dataset: str,
    excel_path: str,
    k_values: list[int],
    cam_method: str = "cluster",
    top_n: int | None = None,
    model_name: str | None = None,
    batch_size: int = 16,
    num_workers: int = 4
):
    """
    Vectorized batch_test: tính CAM per-batch (với fallback per-sample cho cluster CAM) và metrics batch.
    """
    if model_name is None:
        raise ValueError("batch_test: cần truyền model_name (chuỗi)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # 1. Lấy danh sách ảnh
    all_paths = list_image_paths(dataset)
    if not all_paths:
        raise RuntimeError(f"No images found in {dataset}")
    image_paths = all_paths if top_n is None else all_paths[:top_n]

    # 2. Dataset & DataLoader
    transform = get_transform_for_model(model)
    ds = ImageFolderDataset(image_paths, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 3. Chuẩn bị file Excel
    if os.path.isdir(excel_path):
        excel_dir = excel_path; excel_filename = "results.xlsx"
    else:
        excel_dir = os.path.dirname(excel_path) or '.'; excel_filename = os.path.basename(excel_path)
    model_dir = os.path.join(excel_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    full_path = os.path.join(model_dir, excel_filename)

    ks = k_values if cam_method == "cluster" else [None]
    for c in ks:
        sheet_name = cam_method if c is None else f"{cam_method}_K{c}"
        mode = "a" if os.path.exists(full_path) else "w"

        all_paths_out, all_preds, all_drops, all_incs = [], [], [], []

        # Khởi tạo CAM một lần
        if cam_method == "cluster":
            cam = CAM_FACTORY["cluster"](model_dict, num_clusters=c)
        else:
            cam = CAM_FACTORY[cam_method](model_dict)

        # 4. Lặp batch
        num_batches = len(loader)
        for batch_idx, (batch_imgs, batch_paths) in enumerate(loader, start=1):
            print(f"[{cam_method.upper()}] Model={model_name} | Sheet={sheet_name} | Batch {batch_idx}/{num_batches}")
            
            batch_imgs = batch_imgs.to(device)
            # Predict batch
            with torch.no_grad():
                logits = model(batch_imgs)
                preds = logits.argmax(1)

            # Tính saliency maps
            if cam_method == "cluster":
                sal_list = []
                for i, (img, cls) in enumerate(zip(batch_imgs, preds), start=1):
                    print(f"    Sample {i}/{batch_imgs.size(0)}: computing CAM for class {int(cls)}")
                    sal = cam(img.unsqueeze(0), class_idx=int(cls))
                    if sal.dim() == 3:
                        sal = sal.unsqueeze(1)
                    sal_list.append(sal.cpu())
                sal_batch = torch.cat(sal_list, dim=0).to(device)
            elif cam_method in ("polyp", "polym", "polypm"):
                sal_list = []
                for i, (img, cls) in enumerate(zip(batch_imgs, preds), start=1):
                    print(f"    Sample {i}/{batch_imgs.size(0)}: PolyCAM {cam_method} for class {int(cls)}")
                    maps = cam(img.unsqueeze(0), class_idx=int(cls))
                    sal = maps[-1] if isinstance(maps, (list, tuple)) else maps
                    if sal.dim() == 3:
                        sal = sal.unsqueeze(1)
                    sal_list.append(sal.cpu())
                sal_batch = torch.cat(sal_list, dim=0).to(device)
                
            else:
                targets = [ClassifierOutputTarget(int(p)) for p in preds]
                sal_np = cam(input_tensor=batch_imgs, targets=targets)
                sal_batch = torch.from_numpy(sal_np)
                if sal_batch.dim() == 3:
                    sal_batch = sal_batch.unsqueeze(1)
                sal_batch = sal_batch.to(device)

            # Expand to (B,C,H,W)
            sal3 = sal_batch.expand(-1, batch_imgs.size(1), -1, -1)

            # 5. Tính metrics
            batch_cls = preds.to(device)
            drops = AverageDrop()(model=model,
                                   test_images=batch_imgs,
                                   saliency_maps=sal3,
                                   class_idx=batch_cls,
                                   device=device,
                                   apply_softmax=True,
                                   return_mean=False)
            incs = AverageIncrease()(model=model,
                                      test_images=batch_imgs,
                                      saliency_maps=sal3,
                                      class_idx=batch_cls,
                                      device=device,
                                      apply_softmax=True,
                                      return_mean=False)

            # Lưu kết quả
            all_paths_out.extend(batch_paths)
            all_preds.extend(preds.cpu().tolist())
            all_drops.extend(drops.cpu().tolist())
            all_incs.extend(incs.cpu().tolist())

        # 6. Xuất Excel
        df = pd.DataFrame({
            "image_path": all_paths_out,
            "top1_index": all_preds,
            "average_drop": all_drops,
            "increase_confidence": all_incs,
        })
        avg_row = pd.DataFrame([{
            "image_path": "AVERAGE",
            "top1_index": "",
            "average_drop": float(np.mean(all_drops)),
            "increase_confidence": float(np.mean(all_incs))
        }])
        out_df = pd.concat([avg_row, df], ignore_index=True)

        with pd.ExcelWriter(full_path, engine="openpyxl", mode=mode,
                            if_sheet_exists=("replace" if mode=="a" else None)) as writer:
            out_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved sheet {sheet_name} in {full_path}")
