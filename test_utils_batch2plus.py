import os
import torch
import pandas as pd
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torchvision.models import VGG
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


from utils import load_image, basic_visualize, list_image_paths
from metrics.average_drop import AverageDrop
from metrics.average_increase import AverageIncrease
from cam.metacam2 import ClusterScoreCAM
from cam.polycam import PCAMp, PCAMm, PCAMpm
from cam.recipro_cam import ReciproCam

from torch.autograd import Variable
from cam.opticam import Basic_OptCAM, Preprocessing_Layer

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
    
     "opticam": lambda md, **kw: Basic_OptCAM(
        model=md["arch"],
        device=kw.get("device"),
        target_layer=md["target_layer"],
        max_iter=md.get("max_iter", 50),
        learning_rate=md.get("learning_rate", 0.1),
        name_f=md.get("name_f", "logit_predict"),
        name_loss=md.get("name_loss", "norm"),
        name_norm=md.get("name_norm", "max_min"),
        name_mode=md.get("name_mode", "resnet")
    ),
     "reciprocam": lambda md, **kw: ReciproCam(
        model=md["arch"],
        device=kw.get("device"),
        target_layer_name=md.get("target_layer_name", None)
    )
}


def batch_test(
    model: nn.Module,
    model_dict: dict,
    dataset: str,
    excel_path: str,
    k_values: list[int],
    cam_method: str = "cluster",
    top_n: int | None = None,
    start_idx: int | None = None,
    end_idx: int | None = None,
    model_name: str | None = None,
    batch_size: int = 16,
    num_workers: int = 4, 
    mode: str = "test"
):
    """
    Thực thi batch_test với tính toán metric ở cấp batch (GPU chạy batch).
    mode: "test" (AverageDrop/Increase) hoặc "validation" (mean/std shift)
    """
    if model_name is None:
        raise ValueError("batch_test: cần truyền model_name (chuỗi)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    all_paths = list_image_paths(dataset)
    if not all_paths:
        raise RuntimeError(f"No images found in {dataset}")
    if mode == "validation" and cam_method != "cluster":
        raise ValueError("Validation mode chỉ hỗ trợ 'cluster' CAM")

    # Lọc paths theo tham số
    if start_idx is not None or end_idx is not None:
        image_paths = all_paths[start_idx:end_idx]
    elif top_n is not None:
        image_paths = all_paths[:top_n]
    else:
        image_paths = all_paths

    # DataLoader
    transform = get_transform_for_model(model)
    ds = ImageFolderDataset(image_paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # Chuẩn bị file Excel
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
        mode_open = "a" if os.path.exists(full_path) else "w"

        # cam = CAM_FACTORY[cam_method](model_dict, num_clusters=c, device=device)
        if cam_method == "cluster":
            cam = CAM_FACTORY["cluster"](model_dict, num_clusters=c)
        else:
            cam = CAM_FACTORY[cam_method](model_dict, device=device)
            
        records = []

        # Xử lý từng batch
        for batch_idx, (batch_imgs, _) in enumerate(loader, start=1):
            batch_imgs = batch_imgs.to(device)
            with torch.no_grad():
                logits = model(batch_imgs)
                preds = logits.argmax(1)

            # CAM cho batch
            if cam_method == "cluster":
                cls_list = preds.detach().cpu().tolist()
                sal_batch = cam(batch_imgs, class_idx=cls_list)  # (B,1,H,W)
            else:
                # TODO: hỗ trợ batch cho các CAM khác tương tự
                raise NotImplementedError("Batch CAM chỉ implement cho 'cluster' hiện tại")

            if mode == "test":
                sal3 = sal_batch.expand(-1, batch_imgs.size(1), -1, -1)
                drop_b = AverageDrop()(model=model,
                                       test_images=batch_imgs,
                                       saliency_maps=sal3,
                                       class_idx=preds,
                                       device=device,
                                       apply_softmax=True,
                                       return_mean=True).item()
                inc_b = AverageIncrease()(model=model,
                                          test_images=batch_imgs,
                                          saliency_maps=sal3,
                                          class_idx=preds,
                                          device=device,
                                          apply_softmax=True,
                                          return_mean=True).item()
                print(f"[{cam_method.upper()}] k={c} Batch {batch_idx}/{len(loader)} — drop: {drop_b:.4f}, inc: {inc_b:.4f}")
                records.append({
                    "batch_index": batch_idx,
                    "average_drop": drop_b,
                    "increase_confidence": inc_b
                })
            else:
                orig = logits[torch.arange(preds.size(0)), preds]
                masked_imgs = batch_imgs * sal_batch
                with torch.no_grad():
                    lm = model(masked_imgs)
                masked_sc = lm[torch.arange(preds.size(0)), preds]
                shifts = masked_sc - orig
                mean_shift = shifts.mean().item()
                std_shift = shifts.std().item()
                print(f"[{cam_method.upper()}] k={c} Batch {batch_idx}/{len(loader)} — shift μ={mean_shift:.4f}, σ={std_shift:.4f}")
                records.append({
                    "batch_index": batch_idx,
                    "mean_shift": mean_shift,
                    "std_shift": std_shift
                })

        # Xuất kết quả
        df = pd.DataFrame(records)
        if mode == "validation":
            overall_mean = df["mean_shift"].mean()
            overall_std = df["std_shift"].mean()
            summary = pd.DataFrame([{"batch_index": "ALL", "mean_shift": overall_mean, "std_shift": overall_std}])
            out_df = pd.concat([summary, df], ignore_index=True)
            print(f"Validation summary — mean shift: {overall_mean:.4f}, mean std shift: {overall_std:.4f}")
        else:
            overall_drop = df["average_drop"].mean()
            overall_inc  = df["increase_confidence"].mean()
            summary = pd.DataFrame([{"batch_index": "ALL", "average_drop": overall_drop, "increase_confidence": overall_inc}])
            out_df = pd.concat([summary, df], ignore_index=True)
            print(f"Test summary — avg drop: {overall_drop:.4f}, avg increase: {overall_inc:.4f}")

        with pd.ExcelWriter(full_path, engine="openpyxl", mode=mode_open,
                            if_sheet_exists=("replace" if mode_open=="a" else None)) as writer:
            out_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved sheet {sheet_name} to {full_path}")
