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
from cam.metacam import ClusterScoreCAM
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
    mode: str = "test" # test or validation 
):
    """
    - Vectorized batch_test: tính CAM per-batch (với fallback per-sample cho cluster CAM) và metrics batch.
    - mode="test": tính metrics AverageDrop/Increase
    - mode="validation": tính mean_shift, std_shift cho mỗi ảnh
    """
    if model_name is None:
        raise ValueError("batch_test: cần truyền model_name (chuỗi)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # 1. Lấy danh sách ảnh
    all_paths = list_image_paths(dataset)
    if not all_paths:
        raise RuntimeError(f"No images found in {dataset}")
    
    if mode == "validation" and cam_method != "cluster":
        raise ValueError(f"Validation mode is only supported for 'cluster' CAM. Got cam_method='{cam_method}'")
    
    if start_idx is not None or end_idx is not None:
        image_paths = all_paths[start_idx:end_idx]
        
    elif top_n is not None:
        image_paths = all_paths[:top_n]
    else:
        image_paths = all_paths 

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
    
    
    # Với cluster mode có nhiều K, còn các CAM khác chỉ 1 lần
    ks = k_values if cam_method == "cluster" else [None]
    for c in ks:
        sheet_name = cam_method if c is None else f"{cam_method}_K{c}"
        mode_open = "a" if os.path.exists(full_path) else "w"

        # all_paths_out, all_preds, all_drops, all_incs = [], [], [], []

        # Khởi tạo CAM một lần
        if cam_method == "cluster":
            cam = CAM_FACTORY["cluster"](model_dict, num_clusters=c)
        else:
            cam = CAM_FACTORY[cam_method](model_dict, device=device,)
            
        
        #Dữ liệu để lưu kết quả 
        records = []
        img_counter = 0

        # 4. Lặp batch
        for batch_idx, (batch_imgs, batch_paths) in enumerate(loader, start=1):
            print(f"[{cam_method.upper()}] {cam_method} | Model={model_name} | k = {c} | Batch {batch_idx}/{len(loader)}")
            batch_imgs = batch_imgs.to(device)
            # Predict batch
            with torch.no_grad():
                logits = model(batch_imgs)
                preds = logits.argmax(1)
            
            # Với mỗi ảnh trong batch
            for i in range(batch_imgs.size(0)):
                img_counter += 1
                img = batch_imgs[i:i+1]
                path = batch_paths[i]
                cls = int(preds[i])
                print(f"  Processing image {img_counter}/{len(image_paths)}: {path} (class={cls})")
                
            #1, Tính saliency maps
                if cam_method == "cluster":
                    sal = cam(img, class_idx=cls)
                elif cam_method in ("polyp", "polym", "polypm"):
                    maps = cam(img, class_idx=cls)
                    sal = maps[-1] if isinstance(maps, (list, tuple)) else maps
                    if sal.dim() == 3:
                        sal = sal.unsqueeze(1)
                
                elif cam_method == "opticam":
                    sal, _ = cam(img, torch.tensor([cls], device=device))
                    if sal.dim() == 3:
                        sal = sal.unsqueeze(1)

                elif cam_method == "reciprocam":
                    sal_map, _ = cam(img, index=cls)
                    if isinstance(sal_map, list):
                        sal_map = sal_map[0]
                    sal = torch.from_numpy(sal_map) if isinstance(sal_map, np.ndarray) else sal_map
                    sal = sal.unsqueeze(0).unsqueeze(0)
                    H, W = img.size(2), img.size(3)
                    sal = F.interpolate(sal, size=(H, W), mode='bilinear', align_corners=False)

                else:
                    targets = [ClassifierOutputTarget(cls)]
                    sal_np = cam(input_tensor=img, targets=targets)
                    sal = torch.from_numpy(sal_np)
                    if sal.dim() == 3:
                        sal = sal.unsqueeze(1)
                    sal = sal.to(device)
                
                # Compute records
                if mode == "validation":
                    # Tính mean shift và std shift
                    orig_score = logits[i, cls].item()
                    masked = img * sal
                    with torch.no_grad():
                        lm = model(masked)
                    masked_score = lm[0, cls].item()
                    shift = masked_score - orig_score
                    print(f"    Shift: {shift:.4f}")
                    records.append({
                        "image_path": path,
                        "top1_index": cls,
                        "shift": shift
                    })
                else:
                    sal3 = sal.expand(-1, batch_imgs.size(1), -1, -1)
                    drop = AverageDrop()(model=model,
                                        test_images=img,
                                        saliency_maps=sal3,
                                        class_idx=torch.tensor([cls], device=device),
                                        device=device,
                                        apply_softmax=True,
                                        return_mean=True).item()
                    inc = AverageIncrease()(model=model,
                                            test_images=img,
                                            saliency_maps=sal3,
                                            class_idx=torch.tensor([cls], device=device),
                                            device=device,
                                            apply_softmax=True,
                                            return_mean=True).item()
                    print(f"    Drop: {drop:.4f}, Increase: {inc:.4f}")
                    records.append({
                        "image_path": path,
                        "top1_index": cls,
                        "average_drop": drop,
                        "increase_confidence": inc
                    })
                
                
            
            
        # 5. Xuất Excel
        df = pd.DataFrame(records)
        if mode == "validation":
            mean_shift = df["shift"].mean()
            std_shift = df["shift"].std()
            summary = pd.DataFrame([{
                "image_path": "AVERAGE",
                "top1_index": "",
                "shift": mean_shift,
                "std_shift": std_shift
            }])
            out_df = pd.concat([summary, df], ignore_index=True)
            print(f"Summary shift — mean: {mean_shift:.4f}, std: {std_shift:.4f}")
        else:
            avg_drop = df["average_drop"].mean()
            avg_inc = df["increase_confidence"].mean()
            summary = pd.DataFrame([{
                "image_path": "AVERAGE",
                "top1_index": "",
                "average_drop": avg_drop,
                "increase_confidence": avg_inc
            }])
            out_df = pd.concat([summary, df], ignore_index=True)
            print(f"Summary metrics — drop: {avg_drop:.4f}, increase: {avg_inc:.4f}")

        with pd.ExcelWriter(full_path, engine="openpyxl", mode=mode_open,
                            if_sheet_exists=("replace" if mode_open=="a" else None)) as writer:
            out_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved sheet {sheet_name} to {full_path}")   