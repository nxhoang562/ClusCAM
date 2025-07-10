import os
import torch
import pandas as pd
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import load_image, list_image_paths
from metrics.average_drop import AverageDrop
from metrics.average_increase import AverageIncrease
from metrics.coherency import Coherency
from metrics.complexity import Complexity
from cam.Cluscam import ClusterScoreCAM

import torch.nn as nn
import torch.nn.functional as F

# -- Transforms for input images --
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

def get_transform_for_model(model: nn.Module) -> transforms.Compose:
    """
    Chọn transform dựa vào số channel đầu vào của model.
    """
    if hasattr(model, "conv1"):
        in_ch = model.conv1.in_channels
    else:
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        in_ch = convs[0].in_channels if convs else 3
    return gray_transform if in_ch == 1 else rgb_transform

class ImageFolderDataset(Dataset):
    """Load ảnh từ folder và apply transform. Trả về (tensor, path)."""
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


CAM_FACTORY = {
    "cluster": lambda md, num_clusters=None: ClusterScoreCAM(
        md,
        num_clusters=num_clusters,
        zero_ratio=md.get("zero_ratio", 0.5),
        temperature=md.get("temperature", 0.5)
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
    Chạy batch_test với Cluster CAM:
      - mode 'test': tính AverageDrop, AverageIncrease, Coherency, Complexity, ADCC
      - mode 'validation': tính shift mean/std
    """
    if model_name is None:
        raise ValueError("batch_test: cần truyền model_name (string)")
    if mode == "validation" and cam_method != "cluster":
        raise ValueError("Validation mode chỉ hỗ trợ 'cluster' CAM")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    all_paths = list_image_paths(dataset)
    if not all_paths:
        raise RuntimeError(f"No images found in {dataset}")

    # Lọc ảnh theo start/end/top_n
    if start_idx is not None or end_idx is not None:
        image_paths = all_paths[start_idx:end_idx]
    elif top_n is not None:
        image_paths = all_paths[:top_n]
    else:
        image_paths = all_paths

    # Chuẩn bị file Excel
    if os.path.isdir(excel_path):
        excel_dir, excel_filename = excel_path, "results.xlsx"
    else:
        excel_dir = os.path.dirname(excel_path) or "."
        excel_filename = os.path.basename(excel_path)
    model_dir = os.path.join(excel_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    full_path = os.path.join(model_dir, excel_filename)

    # Duyệt các giá trị K
    for c in k_values:
        sheet_name = "cluster" if c is None else f"cluster_K{c}"
        mode_open = "a" if os.path.exists(full_path) else "w"

        # Khởi tạo ClusterScoreCAM
        cam = CAM_FACTORY["cluster"](model_dict, num_clusters=c)

        # DataLoader cho cluster
        transform = get_transform_for_model(model)
        ds = ImageFolderDataset(image_paths, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

        records = []
        for batch_idx, (batch_imgs, _) in enumerate(loader, start=1):
            batch_imgs = batch_imgs.to(device)
            with torch.no_grad():
                logits = model(batch_imgs)
                preds  = logits.argmax(1)

            cls_list  = preds.cpu().tolist()
            sal_batch = cam(batch_imgs, class_idx=cls_list)  # (B,1,H,W)
            sal3      = sal_batch.expand(-1, batch_imgs.size(1), -1, -1)

            if mode == "test":
                drop_b  = AverageDrop()(model, batch_imgs, sal3, preds,
                                        device, apply_softmax=True, return_mean=True)
                inc_b   = AverageIncrease()(model, batch_imgs, sal3, preds,
                                             device, apply_softmax=True, return_mean=True)

                coher_b = Coherency()(
                    model=model,
                    test_images=batch_imgs,
                    saliency_maps=sal3,
                    class_idx=cls_list,
                    attribution_method=cam,
                    upsample_method=lambda *args, **kwargs: kwargs.get("attribution", args[0]),
                    device=device,
                    return_mean=True
                )  
                comp_b  = Complexity()(sal3, return_mean=True)
                adcc_b  = 3 / ((1/coher_b) + 1/(1-comp_b) + 1/(1 - drop_b/100))

                # Scale to percent
                coher_b *= 100
                comp_b  *= 100
                adcc_b  *= 100

                print(f"[TEST][CLUSTER] k={c} Batch {batch_idx}/{len(loader)} \
                      — drop: {drop_b:.4f}, inc: {inc_b:.4f}, \
                      coher: {coher_b:.4f}, comp: {comp_b:.4f}, adcc: {adcc_b:.4f}")

                records.append({
                    "batch_index": batch_idx,
                    "average_drop": float(drop_b),
                    "increase_confidence": float(inc_b),
                    "coherency": float(coher_b),
                    "complexity": float(comp_b),
                    "adcc": float(adcc_b)
                })
            else:
                # validation
                orig = logits[torch.arange(preds.size(0)), preds]
                masked = batch_imgs * sal_batch
                with torch.no_grad():
                    lm = model(masked)
                shifts = lm[torch.arange(preds.size(0)), preds] - orig
                mean_shift = shifts.mean().item()
                std_shift  = shifts.std().item()
                print(f"[VAL][CLUSTER] k={c} Batch {batch_idx} — μ={mean_shift:.4f}, σ={std_shift:.4f}")
                records.append({
                    "batch_index": batch_idx,
                    "mean_shift": mean_shift,
                    "std_shift": std_shift
                })

        # Xuất Excel
        df = pd.DataFrame(records)
        if mode == "validation":
            summary = pd.DataFrame([{"batch_index": "ALL",
                                     "mean_shift": df["mean_shift"].mean(),
                                     "std_shift": df["std_shift"].mean()}])
        else:
            summary = pd.DataFrame([{"batch_index": "ALL",
                                     "average_drop": df["average_drop"].mean(),
                                     "increase_confidence": df["increase_confidence"].mean(),
                                     "coherency": df["coherency"].mean(),
                                     "complexity": df["complexity"].mean(),
                                     "adcc": df["adcc"].mean()}])
        out_df = pd.concat([summary, df], ignore_index=True)

        with pd.ExcelWriter(full_path, engine="openpyxl", mode=mode_open,
                            if_sheet_exists=("replace" if mode_open=="a" else None)) as writer:
            out_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved sheet {sheet_name} to {full_path}")
