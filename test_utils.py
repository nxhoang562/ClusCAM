import os
import torch
import pandas as pd
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils import load_image, basic_visualize, list_image_paths
from cam.main_clusterscorecam import ClusterScoreCAM
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM,
    AblationCAM, ShapleyCAM
)
from metrics.average_drop import AverageDrop
from metrics.average_increase import AverageIncrease
from torchvision import transforms

# Định nghĩa transforms cho ảnh RGB và Grayscale
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
    """
    Tự động chọn pipeline transform:
    - Nếu conv1 chỉ nhận 1 channel (grayscale), dùng gray_transform
    - Ngược lại dùng rgb_transform
    """
    in_ch = model.conv1.in_channels
    return gray_transform if in_ch == 1 else rgb_transform


def predict_top1_indices(image_paths, model, device):
    """Dự đoán chỉ số lớp top-1 cho danh sách ảnh sử dụng transform phù hợp."""
    model = model.to(device).eval()
    transform = get_transform_for_model(model)
    top1 = []
    with torch.no_grad():
        for path in image_paths:
            img = load_image(path)
            inp = transform(img).unsqueeze(0).to(device)
            logits = model(inp)
            top1.append(logits.argmax(dim=1).item())
    return top1

# Mapping giữa tên method và hàm khởi tạo tương ứng
CAM_FACTORY = {
    "cluster": lambda md, num_clusters=None: ClusterScoreCAM(
        md,
        num_clusters=num_clusters,
        zero_ratio=md.get("zero_ratio", 0.5),
        temperature=md.get("temperature", 1.0)
    ),
    "gradcam": lambda md, **kw: GradCAM(
        model=md["arch"],
        target_layers=[getattr(md["arch"], md["layer_name"])],
        **kw
    ),
    "gradcamplusplus": lambda md, **kw: GradCAMPlusPlus(
        model=md["arch"],
        target_layers=[getattr(md["arch"], md["layer_name"])],
        **kw
    ),
    "layercam": lambda md, **kw: LayerCAM(
        model=md["arch"],
        target_layers=[getattr(md["arch"], md["layer_name"])],
        **kw
    ),
    "scorecam": lambda md, **kw: ScoreCAM(
        model=md["arch"],
        target_layers=[getattr(md["arch"], md["layer_name"])],
        **kw
    ),
    "ablationcam": lambda md, **kw: AblationCAM(
        model=md["arch"],
        target_layers=[getattr(md["arch"], md["layer_name"])],
        **kw
    ),
    "shapleycam": lambda md, **kw: ShapleyCAM(
        model=md["arch"],
        target_layers=[getattr(md["arch"], md["layer_name"])],
        **kw
    ),
}

def test_single_image(
    model, model_dict, img_path, save_prefix,
    cam_method="cluster", num_clusters=5, device=None
):
    """
    Chạy CAM được chọn và tính metric cho 1 ảnh.
    - Tự động chuyển sang grayscale nếu model chỉ nhận 1 channel.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Chọn transform phù hợp
    transform = get_transform_for_model(model)

    # Load và preprocess ảnh
    img = load_image(img_path)
    inp = transform(img).unsqueeze(0).to(device)  # shape [1,C,224,224]

    # Dự đoán lớp top-1
    with torch.no_grad():
        logits = model(inp)
        target_cls = logits.argmax(dim=1).item()

    # Tính saliency map
    if cam_method == "cluster":
        cam = CAM_FACTORY["cluster"](model_dict, num_clusters=num_clusters)
        sal_map = cam(inp, class_idx=target_cls).cpu().squeeze(0)
    else:
        cam = CAM_FACTORY[cam_method](model_dict)
        saliency_np = cam(input_tensor=inp, targets=[ClassifierOutputTarget(target_cls)])
        sal_map = torch.from_numpy(saliency_np).cpu().squeeze(0)

    # Lưu heatmap
    filename = f"{save_prefix}_{cam_method}"
    if cam_method == "cluster":
        filename += f"_K{num_clusters}"
    basic_visualize(inp.cpu().squeeze(0), sal_map, save_path=filename + ".png")

    # Mở rộng kênh cho metric (1,C,H,W)
    sal3 = sal_map.unsqueeze(0).repeat(1, inp.size(1), 1, 1)

    # Tính metrics
    drop_val = AverageDrop()(   
        model=model,
        test_images=inp,
        saliency_maps=sal3,
        class_idx=target_cls,
        device=device,
        apply_softmax=True,
        return_mean=True
    )
    inc_val = AverageIncrease()(   
        model=model,
        test_images=inp,
        saliency_maps=sal3,
        class_idx=target_cls,
        device=device,
        apply_softmax=True,
        return_mean=True
    )
    return drop_val, inc_val


def batch_test(
    model, model_dict, dataset, excel_path,
    k_values, cam_method="cluster", top_n=None, 
    model_name=None
):
    """
    Test CAM trên nhiều ảnh và nhiều giá trị K (nếu cần), lưu kết quả vào Excel.
    - Tự động xử lý grayscale nếu cần.
    """
    if model_name is None:
        raise ValueError("batch_test: cần truyền model_name (chuỗi)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Chọn transform phù hợp
    transform = get_transform_for_model(model)

    # Lấy toàn bộ đường dẫn ảnh
    all_paths = list_image_paths(dataset)
    if not all_paths:
        raise RuntimeError(f"No images found in {dataset}")
    image_paths = all_paths if top_n is None else all_paths[:top_n]

    # Dự đoán top1 cho toàn bộ ảnh
    top1_idxs = predict_top1_indices(image_paths, model, device)

    # Chuẩn bị lưu Excel
    if os.path.isdir(excel_path):
        excel_dir = excel_path
        excel_filename = "results.xlsx"
    else:
        excel_dir = os.path.dirname(excel_path) or '.'
        excel_filename = os.path.basename(excel_path)
    model_dir = os.path.join(excel_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    full_path = os.path.join(model_dir, excel_filename)

    ks = k_values if cam_method == "cluster" else [None]
    for c in ks:
        info = f"method={cam_method}" + (f", K={c}" if c else "")
        print(f"\n=== Testing {info} ===")
        drops, incs = [], []
        # Khởi tạo CAM
        if cam_method == "cluster":
            cam = CAM_FACTORY["cluster"](model_dict, num_clusters=c)
        else:
            cam = CAM_FACTORY[cam_method](model_dict)

        for idx, (path, cls) in enumerate(zip(image_paths, top1_idxs), 1):
            print(f"[{idx}/{len(image_paths)}] {os.path.basename(path)} -> class {cls}")
            img = load_image(path)
            img_tensor = transform(img).unsqueeze(0).to(device)

            if cam_method == "cluster":
                sal_map = cam(img_tensor, class_idx=cls).cpu().squeeze(0)
            else:
                saliency_np = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(cls)])
                sal_map = torch.from_numpy(saliency_np).cpu().squeeze(0)

            sal3 = sal_map.unsqueeze(0).repeat(1, img_tensor.size(1), 1, 1)
            drops.append(AverageDrop()(model, img_tensor, sal3, cls, device, True))
            incs.append(AverageIncrease()(model, img_tensor, sal3, cls, device, True))

        # Ghi kết quả vào Excel
        df = pd.DataFrame({
            "image_path": image_paths,
            "top1_index": top1_idxs,
            "average_drop": drops,
            "increase_confidence": incs,
        })
        avg_row = pd.DataFrame([{"image_path": "AVERAGE", "top1_index": "", "average_drop": np.mean(drops), "increase_confidence": np.mean(incs)}])
        df = pd.concat([avg_row, df], ignore_index=True)
        sheet_name = cam_method if c is None else f"{cam_method}_K{c}"
        mode = "a" if os.path.exists(full_path) else "w"
        with pd.ExcelWriter(full_path, engine="openpyxl", mode=mode, if_sheet_exists="replace" if mode=="a" else None) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Saved sheet {sheet_name} in {full_path}")
