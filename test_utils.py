import os
import torch
import pandas as pd
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils import (
    load_image, apply_transforms, basic_visualize,
    list_image_paths, preprocess_image, predict_top1_indices
)
from cam.main_clusterscorecam import ClusterScoreCAM
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM,
    AblationCAM, ShapleyCAM
)
from metrics.average_drop import AverageDrop
from metrics.average_increase import AverageIncrease

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
    cam_method: phương pháp CAM, key trong CAM_FACTORY.
    num_clusters: chỉ dùng khi cam_method=="cluster".
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Khởi tạo CAM
    if cam_method == "cluster":
        cam = CAM_FACTORY["cluster"](model_dict, num_clusters=num_clusters)
    else:
        cam = CAM_FACTORY[cam_method](model_dict)

    # Load và preprocess ảnh
    img = load_image(img_path)
    inp = apply_transforms(img).to(device)

    # Dự đoán lớp top-1
    with torch.no_grad():
        logits = model(inp)
        target_cls = logits.argmax(dim=1).item()

    # Tính saliency map
    if cam_method == "cluster":
        sal_map = cam(inp, class_idx=target_cls).cpu().squeeze(0)
    else:
        # sal_map = cam(input_tensor=inp, targets=[target_cls]).cpu().squeeze(0)
        saliency_np = cam(input_tensor=inp, targets=[ClassifierOutputTarget(target_cls)])
        sal_map = torch.from_numpy(saliency_np).cpu().squeeze(0)

    # Lưu heatmap
    filename = f"{save_prefix}_{cam_method}"
    if cam_method == "cluster":
        filename += f"_K{num_clusters}"
    basic_visualize(inp.cpu().squeeze(0), sal_map, save_path=filename + ".png")

    # Mở rộng kênh cho metric (1,3,H,W)
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
    k_values, cam_method="cluster", top_n=100, 
    model_name=None
):
    """
    Test CAM trên nhiều ảnh và nhiều giá trị K (nếu cần), lưu kết quả vào Excel.
    """
    if model_name is None:
        raise ValueError("batch_test: cần truyền model_name (chuỗi)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    image_paths = list_image_paths(dataset)[:top_n]
    if not image_paths:
        raise RuntimeError(f"No images found in {dataset} (top_n={top_n})")
    top1_idxs = predict_top1_indices(image_paths, model, device)

    if os.path.isdir(excel_path):
        excel_dir = excel_path
        excel_filename = "results.xlsx"
    else:
        excel_dir = os.path.dirname(excel_path) or '.'
        excel_filename = os.path.basename(excel_path)

    # Tạo thư mục theo model
    model_dir = os.path.join(excel_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    full_path = os.path.join(model_dir, excel_filename)

    # Nếu không dùng cluster, ta chỉ cần lặp một lần
    ks = k_values if cam_method == "cluster" else [None]

    for c in ks:
        info = f"method={cam_method}" + (f", K={c}" if c else "")
        print(f"\n=== Testing {info} ===")
        drops, incs = [], []

        # Khởi tạo CAM cho mỗi giá trị
        if cam_method == "cluster":
            cam = CAM_FACTORY["cluster"](model_dict, num_clusters=c)
        else:
            cam = CAM_FACTORY[cam_method](model_dict)

        for idx, (path, cls) in enumerate(zip(image_paths, top1_idxs), 1):
            print(f"[{idx}/{len(image_paths)}] {os.path.basename(path)} -> class {cls}")
            img_tensor = preprocess_image(path, device)

            if cam_method == "cluster":
                sal_map = cam(img_tensor, class_idx=cls).cpu().squeeze(0)
            else:
                saliency_np = cam(input_tensor=img_tensor,  targets=[ClassifierOutputTarget(cls)])
                sal_map = torch.from_numpy(saliency_np).cpu().squeeze(0)

            sal3 = sal_map.unsqueeze(0).repeat(1, img_tensor.size(1), 1, 1)
            drops.append(AverageDrop()(model, img_tensor, sal3, cls, device, True))
            incs.append(AverageIncrease()(model, img_tensor, sal3, cls, device, True))

        # Tạo DataFrame và ghi Excel
        df = pd.DataFrame({
            "image_path": image_paths,
            "top1_index": top1_idxs,
            "average_drop": drops,
            "increase_confidence": incs,
        })
        avg_row = pd.DataFrame([{
            "image_path": "AVERAGE",
            "top1_index": "",
            "average_drop": np.mean(drops),
            "increase_confidence": np.mean(incs)
        }])
        df = pd.concat([avg_row, df], ignore_index=True)

        sheet_name = cam_method if c is None else f"{cam_method}_K{c}"
        mode = "a" if os.path.exists(excel_path) else "w"
        with pd.ExcelWriter(
            excel_path, engine="openpyxl", mode=mode,
            if_sheet_exists="replace" if mode=="a" else None
        ) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Saved sheet {sheet_name} in {excel_path}")
