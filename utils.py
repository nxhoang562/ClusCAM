import os
import torch
import pandas as pd
import numpy as np
from utils import (
    load_image, apply_transforms, basic_visualize,
    list_image_paths, preprocess_image, predict_top1_indices
)
from cluster_cam.cam.maine_clusterscorecam import ClusterScoreCAM
from metrics.average_drop import AverageDrop
from metrics.average_increase import AverageIncrease




def test_single_image(model, model_dict, img_path, save_prefix, num_clusters=5, device=None):
    """
    Chạy ClusterScoreCAM và tính metric AverageDrop, AverageIncrease cho 1 ảnh.
    Trả về tuple (drop, increase).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    cam = ClusterScoreCAM2(model_dict, num_clusters=num_clusters)
    avg_drop = AverageDrop()
    avg_inc = AverageIncrease()

    # Load và preprocess ảnh
    img = load_image(img_path)
    inp = apply_transforms(img).to(device)

    # Dự đoán lớp top-1
    with torch.no_grad():
        logits = model(inp)
        target_cls = logits.argmax(dim=1).item()

    # Tính saliency map
    sal_map = cam(inp, class_idx=target_cls).cpu().squeeze(0)

    # Lưu heatmap
    basic_visualize(
        inp.cpu().squeeze(0),
        sal_map,
        save_path=f"{save_prefix}_clusters{num_clusters}.png"
    )

    # Mở rộng kênh cho metric (1,3,H,W)
    sal3 = sal_map.unsqueeze(0).repeat(1, inp.size(1), 1, 1)

    # Tính metrics
    drop_val = avg_drop(
        model=model,
        test_images=inp,
        saliency_maps=sal3,
        class_idx=target_cls,
        device=device,
        apply_softmax=True,
        return_mean=True
    )
    inc_val = avg_inc(
        model=model,
        test_images=inp,
        saliency_maps=sal3,
        class_idx=target_cls,
        device=device,
        apply_softmax=True,
        return_mean=True
    )
    return drop_val, inc_val


def batch_test(model, model_dict, image_dir, excel_path, k_values, top_n=100):
    """
    Test ClusterScoreCAM trên nhiều ảnh và nhiều giá trị K, lưu kết quả vào Excel theo sheet.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    image_paths = list_image_paths(image_dir)[:top_n]
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir} (top_n={top_n})")
    top1_idxs = predict_top1_indices(image_paths, model, device)

    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    for c in k_values:
        print(f"\n=== Testing with K={c} ===")
        drops, incs = [], []
        for idx, (path, cls) in enumerate(zip(image_paths, top1_idxs), 1):
            print(f"[{idx}/{len(image_paths)}] {os.path.basename(path)} -> class {cls}")
            img_tensor = preprocess_image(path, device)
            sal_map = ClusterScoreCAM2(model_dict, num_clusters=c)(img_tensor, class_idx=cls).cpu().squeeze(0)
            sal3 = sal_map.unsqueeze(0).repeat(1, img_tensor.size(1), 1, 1)
            drop = AverageDrop()(model, img_tensor, sal3, cls, device, True)
            inc = AverageIncrease()(model, img_tensor, sal3, cls, device, True)
            drops.append(drop)
            incs.append(inc)

        # Tạo DataFrame và tính trung bình
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
        df = pd.concat([df, avg_row], ignore_index=True)

        # Ghi vào sheet Excel
        sheet = f"num_clusters_{c}"
        mode = "a" if os.path.exists(excel_path) else "w"
        with pd.ExcelWriter(
            excel_path, engine="openpyxl", mode=mode,
            if_sheet_exists="replace" if mode=="a" else None
        ) as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"Saved sheet {sheet} in {excel_path}")