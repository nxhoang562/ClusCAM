#!/usr/bin/env python3
"""
test_cluster_visualize.py
Visualize ClusterScoreCAM: lưu ảnh gốc được mask theo từng cluster và overlay saliency map.
Ngoài ra, lưu thông tin ground truth, nhãn dự đoán và xác suất dự đoán của model.
"""
import argparse
import os
import csv
from glob import glob
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

from utils_folder import load_image, apply_transforms
from cam.Cluscam import ClusterScoreCAM


def load_ground_truth(csv_path):
    """
    Đọc file CSV chứa các cột: filename (không có đuôi), label (ground truth)
    Trả về dict: {filename: label}
    """
    gt = {}
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt[row['filename']] = row['label']
    return gt


def visualize_clusters_and_saliency(
    img_path: str,
    model_dict: dict,
    out_dir: str,
    num_clusters: int = 5,
    device: str = None,
    save_clusters: bool = True,
    ground_truth: str = None,
    idx_to_labels: list = None
):
    """
    1) Load ảnh gốc (PIL) và resize về input_size
    2) Preprocess thành tensor normalized
    3) Chạy ClusterScoreCAM để lấy rep_maps và saliency map
    4) Lưu ảnh gốc * mask theo từng cluster
    5) Lưu overlay saliency map lên ảnh gốc
    6) Dự đoán nhãn, xác suất và lưu thông tin cùng ground truth
    """

    os.makedirs(out_dir, exist_ok=True)

    # 1) Load ảnh gốc và resize
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img_pil = load_image(img_path)
    img_resized = img_pil.resize(model_dict['input_size'], Image.BILINEAR)
    img_resized.save(os.path.join(out_dir, "original.png"))
    arr = np.array(img_resized).astype(np.float32) / 255.0  # (H,W,3), [0,1]

    # 2) Preprocess thành tensor normalized
    inp = apply_transforms(img_resized)  # (1,3,H,W), normalized
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inp = inp.to(device)

    # 3) Khởi tạo CAM và chạy
    cam = ClusterScoreCAM(model_dict, num_clusters=num_clusters, zero_ratio=0.5, temperature=0.5)
    _ = cam(inp)
    rep_maps = cam.rep_maps.cpu().numpy()  # (K,H,W), [0,1]
    sal = cam(inp)                         # (1,1,H,W), [0,1]
    sal_map = sal.squeeze().cpu().numpy()  # (H,W), [0,1]

    # 4) Lưu từng ảnh gốc * mask(cluster)
    if save_clusters:
        for k in range(num_clusters):
            mask = rep_maps[k]                   # (H,W)
            masked = (arr * mask[..., None])     # (H,W,3)
            out = (masked * 255).astype(np.uint8)
            Image.fromarray(out).save(
                os.path.join(out_dir, f"cluster_{k}.png")
            )

    # 5) Vẽ và lưu overlay saliency map lên ảnh gốc
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr)
    ax.imshow(sal_map, cmap='jet', alpha=0.5)
    ax.axis('off')
    fig.savefig(os.path.join(out_dir, "saliency_overlay.png"),
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # 6) Dự đoán nhãn & lưu thông tin
    # Forward pass để lấy logits
    model = model_dict['arch']
    model.eval()
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1).squeeze()
        top1_prob, top1_idx = torch.max(probs, dim=0)
    predicted_idx = top1_idx.item()
    predicted_label = idx_to_labels[predicted_idx] if idx_to_labels else str(predicted_idx)
    predicted_prob = float(top1_prob)

    # Lưu thông tin vào file text
    info_path = os.path.join(out_dir, "info.txt")
    with open(info_path, 'w') as f:
        f.write(f"ground_truth: {ground_truth}\n")
        f.write(f"predicted_label: {predicted_label}\n")
        f.write(f"predicted_probability: {predicted_prob:.4f}\n")


if __name__ == "__main__":
    # --- CLI args ---
    parser = argparse.ArgumentParser(
        description="Visualize ClusterScoreCAM và lưu metadata"
    )
    parser.add_argument(
        "--start-from", "-s",
        type=str,
        default=None,
        help="Tên file (không có đuôi) để bắt đầu"
    )
    parser.add_argument(
        "--annotations", "-a",
        type=str,
        default=None,
        help="Đường dẫn đến CSV chứa ground truth"
    )
    args = parser.parse_args()
    start_from = args.start_from
    annotation_csv = args.annotations

    # --- Load ground truth ---
    ground_truth_dict = load_ground_truth(annotation_csv)

    # --- Cấu hình model & labels ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
    labels = models.ResNet18_Weights.DEFAULT.meta['categories']
    model_dict = {
        "type": "resnet18",
        "arch": resnet,
        "layer_name": "layer4",
        "input_size": (224, 224)
    }

    # --- Thư mục ảnh ---
    IMG_FOLDER = "/home/infres/ltvo/ClusCAM/datasets/test"
    IMAGE_PATHS = sorted(glob(os.path.join(IMG_FOLDER, "*.JPEG")))

    # --- Xử lý ---
    started = (start_from is None)
    # ks_no_save = list(range(10, 100, 10))
    ks_no_save = [2]

    for img_path in IMAGE_PATHS:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        if not started:
            if img_name == start_from:
                started = True
            else:
                continue

        for k in ks_no_save:
            out_dir = f"visualizations/labels/singleobject/{img_name}/{img_name}_k{k}_clusters"
            visualize_clusters_and_saliency(
                img_path=img_path,
                model_dict=model_dict,
                out_dir=out_dir,
                num_clusters=k,
                device=device,
                save_clusters=False,
                ground_truth=ground_truth_dict.get(img_name, None),
                idx_to_labels=labels
            )
            print(f"[✓] {img_name} k={k} cluster")
