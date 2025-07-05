#!/usr/bin/env python3
"""
test_cluster_visualize.py

Visualize ClusterScoreCAM: lưu ảnh gốc được mask theo từng cluster và overlay saliency map.
"""

import os

import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

from utils import load_image, apply_transforms
from cam.metacam import ClusterScoreCAM


def visualize_clusters_and_saliency(
    img_path: str,
    model_dict: dict,
    out_dir: str,
    num_clusters: int = 5,
    device: str = None,
    save_clusters: bool = True
):
    """
    1) Load ảnh gốc (PIL) và resize về input_size
    2) Preprocess thành tensor normalized
    3) Chạy ClusterScoreCAM để lấy rep_maps và saliency map
    4) Lưu ảnh gốc * mask theo từng cluster
    5) Lưu overlay saliency map lên ảnh gốc
    """

    os.makedirs(out_dir, exist_ok=True)

    # 1) Load ảnh gốc và resize
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
    cam = ClusterScoreCAM(model_dict, num_clusters=num_clusters, zero_ratio=0.5, temperature = 0.5)
    sal = cam(inp)                         # (1,1,H,W), [0,1]
    rep_maps = cam.rep_maps.cpu().numpy()  # (K,H,W), [0,1]
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


if __name__ == "__main__":
    img_path = "/home/infres/xnguyen-24/cluster_cam/datasets/imagenet/n02279972_monarch.JPEG"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    ).to(device).eval()
    model_dict = {
        "type": "resnet18",
        "arch": resnet,
        "layer_name": "layer4",
        "input_size": (224, 224)
    }

   
    # Define k sets
    ks_save = list(range(3, 11)) + [15, 20, 25, 30]
    ks_no_save = list(range(35, 201, 5))

    # Loop for ks where we save cluster masks
    for k in ks_save:
        out_dir = f"visualizations/monarch/monarch_{k}_clusters"
        visualize_clusters_and_saliency(
            img_path=img_path,
            model_dict=model_dict,
            out_dir=out_dir,
            num_clusters=k,
            device=device,
            save_clusters=True
        )
        print(f"Done k={k} (with clusters), outputs → {out_dir}")

    # Loop for ks where we do _not_ save cluster masks
    for k in ks_no_save:
        out_dir = f"visualizations/monarch/monarch_{k}_clusters"
        visualize_clusters_and_saliency(
            img_path=img_path,
            model_dict=model_dict,
            out_dir=out_dir,
            num_clusters=k,
            device=device,
            save_clusters=False
        )
        print(f"Done k={k} (no clusters), outputs → {out_dir}")
