#!/usr/bin/env python3
"""
baseline_visualize.py

Chỉ lưu overlay saliency map cho nhiều phương pháp CAM,
cả default từ pytorch_grad_cam và các custom CAM.
"""
import cv2
import os
import torch
import numpy as np
from glob import glob
from PIL import Image
import inspect
import torchvision.models as models

from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    ScoreCAM,
    AblationCAM,
    ShapleyCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Các phương pháp custom của bạn
from cam.opticam import Basic_OptCAM
from cam.polycam import PCAMpm
from cam.recipro_cam import ReciproCam

from utils import load_image, apply_transforms


def save_cam_overlay_only(
    cam_class,
    img_path: str,
    model: torch.nn.Module,
    target_layer,
    out_dir: str,
    device: str,
    method: str = "gradcam",
    class_idx: int = None
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load & preprocess ảnh
    img_pil = load_image(img_path)
    img_resized = img_pil.resize((224, 224), Image.BILINEAR)
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    input_tensor = apply_transforms(img_resized).to(device)

    # 2) Xác định class target (top-1 nếu None)
    if class_idx is None:
        with torch.no_grad():
            class_idx = int(model(input_tensor).argmax(dim=-1).item())
    targets = [ClassifierOutputTarget(class_idx)]

        # 3) Khởi tạo CAM và tính saliency map
    if method == "basic_optcam":
        # Chỉ test Basic_OptCAM
        cam = Basic_OptCAM(
            model=model,
            device=device,
            target_layer=[target_layer],   
            name_mode='resnet'             
        )
        # Basic_OptCAM.__call__ trả về (saliency_map, ...)
        out = cam(input_tensor, torch.tensor([class_idx], device=device))
        grayscale_cam = out[0] if isinstance(out, (tuple, list)) else out
        # Đảm bảo shape [H,W]
        if grayscale_cam.ndim == 4:
            grayscale_cam = grayscale_cam[0,0]
        grayscale_cam = grayscale_cam.detach().cpu().numpy()
    elif method == "pcampm":
        # PCAM+/- từ cam/plyocam.py

        target_layer_name = "layer4.1"
        cam = PCAMpm(
            model=model,
            target_layer_list=[target_layer_name ],
            batch_size=32,
            intermediate_maps=False,
            lnorm=True
        )
        # cam(...) trả về list saliency maps cho từng layer
        cam_maps = cam(input_tensor, torch.tensor([class_idx], device=device))
        # chọn map cuối cùng (layer cuối) để vẽ overlay
        grayscale_cam = cam_maps[-1]
        # nếu tensor vẫn require_grad, cần detach trước khi numpy
        if isinstance(grayscale_cam, torch.Tensor):
            grayscale_cam = grayscale_cam.detach().cpu().numpy()
        # nếu ra 4-D [1,1,H,W] thì hạ về [H,W]
        if grayscale_cam.ndim == 4:
            grayscale_cam = grayscale_cam[0, 0]
    elif method == "reciprocam":
        # 3.1) Tìm tên layer trong model tương ứng với target_layer object
        target_layer_name = None
        for name, module in model.named_modules():
            if module is target_layer:
                target_layer_name = name
                break

        # 3.2) Khởi tạo ReciproCam với đúng tên layer
        cam = ReciproCam(
            model=model,
            device=device,
            target_layer_name=target_layer_name
        )
        # 3.3) Tính saliency map, __call__ trả về (cam_map, class_idx)
        out = cam(input_tensor, torch.tensor([class_idx], device=device))
        # Nếu trả về tuple/list thì lấy phần tử đầu làm bản đồ
        grayscale_cam = out[0] if isinstance(out, (tuple, list)) else out
        # 3.4) Chuyển sang numpy nếu là Tensor
        if isinstance(grayscale_cam, torch.Tensor):
            grayscale_cam = grayscale_cam.detach().cpu().numpy()
        # 3.5) Chuẩn hoá shape về [H, W]
        if grayscale_cam.ndim == 4:      # [1,1,H,W]
            grayscale_cam = grayscale_cam[0, 0]
        elif grayscale_cam.ndim == 3:    # [1,H,W]
            grayscale_cam = grayscale_cam[0]
        
        h_img, w_img = img_np.shape[:2]
        grayscale_cam = cv2.resize(
            grayscale_cam,
            (w_img, h_img),
            interpolation=cv2.INTER_LINEAR
        )
        
    else:
        # Default pytorch_grad_cam API
        cam = cam_class(
            model=model,
            target_layers=[target_layer],
            reshape_transform=None
        )
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        # Cleanup hooks
        cam.activations_and_grads.release()

    # 4) Tạo overlay và lưu và lưu
    overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    out_path = os.path.join(out_dir, f"{method}.png")
    Image.fromarray(overlay).save(out_path)


if __name__ == "__main__":
    # --- Bật chạy custom methods hoặc default ---
    # USE_CUSTOM = True  # đặt True để chỉ chạy custom methods

    # --- Cấu hình model & layer ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    ).to(device).eval()
    target_layer = model.layer4[-1]

    # --- Thư mục ảnh ---
    IMG_FOLDER = "/home/infres/xnguyen-24/cluster_cam/datasets/test_multiobjects"
    IMAGE_PATHS = sorted(glob(os.path.join(IMG_FOLDER, "*.JPEG")))

    # --- Chọn phương pháp CAM ---
    # if USE_CUSTOM:
    #     cam_methods = {
    #         "basic_optcam": Basic_OptCAM,
    #         "pcampm": PCAMpm,
    #         "reciprocam": ReciproCam
    #     }
    # else:
    #     cam_methods = {
    #         "gradcam": GradCAM,
    #         "gradcamplusplus": GradCAMPlusPlus,
    #         "layercam": LayerCAM,
    #         "scorecam": ScoreCAM,
    #         "ablationcam": AblationCAM,
    #         "shapleycam": ShapleyCAM,
    #     }
    
    cam_methods = {
        "gradcam": GradCAM,
        "gradcamplusplus": GradCAMPlusPlus,
        "layercam": LayerCAM,
        "scorecam": ScoreCAM,
        "ablationcam": AblationCAM,
        "shapleycam": ShapleyCAM,
        "basic_optcam": Basic_OptCAM,
        "pcampm": PCAMpm,
        "reciprocam": ReciproCam
    }

    # --- Loop và lưu overlay ---
    for img_path in IMAGE_PATHS:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join("visualizations/baselines_Resnet18/multiobjects", img_name)
        for method_name, method_class in cam_methods.items():
            save_cam_overlay_only(
                cam_class=method_class,
                img_path=img_path,
                model=model,
                target_layer=target_layer,
                out_dir=out_dir,
                method=method_name,
                device=device
            )
            print(f"[✓] {img_name} — {method_name} overlay saved → {out_dir}")
