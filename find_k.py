import os
import random
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights,
    inception_v3, Inception_V3_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    vit_b_16, ViT_B_16_Weights,
    swin_b,  Swin_B_Weights
)
from torchvision.models.inception import InceptionOutputs
from utils_folder import load_image, list_image_paths
from cam.basecam import BaseCAM
from sklearn.cluster import KMeans
import torch.nn.functional as F
from tqdm import tqdm  # progress bar
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn
class ClusterScoreCAM(BaseCAM):
    """
    Score-CAM với clustering, tuỳ chỉnh zero-out và temperature đối với một class nhất định.

    Args:
        model_dict: dict giống BaseCAM
        num_clusters: số cụm K
        zero_ratio: tỉ lệ phần trăm cluster nhỏ nhất bị loại (0-1)
        temperature_dict: dict class_idx, temperature
        default_temperature:
    """
    def __init__(
        self,
        model_dict,
        num_clusters=10,
        zero_ratio=0.5,
        temperature_dict=None,
        temperature=0.5
    ):
        super().__init__(model_dict)
        self.K = num_clusters
        self.zero_ratio = zero_ratio
        self.temperature_dict = temperature_dict or {}
        self.temperature = temperature

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        outputs = self.model_arch(input)
        if isinstance(outputs, InceptionOutputs):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            logits = outputs
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        elif isinstance(class_idx, torch.Tensor):
            class_idx = int(class_idx)
        base_score = logits[0, class_idx]
        self.model_arch.zero_grad()
        base_score.backward(retain_graph=retain_graph)
        activations = self.activations['value'][0]
        nc, u, v = activations.shape

        # Memory-efficient upsampling & normalization on CPU
        up_maps = []
        for i in range(nc):
            a = activations[i:i+1].unsqueeze(0)
            a_up = F.interpolate(a, size=(h, w), mode='bilinear', align_corners=False)[0, 0]
            a_up_cpu = a_up.detach().cpu()
            if a_up_cpu.max() != a_up_cpu.min():
                a_up_cpu = (a_up_cpu - a_up_cpu.min()) / (a_up_cpu.max() - a_up_cpu.min())
            up_maps.append(a_up_cpu.to(activations.device))
        up_maps = torch.stack(up_maps, dim=0)

        # Clustering on CPU
        flat_maps = up_maps.detach().cpu().reshape(nc, -1).numpy()
        kmeans = KMeans(n_clusters=self.K, init='k-means++', random_state=0)
        kmeans.fit(flat_maps)
        rep_maps = torch.from_numpy(kmeans.cluster_centers_.reshape(self.K, h, w)).to(activations.device)
        self.rep_maps = rep_maps
        self.base_score = base_score

        # Score differences
        diffs = torch.zeros(self.K, device=activations.device)
        with torch.no_grad():
            for k in range(self.K):
                mask = rep_maps[k:k+1].unsqueeze(0)
                out = self.model_arch(input * mask)
                diffs[k] = out[0, class_idx] - base_score

        # Zero-out smallest clusters
        num_zero = int(self.zero_ratio * self.K)
        if num_zero > 0:
            lowest = torch.argsort(diffs)[:num_zero]
            diffs[lowest] = float('-inf')

        # Softmax weights
        T = self.temperature_dict.get(class_idx, self.temperature)
        weights = F.softmax(diffs / T, dim=0)

        # Combine saliency
        sal = torch.zeros(1,1,h,w, device=activations.device)
        for k in range(self.K):
            sal += weights[k] * rep_maps[k:k+1].unsqueeze(0)
        sal = F.relu(sal)
        mn, mx = sal.min(), sal.max()
        if mn == mx:
            return None
        sal = (sal - mn) / (mx - mn)
        self.last_saliency = sal
        return sal

    def __call__(self, input_tensor, targets=None, class_idx=None, retain_graph=False):
        if targets and isinstance(targets[0], ClassifierOutputTarget):
            class_idx = targets[0].category
        return self.forward(input_tensor, class_idx, retain_graph)


def test_cluster_cam_single(
    model: torch.nn.Module,
    model_dict: dict,
    image_folder: str,
    k_list: list[int],
    output_excel: str,
    top_n: int | None = None,
    random_sample: bool = False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    model_dict["arch"] = model

    # 1) Load image paths
    paths = list_image_paths(image_folder)
    if top_n is not None and len(paths) > top_n:
        paths = random.sample(paths, top_n) if random_sample else paths[:top_n]
    print(f"Total images to process: {len(paths)}")

    # 2) Preprocessing transform
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    records = []
    # 3) Loop over each K
    for K in k_list:
        print(f"\n→ Bắt đầu chạy với K = {K}")
        cam = ClusterScoreCAM(model_dict, num_clusters=K)
        shifts = []

        for img_path in tqdm(paths, desc=f"K={K}"):
            # load & preprocess
            img = load_image(img_path)
            x = tf(img).unsqueeze(0).to(device)  # (1,C,H,W)

            # predict class
            with torch.no_grad():
                logits = model(x)
                c = int(logits.argmax(1))

            # compute saliency and masked output
            sal = cam(x, class_idx=c)            # (1,1,H,W)
            masked = x * sal
            with torch.no_grad():
                lm = model(masked)

            # shift = lm_score - original_score
            shift = (lm[0, c] - logits[0, c]).item()
            shifts.append(shift)

        mean, std = float(np.mean(shifts)), float(np.std(shifts))
        print(f"→ K={K:2d}: mean_shift={mean:.4f}, std_shift={std:.4f}")
        records.append({"K": K, "mean_shift": mean, "std_shift": std})

    # 4) Save to Excel
    out_dir = os.path.dirname(output_excel)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"Created directory: {out_dir}")
        
    pd.DataFrame(records).to_excel(output_excel, index=False)
    print(f"\nSaved results to {output_excel}")


if __name__ == "__main__":
    # --- Ví dụ dùng với ResNet-18 ---
    # weights = ResNet18_Weights.DEFAULT
    # r18 = resnet18(weights=weights)
    # r18_dict = {"arch": r18, "target_layer": r18.layer4[-1]}
    
    # test_cluster_cam_single(
    #     model=r18,
    #     model_dict=r18_dict,
    #     image_folder="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened", 
    #     k_list=[2, 4, 6, 8, 10, 15,20,25,30,35,40,45,50,55,60,65,70,80,90,100,110,120,130],
    #     output_excel="/home/infres/ltvo/ClusCAM/results/validation/Resnet18_zr-0.5_t-0.5_1000-imgs.xlsx",
    #     top_n=1000,
    #     random_sample=True
    # )
    
     # --- Chạy với ResNet-34 ---
    # weights = ResNet34_Weights.DEFAULT
    # r34 = resnet34(weights=weights)
    # # chọn target_layer là block cuối cùng của layer4
    # r34_dict = {
    #     "arch": r34,
    #     "target_layer": r34.layer4[-1]
    # }

    # test_cluster_cam_single(
    #     model=r34,
    #     model_dict=r34_dict,
    #     image_folder="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened",
    #     k_list=[2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 110, 120, 130],
    #     output_excel="/home/infres/ltvo/ClusCAM/results/validation/Resnet34_zr-0.5_t-0.5_1000-imgs.xlsx",
    #     top_n=1000,
    #     random_sample=True
    # )
    
    
    #  # --- ResNet-50 ---
    # weights = ResNet50_Weights.IMAGENET1K_V2
    # r50 = resnet50(weights=weights)
    # r50_dict = {"arch": r50, "target_layer": r50.layer4[-1]}
    # test_cluster_cam_single(
    #     model=r50,
    #     model_dict=r50_dict,
    #     image_folder="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened",
    #     k_list=[2, 5, 6, 8, 10, 15 , 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
    #     output_excel="./results/ResNet50_z0.5_t0.5_1k.xlsx",
    #     top_n=1000,
    #     random_sample=True
    # )

    # # --- ResNet-101 ---
    # weights = ResNet101_Weights.IMAGENET1K_V2
    # r101 = resnet101(weights=weights)
    # r101_dict = {"arch": r101, "target_layer": r101.layer4[-1]}
    # test_cluster_cam_single(
    #     model=r101,
    #     model_dict=r101_dict,
    #     image_folder="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened",
    #     k_list=[2, 5, 6, 8, 10, 15 , 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
    #     output_excel="./results/ResNet101_z0.5_t0.5_1k.xlsx",
    #     top_n=1000,
    #     random_sample=True
    # )

    # # --- ResNet-152 ---
    # weights = ResNet152_Weights.IMAGENET1K_V2
    # r152 = resnet152(weights=weights)
    # r152_dict = {"arch": r152, "target_layer": r152.layer4[-1]}
    # test_cluster_cam_single(
    #     model=r152,
    #     model_dict=r152_dict,
    #     image_folder="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened",
    #     k_list=[2, 5, 6, 8, 10, 15 , 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
    #     output_excel="./results/ResNet152_z0.5_t0.5_1k.xlsx",
    #     top_n=1000,
    #     random_sample=True
    # )
    
    #    # --- Chạy với EfficientNet-B0 ---
    # weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    # eff = efficientnet_b0(weights=weights)
    # # Chọn lớp conv cuối cùng trước pooling: đây thường là features[-1][2]
    # for m in eff.modules():
    #     if isinstance(m, nn.SiLU):
    #         m.inplace = False
    # eff_dict = {
    #     "arch": eff,
    #     "target_layer": eff.features[-1][2]  # conv2d trong block cuối
    # }

    # test_cluster_cam_single(
    #     model=eff,
    #     model_dict=eff_dict,
    #     image_folder="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened",
    #     k_list=[2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 110, 120, 130],
    #     output_excel="/home/infres/ltvo/ClusCAM/results/validation/EfficientNetB0_zr-0.5_t-0.5_1000-imgs.xlsx",
    #     top_n=1000,
    #     random_sample=True
    # )
    
   # The above code snippet is using the Inception V3 model to perform cluster-based Class Activation
   # Mapping (CAM) on images from the ImageNet dataset. Here is a breakdown of the code:
    # weights = Inception_V3_Weights.DEFAULT
    # inc = inception_v3(weights=weights)  # tắt aux để forward chỉ trả về logits chính
    # inc_dict = {
    #     "arch": inc,
    #     # “target_layer” có thể chọn lớp cuối cùng trước pooling, ví dụ Mixed_7c
    #     "target_layer": inc.Mixed_7c
    # }
    


    # test_cluster_cam_single(
    #     model=inc,
    #     model_dict=inc_dict,
    #     image_folder="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened", 
    #     k_list=[2, 5, 6, 8, 10, 15 , 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
    #     output_excel="/home/infres/ltvo/ClusCAM/results/validation/Inceptionet_V3_zr-0.5_t-0.5_1000-imgs.xlsx",
    #     top_n=1000,
    #     random_sample=True
    # )
    
    #===============Validation with ViT=================#
    
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    vit = vit_b_16(weights=weights)
    vit.eval()
    # chọn target layer: activations sau conv_proj hoặc patch_embed
    try:
        vit_target = vit.conv_proj
    except AttributeError:
        vit_target = vit.patch_embed.proj
    vit_dict = {"arch": vit, "target_layer": vit_target}
    test_cluster_cam_single(
        model=vit,
        model_dict=vit_dict,
        image_folder="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened",
        k_list=[2, 5, 6, 8, 10, 15 , 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
        output_excel="./results/ViT_B16_z0.5_t0.5_1k.xlsx",
        top_n=1000,
        random_sample=True
    )
    
    #============Validation with Swin===============#
    
     
    weights = Swin_B_Weights.IMAGENET1K_V1
    swin = swin_b(weights=weights)
    swin.eval()
    # chọn target layer: activations sau conv_proj hoặc patch_embed
    swin_target = swin.features[0][0]
    swin_dict = {"arch": swin, "target_layer":  swin_target}
    test_cluster_cam_single(
        model=swin,
        model_dict=swin_dict,
        image_folder="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened",
        k_list=[2, 5, 6, 8, 10, 15 , 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        output_excel="./results/swin_b_z0.5_t0.5_1k.xlsx",
        top_n=1000,
        random_sample=True
    )
    
    
