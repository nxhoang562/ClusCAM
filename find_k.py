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
from tqdm import tqdm
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn

class ClusterScoreCAM(BaseCAM):
    def __init__(
        self,
        model_dict,
        num_clusters=10,
        zero_ratio=0,
        temperature_dict=None,
        temperature=1
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

        # 1) chọn class & base_score
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        elif isinstance(class_idx, torch.Tensor):
            class_idx = int(class_idx)
        base_score = logits[0, class_idx]

        # 2) backprop lấy activation maps
        self.model_arch.zero_grad()
        base_score.backward(retain_graph=retain_graph)
        activations = self.activations['value'][0]  # (nc, u, v)
        nc, u, v = activations.shape

        # 3) upsample + normalize
        up_maps = []
        for i in range(nc):
            a = activations[i:i+1].unsqueeze(0)
            a_up = F.interpolate(a, size=(h, w), mode='bilinear', align_corners=False)[0,0]
            if a_up.max() != a_up.min():
                a_up = (a_up - a_up.min())/(a_up.max() - a_up.min())
            up_maps.append(a_up)
        up_maps = torch.stack(up_maps, dim=0)  # (nc, h, w)

        # 4) clustering KMeans
        flat = up_maps.reshape(nc, -1).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.K, init='k-means++', random_state=0)
        print(f"[ClusterScoreCAM] Running KMeans++ with {self.K} clusters...")
        kmeans.fit(flat)
        rep_maps = torch.from_numpy(
            kmeans.cluster_centers_.reshape(self.K, h, w)
        ).to(activations.device)

        self.rep_maps  = rep_maps
        self.base_score = base_score

        # 5) tính diffs cho mỗi cluster mask
        diffs = torch.zeros(self.K, device=activations.device)
        with torch.no_grad():
            for k in range(self.K):
                mask = rep_maps[k:k+1].unsqueeze(0)  # (1,1,h,w)
                out  = self.model_arch(input * mask)
                diffs[k] = out[0, class_idx] - base_score

        # lưu raw_diffs (shape=(K,)) để xuất sau
        self.raw_diffs = diffs.clone()

        # 6) zero-out cụm yếu nhất
        num_zero = int(self.zero_ratio * self.K)
        if num_zero > 0:
            lowest = torch.argsort(diffs)[:num_zero]
            diffs[lowest] = float('-inf')

        # 7) softmax với nhiệt độ
        T = self.temperature_dict.get(class_idx, self.temperature)
        weights = F.softmax(diffs / T, dim=0)

        # 8) kết hợp saliency
        sal = torch.zeros(1,1,h,w, device=activations.device)
        for k in range(self.K):
            sal += weights[k] * rep_maps[k:k+1].unsqueeze(0)

        # 9) normalize saliency
        sal = F.relu(sal)
        mn, mx = sal.min(), sal.max()
        if mn == mx:
            return None
        sal = (sal - mn)/(mx - mn)

        self.last_saliency = sal  # (1,1,h,w)
        return sal

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: list[ClassifierOutputTarget] | None = None,
        class_idx: int | None = None,
        retain_graph: bool = False
    ):
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

    paths = list_image_paths(image_folder)
    if top_n and len(paths) > top_n:
        paths = random.sample(paths, top_n) if random_sample else paths[:top_n]
    V = len(paths)
    print(f"Total images to process: {V}")

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # records sẽ được cập nhật dần
    records = []
    out_dir = os.path.dirname(output_excel)
    os.makedirs(out_dir, exist_ok=True)

    for K in k_list:
        print(f"\n→ Bắt đầu chạy với K = {K}")
        cam = ClusterScoreCAM(model_dict, num_clusters=K)
        cluster_shifts = []
        mask_shifts    = []
        raw_diffs_all  = []

        for img_path in tqdm(paths, desc=f"K={K}"):
            img = load_image(img_path)
            x   = tf(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits     = model(x)
                c          = int(logits.argmax(1))
                orig_score = logits[0, c].item()

            _ = cam(x, class_idx=c)
            raw = cam.raw_diffs.cpu().numpy()
            raw_diffs_all.append(raw)

            cluster_shifts.append(float(raw.sum() / K))

            sal     = cam.last_saliency
            masked  = x * sal
            with torch.no_grad():
                lm = model(masked)
            mask_shifts.append((lm[0, c] - orig_score).item())

        # lưu ma trận raw_diffs ngay khi xong K này
        raw_matrix = np.stack(raw_diffs_all, axis=1)  # shape (K, V)
        npy_name   = f"raw_diffs_K{K}_V{V}.npy"
        np.save(os.path.join(out_dir, npy_name), raw_matrix)
        print(f"Saved raw diffs matrix to {npy_name}")

        # tính và thêm record cho K này
        rec = {
            "K": K,
            "mean_cluster_shift": float(np.mean(cluster_shifts)),
            "std_cluster_shift":  float(np.std(cluster_shifts)),
            "mean_mask_shift":    float(np.mean(mask_shifts)),
            "std_mask_shift":     float(np.std(mask_shifts))
        }
        records.append(rec)

        # *** LƯU EXCEL NGAY SAU K NÀY ***
        pd.DataFrame(records).to_excel(output_excel, index=False)
        print(f"Updated summary Excel with K={K}")
        
    print(f"\nFinished all K – final summary saved to {output_excel}")


if __name__ == "__main__":
    
    image_folder="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened"
    top_n = 500
    k_list=[10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    
    
    # --- Ví dụ chạy với EfficientNet-B0 ---
    model = "EfficientNetB0"
    output_excel= f"/home/infres/ltvo/ClusCAM/results/tinhlai_validation/{model}/{model}_{top_n}imgs.xlsx"
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    eff = efficientnet_b0(weights=weights)
    for m in eff.modules():
        if isinstance(m, nn.SiLU):
            m.inplace = False
    eff_dict = {
        "arch": eff,
        "target_layer": eff.features[-1][2]
    }
    
    test_cluster_cam_single(
        model=eff,
        model_dict=eff_dict,
        image_folder= image_folder,
        k_list=k_list,
        output_excel= output_excel,
        top_n=top_n,
        random_sample=False
        )
    
    ##=========Resnet18============
    
    # model = "Resnet18"
    # output_excel= f"/home/infres/ltvo/ClusCAM/results/tinhlai_validation/{model}/{model}_{top_n}imgs.xlsx"
    # weights = ResNet18_Weights.DEFAULT
    # r18 = resnet18(weights=weights)
    # r18_dict = {"arch": r18, "target_layer": r18.layer4[-1]}
    
    # test_cluster_cam_single(
    #     model=r18,
    #     model_dict=r18_dict,
    #     image_folder= image_folder, 
    #     k_list=k_list,
    #     output_excel= output_excel,
    #     top_n=top_n,
    #     random_sample=True
    # )
    
    # ##======Resnet34========##
    # model = "Resnet34"
    # output_excel= f"/home/infres/ltvo/ClusCAM/results/tinhlai_validation/{model}/{model}_{top_n}imgs.xlsx"
    # weights =ResNet34_Weights.DEFAULT
    # r34 = resnet34(weights=weights)
    # r34_dict = {
    #     "arch": r34,
    #     "target_layer": r34.layer4[-1]
    # }
    
    # test_cluster_cam_single(
    #     model=r34,
    #     model_dict=r34_dict,
    #     image_folder= image_folder, 
    #     k_list=k_list,
    #     output_excel= output_excel,
    #     top_n=top_n,
    #     random_sample=True
    # )
    
    # --- ResNet-50 ---
    # model = "Resnet50"
    # output_excel= f"/home/infres/ltvo/ClusCAM/results/tinhlai_validation/{model}/{model}_{top_n}imgs.xlsx"
    # weights = ResNet50_Weights.IMAGENET1K_V2
    # r50 = resnet50(weights=weights)
    # r50_dict = {"arch": r50, "target_layer": r50.layer4[-1]}
    
    # test_cluster_cam_single(
    #     model=r50,
    #     model_dict=r50_dict,
    #     image_folder= image_folder, 
    #     k_list=k_list,
    #     output_excel= output_excel,
    #     top_n=top_n,
    #     random_sample=True
    # )
    
    #     # --- ResNet-101 ---
    # model = "ResNet101"
    # output_excel= f"/home/infres/ltvo/ClusCAM/results/tinhlai_validation/{model}/{model}_{top_n}imgs.xlsx"
    # weights = ResNet101_Weights.IMAGENET1K_V2
    # r101 = resnet101(weights=weights)
    # r101_dict = {"arch": r101, "target_layer": r101.layer4[-1]}
    
    # test_cluster_cam_single(
    #     model=r101,
    #     model_dict=r101_dict,
    #     image_folder= image_folder, 
    #     k_list=k_list,
    #     output_excel= output_excel,
    #     top_n=top_n,
    #     random_sample=True
    # )
    
    #  ----- Inception_V3 -------
    
    # model = " Inception_V3"
    # output_excel= f"/home/infres/ltvo/ClusCAM/results/tinhlai_validation/{model}/{model}_{top_n}imgs.xlsx"
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
    #     image_folder= image_folder, 
    #     k_list=k_list,
    #     output_excel= output_excel,
    #     top_n=top_n,
    #     random_sample=True
    # )
    
    
    #===============Validation with ViT=================#
    
    # model = "ViT"
    # output_excel= f"/home/infres/ltvo/ClusCAM/results/tinhlai_validation/{model}/{model}_{top_n}imgs.xlsx"
    # weights = ViT_B_16_Weights.IMAGENET1K_V1
    # vit = vit_b_16(weights=weights)
    # vit.eval()
    # # chọn target layer: activations sau conv_proj hoặc patch_embed
    # try:
    #     vit_target = vit.conv_proj
    # except AttributeError:
    #     vit_target = vit.patch_embed.proj
    # vit_dict = {"arch": vit, "target_layer": vit_target}
    
    # test_cluster_cam_single(
    #     model=vit,
    #     model_dict=vit_dict,
    #     image_folder= image_folder, 
    #     k_list=k_list,
    #     output_excel= output_excel,
    #     top_n=top_n,
    #     random_sample=True
    # )
    
    #============Validation with Swin===============#
    # model = "Swin"
    # output_excel= f"/home/infres/ltvo/ClusCAM/results/tinhlai_validation/{model}/{model}_{top_n}imgs.xlsx"
    # weights = Swin_B_Weights.IMAGENET1K_V1
    # swin = swin_b(weights=weights)
    # swin.eval()
    # # chọn target layer: activations sau conv_proj hoặc patch_embed
    # swin_target = swin.features[0][0]
    # swin_dict = {"arch": swin, "target_layer":  swin_target}
    
    # test_cluster_cam_single(
    #     model=swin,
    #     model_dict=swin_dict,
    #     image_folder= image_folder, 
    #     k_list=k_list,
    #     output_excel= output_excel,
    #     top_n=top_n,
    #     random_sample=True
    # )
    
