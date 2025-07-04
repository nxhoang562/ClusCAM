import os
import random
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from utils import load_image, list_image_paths
from cam.basecam import BaseCAM
from tqdm import tqdm  # progress bar

class ClusterScoreCAM(BaseCAM):
    def __init__(
        self,
        model_dict,
        num_clusters: int = 10,
        zero_ratio: float = 0.5,
        temperature_dict: dict | None = None,
        temperature: float = 1.0
    ):
        super().__init__(model_dict)
        self.K = num_clusters
        self.zero_ratio = zero_ratio
        self.temperature_dict = temperature_dict or {}
        self.temperature = temperature

    def forward(self, input: torch.Tensor, class_idx: int | None = None, retain_graph: bool = False):
        b, c, h, w = input.size()
        # 1) Forward để lấy logits gốc
        logits = self.model_arch(input)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        elif isinstance(class_idx, torch.Tensor):
            class_idx = int(class_idx)
        base_score = logits[0, class_idx]

        # 2) Backprop để lấy activation maps
        self.model_arch.zero_grad()
        base_score.backward(retain_graph=retain_graph)
        activations = self.activations['value'][0]  # shape (nc, u, v)
        nc, u, v = activations.shape

        # 3) Upsample & normalize mỗi activation map
        up_maps = []
        for i in range(nc):
            a = activations[i:i+1].unsqueeze(0)  # (1,1,u,v)
            a_up = F.interpolate(a, size=(h, w), mode='bilinear', align_corners=False)[0,0]
            if a_up.max() != a_up.min():
                a_up = (a_up - a_up.min()) / (a_up.max() - a_up.min())
            up_maps.append(a_up)
        up_maps = torch.stack(up_maps, dim=0)  # (nc, h, w)

        # 4) Flatten & K-means
        flat = up_maps.flatten(1).detach().cpu().numpy()  # (nc, h*w)
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(flat)
        reps = torch.from_numpy(
            kmeans.cluster_centers_.reshape(self.K, h, w)
        ).to(up_maps.device)

        # 5) Tính score difference cho mỗi cluster prototype
        diffs = torch.zeros(self.K, device=up_maps.device)
        with torch.no_grad():
            # gom batch K lên GPU 1 lần
            masks = reps.unsqueeze(1)  # (K,1,h,w)
            inps = input.repeat(self.K, 1, 1, 1) * masks  # (K,C,h,w)
            outs = self.model_arch(inps)  # (K, num_classes)
            base = base_score.unsqueeze(0) if base_score.ndim else base_score
            diffs = outs[:, class_idx] - base

        # 6) Zero-out các cluster có diff nhỏ nhất
        num_zero = int(self.zero_ratio * self.K)
        if num_zero > 0:
            lowest = torch.argsort(diffs)[:num_zero]
            diffs[lowest] = float("-inf")

        # 7) Softmax với nhiệt độ
        T = self.temperature_dict.get(class_idx, self.temperature)
        weights = F.softmax(diffs / T, dim=0)

        # 8) Kết hợp để tạo saliency map
        sal = torch.zeros(1, 1, h, w, device=up_maps.device)
        for k in range(self.K):
            sal += weights[k] * reps[k:k+1].unsqueeze(0)
        sal = F.relu(sal)
        mn, mx = sal.min(), sal.max()
        if mn == mx:
            return torch.zeros_like(sal)
        return (sal - mn) / (mx - mn)

    __call__ = forward


def test_cluster_cam(
    model: torch.nn.Module,
    model_dict: dict,
    image_folder: str,
    k_list: list[int],
    output_excel: str,
    top_n: int | None = None,       # số ảnh muốn test (None → tất cả)
    random_sample: bool = False,    # True → lấy ngẫu nhiên top_n
    batch_size: int = 8,
    num_workers: int = 4,
    use_tqdm: bool = True           # thêm tùy chọn hiển thị tqdm
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    model_dict["arch"] = model

    # 1) Load list ảnh và cắt theo top_n
    paths = list_image_paths(image_folder)
    if top_n is not None and len(paths) > top_n:
        paths = random.sample(paths, top_n) if random_sample else paths[:top_n]
    print(f"Total images to process: {len(paths)}")

    # 2) Dataset & DataLoader
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    class ImgDS(Dataset):
        def __init__(self, ps, tf):
            self.ps = ps
            self.tf = tf
        def __len__(self):
            return len(self.ps)
        def __getitem__(self, idx):
            img = load_image(self.ps[idx])
            return self.tf(img), self.ps[idx]

    dl = DataLoader(
        ImgDS(paths, tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 3) Chạy thử cho mỗi K
    records = []
    for K in k_list:
        print(f"\n→ Bắt đầu chạy với K = {K}")
        cam = ClusterScoreCAM(model_dict, num_clusters=K)
        shifts = []

        iterator = dl
        if use_tqdm:
            iterator = tqdm(dl, desc=f"K={K}", unit="batch")

        for batch_idx, (imgs, _) in enumerate(iterator):
            imgs = imgs.to(device)
            with torch.no_grad():
                logits = model(imgs)
                preds = logits.argmax(1)

            for i in range(imgs.size(0)):
                x = imgs[i:i+1]
                c = int(preds[i])
                sal = cam(x, class_idx=c)           # (1,1,H,W)
                masked = x * sal
                with torch.no_grad():
                    lm = model(masked)
                shift = (-logits[i, c] + lm[0, c]).item()
                shifts.append(shift)

            if not use_tqdm and (batch_idx + 1) % 10 == 0:
                print(f"   Đã xử lý {batch_idx+1}/{len(dl)} batches")

        mean, std = float(np.mean(shifts)), float(np.std(shifts))
        print(f"→ K={K:2d}: mean_shift={mean:.4f}, std_shift={std:.4f}")
        records.append({"K": K, "mean_shift": mean, "std_shift": std})

    # 4) Lưu kết quả ra Excel
    pd.DataFrame(records).to_excel(output_excel, index=False)
    print(f"\nSaved results to {output_excel}")


if __name__ == "__main__":

    # --- Test với ResNet-18, random 100 ảnh ---
    weights = ResNet18_Weights.DEFAULT
    r18 = resnet18(weights=weights)
    r18_dict = {"arch": r18, "target_layer": r18.layer4[-1]}
    test_cluster_cam(
        model=r18,
        model_dict=r18_dict,
        image_folder="/home/infres/xnguyen-24/cluster_cam/datasets/imagenet",
        k_list=[2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        output_excel="k_results_resnet18.xlsx",
        top_n=100,
        random_sample=True,
        batch_size=5,
        num_workers=4,
        use_tqdm=True
    )
