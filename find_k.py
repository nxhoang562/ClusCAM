import os
import random
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, inception_v3, Inception_V3_Weights
from torchvision.models.inception import InceptionOutputs
from utils_folder import load_image, list_image_paths
from cam.basecam import BaseCAM
from sklearn.cluster import KMeans
import torch.nn.functional as F
from tqdm import tqdm  # progress bar
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
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
        # Input: (1,C,H,W)
        b, c, h, w = input.size()
        
        outputs = self.model_arch(input)
        if isinstance(outputs, InceptionOutputs):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            logits = outputs
        
        # 1) Forward pass + chọn class
        # logits = self.model_arch(input)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        # nếu class_idx là tensor, chuyển về int
        elif isinstance(class_idx, torch.Tensor):
            class_idx = int(class_idx)
        base_score = logits[0, class_idx]

        # 2) Backprop lấy activation maps (low-res)
        self.model_arch.zero_grad()
        base_score.backward(retain_graph=retain_graph)
        activations = self.activations['value'][0]  # (nc, u, v)
        nc, u, v = activations.shape

        # 3) Upsample & normalize mỗi activation map lên input size
        up_maps = []
        for i in range(nc):
            a = activations[i:i+1].unsqueeze(0)  # (1,1,u,v)
            a_up = F.interpolate(
                a, size=(h, w), mode='bilinear', align_corners=False
            )[0, 0]
            if a_up.max() != a_up.min():
                a_up = (a_up - a_up.min()) / (a_up.max() - a_up.min())
            up_maps.append(a_up)
        up_maps = torch.stack(up_maps, dim=0)  # (nc, h, w)

        # 4) Flatten upsampled maps và clustering
        flat_maps = up_maps.reshape(nc, -1).detach().cpu().numpy()  # (nc, h*w)
        kmeans = KMeans(n_clusters=self.K, init='k-means++', random_state=0)
        print(f"[ClusterScoreCAM] Running KMeans++ with {self.K} clusters...")
        kmeans.fit(flat_maps)
        rep_maps = torch.from_numpy(
            kmeans.cluster_centers_.reshape(self.K, h, w)
        ).to(activations.device)
        
        #them de visualize 
        self.rep_maps = rep_maps  # tensor (K, h, w)
        self.base_score = base_score  # cũng lưu nếu cần debug

        # 5) Tính score difference mỗi mask
        diffs = torch.zeros(self.K, device=activations.device)
        with torch.no_grad():
            for k in range(self.K):
                mask = rep_maps[k:k+1].unsqueeze(0)  # (1,1,h,w)
                out = self.model_arch(input * mask)
                diffs[k] = out[0, class_idx] - base_score

        # 6) Zero-out cluster nhỏ nhất
        num_zero = int(self.zero_ratio * self.K)
        if num_zero > 0:
            lowest = torch.argsort(diffs)[:num_zero]
            diffs[lowest] = float('-inf')

        # 7) Softmax với nhiệt độ class
        T = self.temperature_dict.get(class_idx, self.temperature)
        weights = F.softmax(diffs / T, dim=0)

        # 8) Kết hợp saliency map
        sal = torch.zeros(1,1,h,w, device=activations.device)
        for k in range(self.K):
            sal += weights[k] * rep_maps[k:k+1].unsqueeze(0)

        # 9) Post-process + normalize
        sal = F.relu(sal)
        mn, mx = sal.min(), sal.max()
        if mn == mx:
            return None
        sal = (sal - mn) / (mx - mn)
        
        self.last_saliency = sal  # tensor (1,1,h,w)

        return sal
# khong chay voi batch
    # def __call__(self, input, class_idx=None, retain_graph=False):
    #     return self.forward(input, class_idx, retain_graph)
# chay voi batch 
    def __call__(self,
                input_tensor: torch.Tensor,
                targets: list[ClassifierOutputTarget] | None = None,
                class_idx: int | None = None,
                retain_graph: bool = False):
        
        if targets is not None and len(targets) > 0 and isinstance(targets[0], ClassifierOutputTarget):
            class_idx = targets[0].category

        # if class_idx is None:
        #     raise ValueError("ClusterCAM: need a class_idx")
        
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
    
    weights = Inception_V3_Weights.DEFAULT
    inc = inception_v3(weights=weights)  # tắt aux để forward chỉ trả về logits chính
    inc_dict = {
        "arch": inc,
        # “target_layer” có thể chọn lớp cuối cùng trước pooling, ví dụ Mixed_7c
        "target_layer": inc.Mixed_7c
    }

    test_cluster_cam_single(
        model=inc,
        model_dict=inc_dict,
        image_folder="/home/infres/ltvo/ClusCAM/datasets/imagenet/val_flattened", 
        k_list=[2, 4, 6, 8, 10, 15,20,25,30,35,40,45,50,55,60,65,70,80,90,100,110,120,130],
        output_excel="/home/infres/ltvo/ClusCAM/results/validation/InceptionetV3_zr-0.5_t-0.5_1000-imgs.xlsx",
        top_n=1000,
        random_sample=True
    )
