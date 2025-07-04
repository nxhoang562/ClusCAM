import os
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms, models
from cam.basecam import BaseCAM
from sklearn.cluster import KMeans
import pandas as pd

# ----- ClusterScoreCAMWithDiffs -----
class ClusterScoreCAMWithDiffs(BaseCAM):
    def __init__(self, model_dict, num_clusters=10, zero_ratio=0.5,
                 temperature_dict=None, temperature=1.0):
        super().__init__(model_dict)
        self.K = num_clusters
        self.zero_ratio = zero_ratio
        self.temperature_dict = temperature_dict or {}
        self.temperature = temperature

    def forward(self, input, class_idx=None, retain_graph=False, return_diffs=False):
        b, c, h, w = input.size()

        # 1) Forward + chọn class
        logits = self.model_arch(input)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        elif isinstance(class_idx, torch.Tensor):
            class_idx = int(class_idx)
        base_score = logits[0, class_idx]

        # 2) Backprop lấy activations
        self.model_arch.zero_grad()
        base_score.backward(retain_graph=retain_graph)
        activations = self.activations['value'][0]
        nc, u, v = activations.shape

        # 3) Upsample & normalize
        up_maps = []
        for i in range(nc):
            a = activations[i:i+1].unsqueeze(0)
            a_up = F.interpolate(a, size=(h, w), mode='bilinear', align_corners=False)[0,0]
            if a_up.max() != a_up.min():
                a_up = (a_up - a_up.min())/(a_up.max() - a_up.min())
            up_maps.append(a_up)
        up_maps = torch.stack(up_maps, dim=0)

        # 4) Clustering
        flat = up_maps.reshape(nc, -1).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(flat)
        rep = torch.from_numpy(
            kmeans.cluster_centers_.reshape(self.K, h, w)
        ).to(activations.device)

        # 5) Tính diffs
        diffs = torch.zeros(self.K, device=activations.device)
        with torch.no_grad():
            for k in range(self.K):
                mask = rep[k:k+1].unsqueeze(0)
                out = self.model_arch(input * mask)
                diffs[k] = out[0, class_idx] - base_score

        # 6) Zero-out cluster nhỏ nhất
        num_zero = int(self.zero_ratio * self.K)
        if num_zero > 0:
            lowest = torch.argsort(diffs)[:num_zero]
            diffs[lowest] = float('-inf')
        diffs_zeroed = diffs.clone()

        # 7) Kết hợp saliency
        T = self.temperature_dict.get(class_idx, self.temperature)
        weights = F.softmax(diffs / T, dim=0)
        sal = torch.zeros(1,1,h,w, device=activations.device)
        for k in range(self.K):
            sal += weights[k] * rep[k:k+1].unsqueeze(0)
        sal = F.relu(sal)
        mn, mx = sal.min(), sal.max()
        sal = None if mn == mx else (sal - mn)/(mx - mn)

        if return_diffs:
            return sal, diffs_zeroed
        return sal


# ----- Hàm benchmark (theo K rồi qua ảnh) -----
def benchmark_folder_per_k(
    model_dict,
    folder_path,
    num_images,
    k_list,
    zero_ratio=0.5,
    temperature_dict=None,
    temperature=1.0,
    class_idx=None,
    output_excel="results_per_k.xlsx",
    device=torch.device("cpu")
):
    # Chuẩn bị transform
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Lọc file hình ảnh theo đuôi phổ biến
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    all_files = sorted(os.listdir(folder_path))
    image_files = [f for f in all_files if f.lower().endswith(valid_exts)]
    if num_images is not None:
        image_files = image_files[:num_images]

    results = []
    for k in k_list:
        sum_diffs_list = []
        print(f"\n=== K = {k} ===")
        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except UnidentifiedImageError:
                print(f"⚠️ Bỏ qua không nhận dạng được file: {img_name}")
                continue

            tensor = transform(img).unsqueeze(0).to(device)
            cam = ClusterScoreCAMWithDiffs(
                model_dict,
                num_clusters=k,
                zero_ratio=zero_ratio,
                temperature_dict=temperature_dict,
                temperature=temperature
            )
            _, diffs = cam.forward(tensor, class_idx=class_idx, return_diffs=True)
            valid = torch.isfinite(diffs)
            sum_diffs = diffs[valid].sum().item()
            sum_diffs_list.append(sum_diffs)
            print(f"  • {img_name}: sum_diffs = {sum_diffs:.4f}")

        # Tính trung bình sum_diffs cho mỗi K
        avg_over_images = sum(sum_diffs_list) / len(sum_diffs_list) if sum_diffs_list else float('nan')
        print(f"→ K = {k}, trung bình sum_diffs trên {len(sum_diffs_list)} ảnh = {avg_over_images:.4f}")
        results.append({
            "K": k,
            "avg_sum_diffs": avg_over_images
        })

    # Lưu kết quả vào Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\nĐã lưu kết quả trung bình vào {output_excel}")


if __name__ == "__main__":
    # Chọn device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model ResNet18 và đưa lên device
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.to(device)
    model.eval()

    # Thiết lập model_dict cho BaseCAM
    model_dict = {
        "arch": model,
        "target_layer": model.layer4[-1]
    }

    # Tham số
    IMAGE_FOLDER = "/home/infres/xnguyen-24/cluster_cam/datasets/imagenet"
    NUM_IMAGES = 100
    K_VALUES = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    # Chạy benchmark
    benchmark_folder_per_k(
        model_dict,
        IMAGE_FOLDER,
        NUM_IMAGES,
        K_VALUES,
        zero_ratio=0.5,
        temperature_dict=None,
        temperature=0.5,
        class_idx=None,
        output_excel="kcluster_avg_results.xlsx",
        device=device
    )
