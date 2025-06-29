import torch
import torch.nn.functional as F
from cam.basecam import BaseCAM
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

class ClusterScoreCAM(BaseCAM):
    """
    Score-CAM với clustering, tùy chỉnh zero-out và temperature đối với một class nhất định.
    Bổ sung bước PCA để giảm chiều dựa trên explained variance trước khi clustering.

    Args:
        model_dict: dict giống BaseCAM
        num_clusters: số cụm K
        zero_ratio: tỉ lệ phần trăm cluster nhỏ nhất bị loại (0-1)
        temperature_dict: dict class_idx -> temperature
        temperature: nhiệt độ mặc định
        pca_ratio: float trong (0,1) để giữ tỉ lệ phương sai qua PCA; None để không dùng PCA
    """
    def __init__(
        self,
        model_dict,
        num_clusters=10,
        zero_ratio=0.5,
        temperature_dict=None,
        temperature=1.0,
        pca_ratio=0.95 
    ):
        super().__init__(model_dict)
        self.K = num_clusters
        self.zero_ratio = zero_ratio
        self.temperature_dict = temperature_dict or {}
        self.temperature = temperature
        self.pca_ratio = pca_ratio

    def forward(self, input, class_idx=None, retain_graph=False):
        # Input: (1, C, H, W)
        b, c, h, w = input.size()

        # 1) Forward pass + chọn class
        logits = self.model_arch(input)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
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

        # 4) Flatten lên (nc, h*w) và (tuỳ chọn) giảm chiều bằng PCA
        flat_maps = up_maps.reshape(nc, -1).detach().cpu().numpy()  # (nc, h*w)
        use_pca = isinstance(self.pca_ratio, float) and 0 < self.pca_ratio < 1
        if use_pca:
            pca = PCA(n_components=self.pca_ratio, random_state=0)
            reduced = pca.fit_transform(flat_maps)  # (nc, n_components)
        else:
            reduced = flat_maps  # không PCA

        # 5) KMeans trên không gian (giảm) chiều
        kmeans = KMeans(n_clusters=self.K, random_state=0)
        kmeans.fit(reduced)

        # 6) Lấy cluster centers, nếu có dùng PCA thì inverse về không gian gốc
        if use_pca:
            centers = pca.inverse_transform(kmeans.cluster_centers_)  # (K, h*w)
        else:
            centers = kmeans.cluster_centers_  # (K, h*w)

        # reshape về (K, h, w) và chuyển về device
        rep_maps = torch.from_numpy(centers.reshape(self.K, h, w)).to(activations.device)

        # 7) Tính score difference mỗi mask
        diffs = torch.zeros(self.K, device=activations.device)
        with torch.no_grad():
            for k in range(self.K):
                mask = rep_maps[k:k+1].unsqueeze(0)  # (1,1,h,w)
                out = self.model_arch(input * mask)
                diffs[k] = out[0, class_idx] - base_score

        # 8) Zero-out cluster nhỏ nhất
        num_zero = int(self.zero_ratio * self.K)
        if num_zero > 0:
            lowest = torch.argsort(diffs)[:num_zero]
            diffs[lowest] = float('-inf')

        # 9) Softmax với nhiệt độ class
        T = self.temperature_dict.get(class_idx, self.temperature)
        weights = F.softmax(diffs / T, dim=0)

        # 10) Kết hợp saliency map
        sal = torch.zeros(1, 1, h, w, device=activations.device)
        for k in range(self.K):
            sal += weights[k] * rep_maps[k:k+1].unsqueeze(0)

        # 11) Post-process + normalize
        sal = F.relu(sal)
        mn, mx = sal.min(), sal.max()
        if mn == mx:
            return None
        sal = (sal - mn) / (mx - mn)
        return sal

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
