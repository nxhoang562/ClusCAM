import torch
import torch.nn.functional as F
from cam.basecam import BaseCAM
from sklearn.cluster import KMeans


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
        num_clusters: int = 10,
        zero_ratio: float = 0.5,
        temperature_dict: dict | None = None,
        temperature: float = 0.5
    ):
        super().__init__(model_dict)
        self.K = num_clusters
        self.zero_ratio = zero_ratio
        self.temperature_dict = temperature_dict or {}
        self.temperature = temperature

    def forward(self, input, class_idx=None, retain_graph=False):
        # Input: (1,C,H,W)
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

        # 4) Flatten upsampled maps và clustering
        flat_maps = up_maps.reshape(nc, -1).detach().cpu().numpy()  # (nc, h*w)
        kmeans = KMeans(n_clusters=self.K, init='k-means++', random_state=0)
        kmeans.fit(flat_maps)
        # cluster centers shape (K, h*w) -> (K, h, w)
        rep_maps = torch.from_numpy(
            kmeans.cluster_centers_.reshape(self.K, h, w)
        ).to(activations.device)

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

        # 7) Softmax class
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
        return sal

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
