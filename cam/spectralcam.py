import torch
import torch.nn.functional as F
import numpy as np
from cam.basecam import BaseCAM
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class SpectralCAM(BaseCAM):
    """
    Score-CAM with Spectral Clustering, customizable zero-out ratio and temperature per class.

    Args:
        model_dict: dict as in BaseCAM
        num_clusters: number of clusters K
        zero_ratio: fraction of smallest clusters to zero out (0-1)
        temperature_dict: dict mapping class_idx to temperature
        temperature: default temperature
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
        # Input: (1, C, H, W)
        b, c, h, w = input.size()

        # 1) Forward pass + select class
        logits = self.model_arch(input)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        elif isinstance(class_idx, torch.Tensor):
            class_idx = int(class_idx)
        base_score = logits[0, class_idx]

        # 2) Backprop to get low-res activations
        self.model_arch.zero_grad()
        base_score.backward(retain_graph=retain_graph)
        activations = self.activations['value'][0]  # (nc, u, v)
        nc, u, v = activations.shape

        # 3) Upsample & normalize each activation map to input size
        up_maps = []
        for i in range(nc):
            a = activations[i:i+1].unsqueeze(0)  # (1,1,u,v)
            a_up = F.interpolate(a, size=(h, w), mode='bilinear', align_corners=False)[0, 0]
            if a_up.max() != a_up.min():
                a_up = (a_up - a_up.min()) / (a_up.max() - a_up.min())
            up_maps.append(a_up)
        up_maps = torch.stack(up_maps, dim=0)  # (nc, h, w)

        # --- Data cleaning for clustering ---
        # 4a) Flatten & convert to CPU numpy array
        flat = up_maps.reshape(nc, -1).detach().cpu().numpy().astype(np.float32)
        # 4b) Replace Inf/NaN
        flat[~np.isfinite(flat)] = 0.0
        # 4c) Standardize features to zero mean, unit variance
        scaler = StandardScaler()
        flat_norm = scaler.fit_transform(flat)

        # 5) Spectral Clustering on normalized data
        spectral = SpectralClustering(
            n_clusters=self.K,
            affinity='nearest_neighbors',
            n_neighbors=min(nc - 1, max(10, self.K * 2)),
            assign_labels='kmeans',
            random_state=0
        )
        print(f"[ClusterScoreCAM] Running SpectralClustering with {self.K} clusters...")
        labels = spectral.fit_predict(flat_norm)

        # 6) Build representative maps by averaging members of each cluster
        rep_list = []
        for k in range(self.K):
            idxs = np.where(labels == k)[0]
            if len(idxs) == 0:
                rep_list.append(np.zeros((h, w), dtype=np.float32))
            else:
                cluster_maps = flat[idxs].reshape(-1, h, w)
                rep_list.append(cluster_maps.mean(axis=0))
        rep_maps = torch.from_numpy(np.stack(rep_list)).to(activations.device)  # (K, h, w)
        self.rep_maps = rep_maps
        self.base_score = base_score

        # 7) Compute score differences for each cluster mask
        diffs = torch.zeros(self.K, device=activations.device)
        with torch.no_grad():
            for k in range(self.K):
                mask = rep_maps[k:k+1].unsqueeze(0)  # (1,1,h,w)
                out = self.model_arch(input * mask)
                diffs[k] = out[0, class_idx] - base_score

        # 8) Zero-out smallest clusters
        num_zero = int(self.zero_ratio * self.K)
        if num_zero > 0:
            lowest = torch.argsort(diffs)[:num_zero]
            diffs[lowest] = float('-inf')

        # 9) Softmax with temperature per class
        T = self.temperature_dict.get(class_idx, self.temperature)
        weights = F.softmax(diffs / T, dim=0)

        # 10) Combine saliency map
        sal = torch.zeros(1, 1, h, w, device=activations.device)
        for k in range(self.K):
            sal += weights[k] * rep_maps[k:k+1].unsqueeze(0)

        # 11) Post-process + normalize
        sal = F.relu(sal)
        mn, mx = sal.min(), sal.max()
        if mn == mx:
            return None
        sal = (sal - mn) / (mx - mn)
        self.last_saliency = sal
        return sal

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: list[ClassifierOutputTarget] | None = None,
        class_idx: int | None = None,
        retain_graph: bool = False
    ):
        if targets is not None and len(targets) > 0 and isinstance(targets[0], ClassifierOutputTarget):
            class_idx = targets[0].category
        return self.forward(input_tensor, class_idx, retain_graph)
