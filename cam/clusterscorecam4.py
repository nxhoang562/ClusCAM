import torch
import torch.nn.functional as F
from cam.basecam import BaseCAM
from sklearn.cluster import KMeans


class ClusterScoreCAM2(BaseCAM):
    """
    Score-CAM with clustering: sử dụng đại diện cụm (centroid) để mask input.
    Clustering is done on the low-res activation maps before upsampling.
    
    Args:
        model_dict: dict giống BaseCAM
        num_clusters: số lượng cụm K
    """
    def __init__(self, model_dict, num_clusters=10):
        super().__init__(model_dict)
        self.K = num_clusters

    def forward(self, input, class_idx=None, retain_graph=False):
        # Input shape: (1, C, H, W)
        b, c, h, w = input.size()

        # 1) Forward pass + select target class
        logits = self.model_arch(input)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        base_score = logits[0, class_idx]  

        # 2) Backprop to get activation maps
        self.model_arch.zero_grad()
        base_score.backward(retain_graph=retain_graph)
        activations = self.activations['value'][0]  # (nc, u, v)
        nc, u, v = activations.shape

        # 3) Flatten low-res maps and perform KMeans clustering
        flat_maps = activations.reshape(nc, -1).detach().cpu().numpy()  # (nc, u*v)
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(flat_maps)
        # Centroids shape: (K, u*v) -> reshape to (K, u, v)
        rep_low = torch.from_numpy(
            kmeans.cluster_centers_.reshape(self.K, u, v)
        ).to(activations.device)

        # 4) Upsample each representative to input size + normalize
        rep_maps = []
        for km in range(self.K):
            m = rep_low[km:km+1].unsqueeze(0)  # (1,1,u,v)
            m_up = F.interpolate(
                m, size=(h, w), mode='bilinear', align_corners=False
            )[0, 0]
            # Normalize to [0,1]
            if m_up.max() != m_up.min():
                m_up = (m_up - m_up.min()) / (m_up.max() - m_up.min())
            rep_maps.append(m_up)
        rep_maps = torch.stack(rep_maps, dim=0)  # (K, h, w)

        # 5) Compute score difference for each mask
        diffs = torch.zeros(self.K, device=activations.device)
        with torch.no_grad():
            for km in range(self.K):
                mask = rep_maps[km].unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
                out = self.model_arch(input * mask)
                diffs[km] = out[0, class_idx] - base_score

        # 6) Zero-out the lowest half to avoid dilution
        num_zero = self.K // 2
        if num_zero > 0:
            lowest_idx = torch.argsort(diffs)[:num_zero]
            diffs[lowest_idx] = float('-inf')

        # 7) Compute weights via softmax with temperature
        T = 0.5
        weights = F.softmax(diffs / T, dim=0)

        # 8) Reconstruct saliency map
        saliency_map = torch.zeros(1, 1, h, w, device=activations.device)
        for km in range(self.K):
            saliency_map += weights[km] * rep_maps[km].unsqueeze(0).unsqueeze(0)

        # 9) Post-process and normalize
        saliency_map = F.relu(saliency_map)
        mn, mx = saliency_map.min(), saliency_map.max()
        if mn == mx:
            return None
        saliency_map = (saliency_map - mn) / (mx - mn)
        return saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
