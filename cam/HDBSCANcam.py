import torch
import torch.nn.functional as F
from cam.basecam import BaseCAM
import hdbscan
import numpy as np

class HDBSCANcam(BaseCAM):
    """
    Score-CAM with clustering: sử dụng đại diện cụm (centroid) từ HDBSCAN.

    Args:
        model_dict: dict giống BaseCAM
        min_cluster_size: kích thước tối thiểu của cụm trong HDBSCAN
        min_samples: số mẫu tối thiểu để xem như core point (mặc định None)
        cluster_selection_epsilon: ngưỡng epsilon cho cluster selection
    """
    def __init__(self, model_dict, min_cluster_size=2, min_samples=None, cluster_selection_epsilon=0.05 ):
        super().__init__(model_dict)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        # 1) forward gốc + chọn class
        logits = self.model_arch(input)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[0, class_idx]

        # backward để lấy activations
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']  # (1, c, u, v)
        _, nc, u, v = activations.shape

        # 2) upsample + normalize từng map
        maps = []
        for i in range(nc):
            m = activations[0, i:i+1]  # (1, u, v)
            m = F.interpolate(
                m.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            )[0, 0]
            if m.max() == m.min():
                maps.append(torch.zeros_like(m))
            else:
                maps.append((m - m.min()) / (m.max() - m.min()))

        # stack và flatten
        all_maps = torch.stack(maps, dim=0).view(nc, -1).detach().cpu().numpy()

        # 3) clustering với HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon
        )
        labels = clusterer.fit_predict(all_maps)
        # print("labels:", labels)
        unique_labels = sorted([lbl for lbl in set(labels) if lbl >= 0])  # bỏ noise (-1)
        n_clusters = len(unique_labels)
        # print("n_clusters:", n_clusters)
        if n_clusters == 0:
            # fallback: không có cụm nào, sử dụng từng map làm cụm riêng
            rep_maps = torch.from_numpy(
                all_maps.reshape(nc, h, w)
            ).to(activations.device)
            n_clusters = nc
        else:
            # tính centroid mỗi cụm
            centroids = []
            for lbl in unique_labels:
                members = all_maps[labels == lbl]
                centroids.append(members.mean(axis=0))
            rep_maps = torch.from_numpy(
                np.stack(centroids).reshape(n_clusters, h, w)
            ).to(activations.device)

        # 4) tính diff score cho từng centroid
        diffs = torch.zeros(n_clusters, device=activations.device)
        with torch.no_grad():
            for idx in range(n_clusters):
                mask = rep_maps[idx].unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
                if mask.max() > mask.min():
                    mask = (mask - mask.min()) / (mask.max() - mask.min())

                out_mask = self.model_arch(input * mask)  # (1, num_classes)
                raw_score = out_mask[0, class_idx]
                diffs[idx] = raw_score - score

        # loại bớt một số cụm có diffs thấp nhất
        zero_percent = 0.5
        num_zero = int(n_clusters * zero_percent)
        if num_zero > 0:
            _, sorted_idx = torch.sort(diffs, descending=False)
            lowest_idx = sorted_idx[:num_zero]
            diffs[lowest_idx] = float('-inf')

        # softmax để chuẩn hóa
        cluster_scores = F.softmax(diffs, dim=0)

        # 5) reconstruct saliency_map
        saliency_map = torch.zeros(1, 1, h, w, device=activations.device)
        for i in range(n_clusters):
            saliency_map += cluster_scores[i] * rep_maps[i].unsqueeze(0).unsqueeze(0)

        # 6) post-process
        saliency_map = F.relu(saliency_map)
        mn, mx = saliency_map.min(), saliency_map.max()
        if mn == mx:
            return None
        saliency_map = (saliency_map - mn) / (mx - mn)

        return saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
