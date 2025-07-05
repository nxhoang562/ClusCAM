import torch
import torch.nn.functional as F
from cam.basecam import BaseCAM
from sklearn.cluster import KMeans


class ClusterScoreCAM(BaseCAM):
    """
    Score-CAM với clustering, tối ưu batch-level để tận dụng GPU parallelism.

    Args:
        model_dict: dict giống BaseCAM
        num_clusters: số cụm K
        zero_ratio: tỉ lệ phần trăm cluster nhỏ nhất bị loại (0-1)
        temperature_dict: dict class_idx -> temperature
        temperature: nhiệt độ mặc định
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

    def forward(self, input: torch.Tensor, class_idx=None, retain_graph=False):
        return self.__call__(input, class_idx, retain_graph)

    def __call__(self, input: torch.Tensor, class_idx=None, retain_graph=False):
        """
        Hỗ trợ batch (B,C,H,W) và single (1,C,H,W).
        """
        if input.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {input.dim()}D")

        # Forward logits and get activations via hook
        logits = self.model_arch(input)  # (local_B, num_classes)
        activations = self.activations['value']  # (local_B, nc, u, v)
        local_B, nc, u, v = activations.shape
        # Track batch size per device
        print(f"[ClusterScoreCAM] Processing local batch size: {local_B}")

        # Determine class indices per sample
        device = input.device
        if class_idx is None:
            cls = logits.argmax(dim=1)
        elif isinstance(class_idx, torch.Tensor):
            cls = class_idx.to(device)
        elif isinstance(class_idx, (list, tuple)):
            cls = torch.tensor(class_idx, device=device)
        else:
            cls = torch.full((local_B,), int(class_idx), device=device, dtype=torch.long)
        base_scores = logits.gather(1, cls.unsqueeze(1)).squeeze(1)  # (local_B,)

        # Single-sample shortcut
        if local_B == 1:
            return self._single_forward(input, class_idx, retain_graph)

        # 2) Backprop gradients w.r.t activations
        self.model_arch.zero_grad()
        grads = torch.autograd.grad(
            outputs=base_scores,
            inputs=activations,
            grad_outputs=torch.ones_like(base_scores, device=device),
            retain_graph=retain_graph
        )[0]  # (local_B, nc, u, v)

        # 3) Upsample + normalize
        H, W = input.shape[2], input.shape[3]
        up = F.interpolate(
            grads.view(-1, 1, u, v), size=(H, W), mode='bilinear', align_corners=False
        ).view(local_B, nc, H, W)
        min_ = up.flatten(2).amin(-1).view(local_B, nc, 1, 1)
        max_ = up.flatten(2).amax(-1).view(local_B, nc, 1, 1)
        up = (up - min_) / (max_ - min_ + 1e-7)

        # 4) Cluster per-sample
        rep_maps = torch.zeros((local_B, self.K, H, W), device=device)
        for i in range(local_B):
            flat = up[i].reshape(nc, -1).detach().cpu().numpy()
            km = KMeans(n_clusters=self.K, random_state=0).fit(flat)
            centers = torch.from_numpy(km.cluster_centers_).to(device)
            rep_maps[i] = centers.view(self.K, H, W)

        # 5) Batch scoring in one forward
        masks = rep_maps.view(local_B * self.K, 1, H, W)
        inps = input.unsqueeze(1).expand(local_B, self.K, input.shape[1], H, W)
        inps = inps.contiguous().view(local_B * self.K, input.shape[1], H, W)
        outs = self.model_arch(inps * masks)  # (local_B*K, num_classes)
        cls_rep = cls.unsqueeze(1).expand(local_B, self.K).contiguous().view(-1)
        scores = outs.gather(1, cls_rep.unsqueeze(1)).squeeze(1)  # (local_B*K,)

        # 6) Compute diffs and zero-out
        diffs = (scores - base_scores.repeat_interleave(self.K)).view(local_B, self.K)
        num_zero = int(self.zero_ratio * self.K)
        if num_zero > 0:
            for i in range(local_B):
                low = torch.argsort(diffs[i])[:num_zero]
                diffs[i, low] = float('-inf')

        # 7) Softmax weights per sample
        T = torch.tensor([
            self.temperature_dict.get(int(cls[i].item()), self.temperature)
            for i in range(local_B)
        ], device=device)
        weights = F.softmax(diffs / T.unsqueeze(1), dim=1)  # (local_B, K)

        # 8) Combine saliency
        sal = (weights.view(local_B, self.K, 1, 1) * rep_maps).sum(dim=1, keepdim=True)
        sal = F.relu(sal)
        mn = sal.flatten(2).amin(-1).view(local_B, 1, 1, 1)
        mx = sal.flatten(2).amax(-1).view(local_B, 1, 1, 1)
        sal = (sal - mn) / (mx - mn + 1e-7)

        return sal

    def _single_forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        logits = self.model_arch(input)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        elif isinstance(class_idx, torch.Tensor):
            class_idx = int(class_idx)
        base_score = logits[0, class_idx]

        self.model_arch.zero_grad()
        base_score.backward(retain_graph=retain_graph)
        activations = self.activations['value'][0]
        nc, u, v = activations.shape

        up = F.interpolate(activations.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
        min_ = up.flatten(1).amin(1).view(nc, 1, 1)
        max_ = up.flatten(1).amax(1).view(nc, 1, 1)
        up = (up - min_) / (max_ - min_ + 1e-7)

        flat = up.reshape(nc, -1).detach().cpu().numpy()
        km = KMeans(n_clusters=self.K, random_state=0).fit(flat)
        rep_maps = torch.from_numpy(km.cluster_centers_).to(input.device).view(self.K, h, w)

        masks = rep_maps.unsqueeze(1)
        inps = input.repeat(self.K, 1, 1, 1)
        outs = self.model_arch(inps * masks)
        scores = outs[:, class_idx]
        diffs = scores - base_score

        num_zero = int(self.zero_ratio * self.K)
        if num_zero > 0:
            low = torch.argsort(diffs)[:num_zero]
            diffs[low] = float('-inf')

        T = self.temperature_dict.get(class_idx, self.temperature)
        weights = F.softmax(diffs / T, dim=0)

        sal = (weights.view(self.K, 1, 1) * rep_maps).sum(0, keepdim=True)
        sal = F.relu(sal)
        mn, mx = sal.min(), sal.max()
        if mx > mn:
            sal = (sal - mn) / (mx - mn)
        return sal.unsqueeze(0)
