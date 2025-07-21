import torch
import torch.nn.functional as F
import numpy as np
from cam.basecam import BaseCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class RandomCam(BaseCAM):
    """
    Score-CAM với random clustering, tuỳ chỉnh zero-out và temperature đối với một class nhất định.

    Args:
        model_dict: dict giống BaseCAM
        num_clusters: số cụm K
        zero_ratio: tỉ lệ phần trăm cluster nhỏ nhất bị loại (0-1)
        temperature_dict: dict class_idx -> temperature
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
        # Input: (1,C,H,W)
        b, c, h, w = input.size()

        # 1) Forward pass + chọn class
        logits = self.model_arch(input)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
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

        # 4) Random clustering: gán ngẫu nhiên nhãn rồi lấy mean
        #    Nếu cluster rỗng, chọn 1 map ngẫu nhiên làm đại diện
        rng = np.random.default_rng(seed=0)
        # tạo nhãn ngẫu nhiên từ 0..K-1 cho mỗi feature-map
        labels = torch.from_numpy(
            rng.integers(0, self.K, size=nc)
        ).to(up_maps.device)
        rep_maps = []
        for k in range(self.K):
            sel = up_maps[labels == k]  # mọi map có label==k
            if sel.numel() == 0:
                # nếu không có map nào, chọn ngẫu nhiên 1 map
                idx_rand = int(rng.integers(0, nc))
                rep_maps.append(up_maps[idx_rand:idx_rand+1])
            else:
                # trung bình tất cả map trong cụm
                rep_maps.append(sel.mean(dim=0, keepdim=True))
        rep_maps = torch.cat(rep_maps, dim=0)  # (K, h, w)

        # Lưu lại để visualize/debug
        self.rep_maps = rep_maps
        self.base_score = base_score

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

        # 7) Softmax với nhiệt độ tương ứng class
        T = self.temperature_dict.get(class_idx, self.temperature)
        weights = F.softmax(diffs / T, dim=0)

        # 8) Kết hợp saliency map
        sal = torch.zeros(1, 1, h, w, device=activations.device)
        for k in range(self.K):
            sal += weights[k] * rep_maps[k:k+1].unsqueeze(0)

        # 9) Post-process + normalize
        sal = F.relu(sal)
        mn, mx = sal.min(), sal.max()
        if mn == mx:
            return None
        sal = (sal - mn) / (mx - mn)

        self.last_saliency = sal  # (1,1,h,w)
        return sal

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: list[ClassifierOutputTarget] | None = None,
        class_idx: int | None = None,
        retain_graph: bool = False
    ):
        if targets is not None and len(targets) > 0 \
           and isinstance(targets[0], ClassifierOutputTarget):
            class_idx = targets[0].category
        return self.forward(input_tensor, class_idx, retain_graph)
