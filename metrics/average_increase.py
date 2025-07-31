import torch
import torch.nn as nn
from .metric_utils import MetricBase, mix_image_with_saliency
import numpy as np

class AverageIncrease(MetricBase):
    def __init__(self):
        super().__init__("AverageIncrease")

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ) -> float:
        """
        Compute Average Increase: tỉ lệ (%) số ảnh mà confidence tăng lên
        khi chỉ cung cấp saliency map. 
        - return_mean=True: trả về mean(%) trên batch (float)
        - return_mean=False: trả về tensor(%) từng sample
        """
        # 1. Move data to device
        test_images = test_images.to(device)
        if isinstance(saliency_maps, np.ndarray):
            saliency_maps = torch.from_numpy(saliency_maps)
        saliency_maps = saliency_maps.to(device)

        # 2. Mix saliency vào ảnh
        saliency_images = mix_image_with_saliency(test_images, saliency_maps)

        # 3. Forward nguyên ảnh và ảnh saliency
        test_preds = model(test_images)          # (N, num_classes)
        saliency_preds = model(saliency_images)  # (N, num_classes)
        if apply_softmax:
            test_preds = nn.functional.softmax(test_preds, dim=1)
            saliency_preds = nn.functional.softmax(saliency_preds, dim=1)

        # 4. Chọn score của class quan tâm
        if isinstance(class_idx, int):
            test_scores = test_preds[:, class_idx]            # (N,)
            saliency_scores = saliency_preds[:, class_idx]    # (N,)
        elif isinstance(class_idx, torch.Tensor):
            idx = torch.arange(test_preds.size(0), device=device)
            test_scores = test_preds[idx, class_idx]
            saliency_scores = saliency_preds[idx, class_idx]
        else:
            raise ValueError("class_idx must be int or torch.Tensor")

        # 5. Tạo mask tăng (1 nếu saliency_score > test_score)
        inc_mask = (saliency_scores > test_scores).float()    # (N,)

        # 6. Chuyển sang phần trăm
        percentages = inc_mask * 100.0                        # (N,)

        if return_mean:
            return percentages.mean().item()                  # float
        else:
            return percentages.item()                                # tensor (N,)

class AverageGain(MetricBase):
    def __init__(self, eps: float = 1e-8):
        """
        eps: độ ổn định khi chia cho p_i (tránh chia cho 0)
        """
        super().__init__("AverageGain")
        self.eps = eps

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ) -> float | torch.Tensor:
        """
        Compute Average Gain:
            AG = (1/N) * sum_i max(0, p̃_i - p_i) / (p_i + eps)

        Args:
            model: nn.Module đã load weights
            test_images: (N, C, H, W) nguyên ảnh
            saliency_maps: (N, 1, H, W) hoặc (N, C, H, W)
            class_idx: int hoặc Tensor([i1, i2, ...]) chỉ class quan tâm
            device: thiết bị tính toán
            apply_softmax: nếu True thì đưa logits về xác suất
            return_mean: nếu True trả về float trung bình, else tensor (N,)
        """
        # 1) chuyển sang device
        test_images = test_images.to(device)
        if isinstance(saliency_maps, np.ndarray):
            saliency_maps = torch.from_numpy(saliency_maps)
        saliency_maps = saliency_maps.to(device)

        # 2) tạo ảnh mix saliency
        saliency_images = mix_image_with_saliency(test_images, saliency_maps)

        # 3) forward
        logits_orig = model(test_images)
        logits_sal = model(saliency_images)
        if apply_softmax:
            probs_orig = nn.functional.softmax(logits_orig, dim=1)
            probs_sal = nn.functional.softmax(logits_sal, dim=1)
        else:
            probs_orig = logits_orig
            probs_sal = logits_sal

        # 4) lấy score class quan tâm
        if isinstance(class_idx, int):
            p_i = probs_orig[:, class_idx]      # (N,)
            p_tilde = probs_sal[:, class_idx]   # (N,)
        elif isinstance(class_idx, torch.Tensor):
            idx = torch.arange(probs_orig.size(0), device=device)
            p_i = probs_orig[idx, class_idx]
            p_tilde = probs_sal[idx, class_idx]
        else:
            raise ValueError("class_idx must be int or torch.Tensor")

        # 5) tính gain từng sample, chỉ giữ phần dương
        gain = torch.clamp(p_tilde - p_i, min=0.0) / (p_i + self.eps)  # (N,)

        if return_mean:
            return gain.mean().item()
        else:
            return gain





