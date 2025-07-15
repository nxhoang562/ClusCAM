import torch
import torch.nn as nn
from .metric_utils import MetricBase, mix_image_with_saliency
import numpy as np

class AverageConfidence(MetricBase):
    def __init__(self):
        super().__init__("average_confidence_increase")

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        device: str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ) -> float:
        # 1. Chuyển device
        test_images = test_images.to(device)
        if isinstance(saliency_maps, np.ndarray):
            saliency_maps = torch.from_numpy(saliency_maps)
        saliency_maps = saliency_maps.to(device)

        # 2. Trộn saliency vào ảnh
        saliency_images = mix_image_with_saliency(test_images, saliency_maps)

        # 3. Forward
        pred_orig = model(test_images)         # (N, num_classes)
        pred_sal  = model(saliency_images)     # (N, num_classes)
        if apply_softmax:
            pred_orig = nn.functional.softmax(pred_orig, dim=1)
            pred_sal  = nn.functional.softmax(pred_sal, dim=1)

        # 4. Chọn score lớp
        if isinstance(class_idx, int):
            score_orig = pred_orig[:, class_idx]
            score_sal  = pred_sal[:, class_idx]
        elif isinstance(class_idx, torch.Tensor):
            idx = torch.arange(score_orig.size(0), device=device)
            score_orig = pred_orig[idx, class_idx]
            score_sal  = pred_sal[idx, class_idx]
        else:
            raise ValueError("class_idx must be int or torch.Tensor")

        # 5. Tính delta và mask chỉ lấy tăng
        delta = score_sal - score_orig           # (N,)
        inc_mask = delta > 0                     # boolean mask
        if inc_mask.sum() == 0:
            # Không có mẫu tăng
            return 0.0 if return_mean else torch.zeros(0)

        # 6. Tính phần trăm tăng cho mỗi mẫu tăng
        percents = (delta[inc_mask] / score_orig[inc_mask]) * 100

        # 7. Trả về
        if return_mean:
            return percents.mean().item()
        else:
            return percents  # tensor các % tăng
