import torch
import torch.nn as nn
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .metric_utils import mix_image_with_saliency, MetricBase


def batch_pearson_coherency(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes Pearson correlation for a batch of matrices.

    Args:
        A (torch.Tensor): Tensor of shape (batch_size, m, n).
        B (torch.Tensor): Tensor of shape (batch_size, m, n).

    Returns:
        torch.Tensor: PCC for each matrix pair in the batch.
    """
    # Reshape to (batch_size, m*n) using reshape to handle non-contiguous tensors
    a = A.reshape(A.shape[0], -1)
    b = B.reshape(B.shape[0], -1)

    # Compute mean-centered matrices
    a_centered = a - a.mean(dim=1, keepdim=True)
    b_centered = b - b.mean(dim=1, keepdim=True)

    # Compute covariance and stds
    cov = (a_centered * b_centered).sum(dim=1) / (a.shape[1] - 1)
    std_a = torch.sqrt((a_centered**2).sum(dim=1) / (a.shape[1] - 1))
    std_b = torch.sqrt((b_centered**2).sum(dim=1) / (a.shape[1] - 1))

    # Avoid division by zero
    eps = 1e-8
    rho = cov / (std_a * std_b + eps)
    return rho


class Coherency(MetricBase):
    def __init__(self):
        super().__init__("coherency")

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,       # (B, C, H, W)
        saliency_maps: torch.Tensor,     # (B, 1, H, W) or (B, C, H, W)
        class_idx: int | torch.Tensor,
        attribution_method,
        upsample_method,
        return_mean: bool = True,
        device: str = "cpu",
        layer: nn.Module = None,
        **kwargs,
    ) -> torch.Tensor:
        # 1) Mix original images with saliency maps
        mixed = mix_image_with_saliency(test_images, saliency_maps)

        # 2) Compute attribution on mixed images via CAM call
        idx = int(class_idx) if not isinstance(class_idx, (list, tuple, torch.Tensor)) else int(class_idx[0])
        targets = [ClassifierOutputTarget(idx)]
        cam_out = attribution_method(mixed, targets)

        # Convert numpy array to torch.Tensor if needed
        if isinstance(cam_out, np.ndarray):
            attr = torch.from_numpy(cam_out)
        else:
            attr = cam_out
        attr = attr.to(device)

        # Ensure shape (B,1,H,W)
        if attr.ndim == 3:
            attr = attr.unsqueeze(1)

        # 3) Upsample attribution to image size
        mixed_attr = upsample_method(
            attribution=attr,
            image=mixed,
            device=device,
            model=model,
            layer=layer
        )

        # 4) Squeeze channel from attribution
        if mixed_attr.ndim == 4:
            mixed_attr = mixed_attr.squeeze(1)  # (B, H, W)

        # 5) Collapse saliency map channels if multi-channel
        if saliency_maps.ndim == 4:
            # average across channel dimension
            sal_map = saliency_maps.mean(dim=1)  # (B, H, W)
        else:
            sal_map = saliency_maps  # already (B, H, W)

        # 6) Compute Pearson correlation then normalize to [0,100]
        rho = batch_pearson_coherency(mixed_attr, sal_map)  # (B,)
        coherency = (rho + 1.0) / 2.0 

        return coherency.mean().item() if return_mean else coherency.item()

