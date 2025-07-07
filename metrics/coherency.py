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
        cam_out = attribution_method(input_tensor=mixed, targets=targets)

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
        coherency = (rho + 1.0) / 2.0 * 100.0

        return coherency.mean() if return_mean else coherency



# def batch_pearson_coherency(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
#     """
#     Computes Pearson correlation for a batch of matrices.

#     Args:
#         A (torch.Tensor): Tensor of shape (batch_size, m, n).
#         B (torch.Tensor): Tensor of shape (batch_size, m, n).

#     Returns:
#         torch.Tensor: PCC for each matrix pair in the batch.
#     """
#     # Reshape to (batch_size, m*n)
#     a = A.view(A.shape[0], -1)
#     b = B.view(B.shape[0], -1)

#     # Compute mean-centered matrices
#     a_centered = a - a.mean(dim=1, keepdim=True)
#     b_centered = b - b.mean(dim=1, keepdim=True)

#     # Compute covariance and stds
#     cov = (a_centered * b_centered).sum(dim=1) / (a.shape[1] - 1)
#     std_a = torch.sqrt((a_centered**2).sum(dim=1) / (a.shape[1] - 1))
#     std_b = torch.sqrt((b_centered**2).sum(dim=1) / (b.shape[1] - 1))

#     # Avoid division by zero
#     eps = 1e-8
#     rho = cov / (std_a * std_b + eps)
#     return rho



# class Coherency(MetricBase):
#     def __init__(self):
#         super().__init__("coherency")

#     def __call__(
#         self,
#         model: nn.Module,
#         test_images: torch.Tensor,       # (B, C, H, W)
#         saliency_maps: torch.Tensor,     # (B, 1, H, W) hoặc (B, H, W)
#         class_idx: int | torch.Tensor,
#         attribution_method: AttributionMethod,
#         upsample_method,
#         return_mean: bool = True,
#         device: str = "cpu",
#         layer: nn.Module = None,
#         **kwargs,
#     ) -> torch.Tensor:
#         # 1) Mix ảnh với CAM gốc
#         mixed = mix_image_with_saliency(test_images, saliency_maps)

#         # 2) Tính attribution trên ảnh đã mix
#         mixed_attr = attribution_method.attribute(
#             input_tensor=mixed,
#             model=model,
#             layer=layer,
#             target=class_idx,
#         )
#         mixed_attr = upsample_method(
#             attribution=mixed_attr,
#             image=mixed,
#             device=device,
#             model=model,
#             layer=layer,
#         )

#         # 3) Squeeze channel nếu cần
#         if mixed_attr.ndim == 4:
#             mixed_attr = mixed_attr.squeeze(1)   # (B, H, W)
#         if saliency_maps.ndim == 4:
#             saliency_maps = saliency_maps.squeeze(1)

#         # 4) Tính Pearson và normalize về [0,1]
#         rho = batch_pearson_coherency(mixed_attr, saliency_maps)  # (B,)
#         coherency = (rho + 1.0) / 2.0
#         coherency_pct = coherency * 100

#         return coherency_pct.mean() if return_mean else coherency_pct






# class Coherency(BaseMetric):
#     def __init__(self):
#         super().__init__("coherency")
#         pass

#     def __call__(
#         self,
#         model: nn.Module,
#         test_images: torch.Tensor,
#         saliency_maps: torch.Tensor,
#         class_idx: int | torch.Tensor,
#         attribution_method: AttributionMethod,
#         previous_attributions: list[torch.Tensor],
#         return_mean: bool = True,
#         layer: nn.Module = None,
#         upsample_method=None,
#         device: str = "cpu",
#         mixer: Mixer = None,
#         **kwargs,
#     ) -> torch.Tensor:
#         # Coherency is defined as as the pearson correlation between the attribution on the image and the attribution on the image * saliency map
#         mixed_images = mix_image_with_saliency(test_images, saliency_maps)

#         mixed_attributions = attribution_method.attribute(
#             input_tensor=mixed_images,
#             model=model,
#             layer=layer,
#             target=class_idx,
#         )

#         mixed_attributions = upsample_method(
#             attribution=mixed_attributions,
#             image=mixed_images,
#             device=device,
#             model=model,
#             layer=layer,
#         )

#         # Filter using the previous attributions, but firts make a copy of the previous attributions
#         previous_attributions_copy = []

#         for i in range(len(previous_attributions)):
#             previous_attributions_copy.append(previous_attributions[i].detach().clone())

#         previous_attributions_copy.append(mixed_attributions)
#         mixed_attributions = mixer(previous_attributions_copy)

#         if mixed_attributions.shape != saliency_maps.shape:
#             raise ValueError(
#                 f"Mixed attributions shape {mixed_attributions.shape} does not match saliency maps shape {saliency_maps.shape}."
#             )

#         if len(mixed_attributions.shape) == 4:
#             mixed_attributions = mixed_attributions.squeeze(1)
#             saliency_maps = saliency_maps.squeeze(1)

#         # Compute the correlation between mixed_attributions and saliency_maps
#         pearson = (batch_pearson_coherency(mixed_attributions, saliency_maps) + 1) / 2

#         if return_mean:
#             return pearson.mean()

#         return pearson.item()