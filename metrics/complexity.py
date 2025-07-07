import torch
import torch.nn as nn
from .metric_utils import mix_image_with_saliency, MetricBase, AttributionMethod


class Complexity( MetricBase):
    def __init__(self):
        super().__init__("complexity")
        pass

    def __call__(
        self,
        saliency_maps: torch.Tensor,
        return_mean: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        saliency_maps_clone = saliency_maps.detach().clone()
        if len(saliency_maps.shape) == 4:
            # If the saliency maps are 4D, we need to reduce them to 3D
            # by removing the channel dimension
            saliency_maps_clone = saliency_maps_clone.squeeze(1)

        # res = torch.linalg.norm(saliency_maps, ord=1, dim=(-2, -1))
        # Iterate over the batch dimension
        res = torch.zeros(
            saliency_maps_clone.shape[0], device=saliency_maps_clone.device
        )
        
        # H, W = saliency_maps_clone.shape[-2], saliency_maps_clone.shape[-1]
        
        for i in range(saliency_maps_clone.shape[0]):
            # print(f"Processing saliency map {i+1}/{saliency_maps_clone}")
            res[i] = torch.mean(saliency_maps_clone[i]) 
        if return_mean:
            res = res.mean()
        return res.item()