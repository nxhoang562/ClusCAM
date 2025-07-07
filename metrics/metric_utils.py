import torch 
import torch.nn as nn 
import numpy as np
import torch
from typing import List, Literal

class MetricBase: 
    def __init__(self, name: str):
        self.name = name 
    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        device: str = "cpu",
        apply_softmax: bool = True, 
        return_mean: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement __call__()") #Exception

def mix_image_with_saliency(
    image: torch.Tensor,
    saliency_map: torch.Tensor,
) -> torch.Tensor:
    """
    Mix the original image with the saliency map to create a new image: 
    Parameters: 
    - image (torch.Tensor): input image, shape (B, C, H, W)
    - saliency_map (torch.Tensor): saliency map, shape (B, C, H, W), element value is in (0,1)
    """  
    # if saliency_map.max() != 1 or saliency_map.min() != 0:
    #     print(f"Saliency map should have be normalized between 0 and 1. Current max value = {saliency_map.max()}, min value = {saliency_map.min()}")
    #     raise ValueError
    new_image = image * saliency_map
    return new_image 

#=========================================================================================================================================================# 
class AttributionMethod:
    def __init__(self):
        pass

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: str | nn.Module,
        target: torch.Tensor,
        baseline_dist: torch.Tensor = None,
    ):
        raise NotImplementedError()


class Mixer:
    def __init__(self, layers_to_combine: Literal["all", "top", "above"] = "all"):
        self.layers_to_combine = layers_to_combine

    def filter_layers(self, attributions: List[torch.Tensor]):
        # attributions = deepcopy(attributions)
        copy_attributions = []
        for attr in attributions:
            if attr is not None:
                copy_attributions.append(attr.clone())
        if len(copy_attributions) == 1:
            return copy_attributions

        if self.layers_to_combine == "all":
            return copy_attributions
        elif self.layers_to_combine == "top":
            # Most coarse + most fine-grained
            return [copy_attributions[0], copy_attributions[-1]]
        elif self.layers_to_combine == "above":
            # Second most fine-grained + most fine-grained
            return [copy_attributions[-2], copy_attributions[-1]]

    def __call__(self, attributions: List[torch.Tensor]):
        raise NotImplementedError()
        pass


class MultiplierMix(Mixer):
    def __init__(self, layers_to_combine: Literal["all", "top", "above"] = "all"):
        Mixer.__init__(self, layers_to_combine)

    def __call__(self, attributions: List[torch.Tensor]):
        """
        The attributions are assumed to be ordered from the most coarse to the most fine-grained.
        """
        if len(attributions) == 1:
            return attributions[0]

        attributions = self.filter_layers(attributions)

        result = attributions[0]
        for attr in attributions[1:]:
            result *= attr
        return result


class LogExpMix(Mixer):
    def __init__(self, layers_to_combine: Literal["all", "top", "above"] = "all"):
        Mixer.__init__(self, layers_to_combine)

    def __call__(self, attributions: List[torch.Tensor]):
        """
        The attributions are assumed to be ordered from the most coarse to the most fine-grained.
        """
        if len(attributions) == 1:
            return attributions[0]

        attributions = self.filter_layers(attributions)

        numerator = torch.log(torch.tensor([(len(attributions))])) + 1
        denominator = torch.log(
            torch.sum(torch.exp(1 / torch.stack(attributions)), dim=0)
        )

        return numerator / denominator
