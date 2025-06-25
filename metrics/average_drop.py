import torch
import torch.nn as nn
from .metric_utils import MetricBase, mix_image_with_saliency
import numpy as np 
import matplotlib.pyplot as plt


#1 Average Drop metric 
class AverageDrop(MetricBase):
    def __init__(self):
        super().__init__("Average_drop")
    def __call__(self, 
        model: nn.Module, 
        test_images: torch.Tensor, 
        saliency_maps: torch.Tensor, 
        class_idx: int | torch.Tensor, 
        device: str = "cpu", 
        apply_softmax: bool = True, 
        return_mean: bool = True,
        **kwargs, 
        ) -> torch.Tensor: 
        """
        The Average Drop refers to the maximum positive difference in the predictions made by the predictor using
        the input image and the prediction using the saliency map.
        Instead of giving to the model the original image, we give the saliency map as input and expect it to drop
        in performances if the saliency map doesn't contain relevant information.

        Args:
            model (torch.nn.Module): The model to be evaluated.
            test_images (torch.Tensor): The images to be tested, shape: (N, C, G, W)
            saliency_maps (torch.Tensor): The saliency maps to be evaluated, shape (N, C, H, W)
            class_idx (int): If int: the index of the class to be evaluated, the same for all the input images.
            if torch.Tensor: the index of the class to be evaluated for each input image. Shape: (N,)
        """
        test_images = test_images.to(device)
        # plt.imshow(test_images[0].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()
        
        if isinstance(saliency_maps, np.ndarray):
            saliency_maps = torch.tensor(saliency_maps)
        saliency_maps = saliency_maps.to(device)
        # plt.imshow(saliency_maps[0].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        saliency_images = mix_image_with_saliency(test_images, saliency_maps)
        # plt.imshow(saliency_images[0].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        test_preds = model(test_images)  # Shape: (N, num_classes)
        saliency_preds = model(saliency_images)  # Shape: (N, num_classes)

        if apply_softmax:
            test_preds = nn.functional.softmax(test_preds, dim=1)
            saliency_preds = nn.functional.softmax(saliency_preds, dim=1)

        #Select only the relevant class
        if isinstance(class_idx, int):
            test_preds = test_preds[:, class_idx]  # Shape: (N,)
            saliency_preds = saliency_preds[:, class_idx]  # Shape: (N,)
        elif isinstance(class_idx, torch.Tensor):
            test_preds = test_preds[torch.arange(test_preds.size(0)), class_idx]
            saliency_preds = saliency_preds[
                torch.arange(saliency_preds.size(0)), class_idx
            ]
        else:
            raise ValueError("class_idx should be either an int or a torch.Tensor")

        numerator = test_preds - saliency_preds
        numerator[numerator < 0] = 0

        denominator = test_preds

        res = torch.sum(numerator / denominator) * 100

        if return_mean:
            res = res.mean()

        return res.item()