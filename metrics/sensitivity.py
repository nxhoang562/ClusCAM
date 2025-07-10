import torch
import torch.nn as nn
import numpy as np
from captum.metrics import sensitivity_max
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .metric_utils import MetricBase, AttributionMethod

class Sensitivity(MetricBase):
    def __init__(self):
        super().__init__("sensitivity")

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        class_idx: int | torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        return_mean: bool = True,
        **kwargs,
    ) -> float | None:
        """
        Compute the sensitivity_max metric using either Captum or PyTorch-Grad-CAM methods.
        Expects kwargs to include:
          - 'layer': target layer module for PyTorch-Grad-CAM
          - 'baseline_dist': distribution for sensitivity_max
        """
        def attribution_wrapper(
            images: torch.Tensor, model, layer=None, targets=None, **inner_kwargs
        ) -> torch.Tensor:
            # Unpack tuple if needed
            if isinstance(images, tuple) and len(images) == 1:
                images = images[0]

            BATCH_SIZE = 1
            FINAL_SIZE = 10 // BATCH_SIZE
            res = []
            ATTRIBUTION_SHAPE = None

            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i+BATCH_SIZE].requires_grad_().to(device)
                # Prepare targets
                if isinstance(targets, torch.Tensor):
                    batch_targets = targets.unsqueeze(0) if targets.dim()==0 else targets
                else:
                    batch_targets = torch.tensor([int(targets)], device=device)
                batch_targets = torch.repeat_interleave(batch_targets, BATCH_SIZE, dim=0)

                # Ensure gradients enabled for backward
                with torch.enable_grad():
                    if hasattr(attribution_method, "attribute"):
                        # Captum-style attribution
                        attribution_res = (
                            attribution_method.attribute(
                                input_tensor=batch,
                                model=model,
                                layer=layer,
                                target=batch_targets
                            )
                            .detach().cpu()
                        )
                    else:
                        # PyTorch-Grad-CAM style returns numpy [B, H, W]
                        cam_targets = [ClassifierOutputTarget(int(t)) for t in batch_targets]
                        sal_np = attribution_method(
                            input_tensor=batch,
                            targets=cam_targets
                        )
                        # Convert to numpy then to tensor
                        sal_np = sal_np.detach().cpu().numpy() if isinstance(sal_np, torch.Tensor) else np.array(sal_np)
                        attribution_res = torch.from_numpy(sal_np)
                        if attribution_res.ndim == 3:
                            attribution_res = attribution_res.unsqueeze(1)
                        attribution_res = attribution_res.cpu()

                ATTRIBUTION_SHAPE = attribution_res.shape
                if torch.isnan(attribution_res).any():
                    continue
                res.append(attribution_res)

            if not res:
                res = [torch.randn(ATTRIBUTION_SHAPE) * 1e6 for _ in range(FINAL_SIZE)]
            if len(res) != FINAL_SIZE:
                res += [res[-1]] * (FINAL_SIZE - len(res))

            return torch.cat(res, dim=0)

        # Prepare for sensitivity_max
        kwargs['model'] = model
        kwargs['targets'] = class_idx

        sens = sensitivity_max(attribution_wrapper, test_images, **kwargs)

        if (sens > 100).any():
            return None
        if return_mean:
            return float(torch.mean(sens).detach().cpu())
        return sens.detach().cpu()


# import torch
# import torch.nn as nn
# import numpy as np
# from captum.metrics import sensitivity_max
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from .metric_utils import MetricBase, AttributionMethod

# class Sensitivity(MetricBase):
#     def __init__(self):
#         super().__init__("sensitivity")

#     def __call__(
#         self,
#         model: nn.Module,
#         test_images: torch.Tensor,
#         class_idx: int | torch.Tensor,
#         attribution_method: AttributionMethod,
#         device: torch.device | str = "cpu",
#         return_mean: bool = True,
#         # Tham số cho sensitivity_max
#         n_perturb_samples: int = 10,
#         step: float = 0.01,
#         **kwargs,
#     ) -> float | torch.Tensor | None:
#         """
#         Compute the sensitivity_max metric via Captum.
#         Expects in kwargs:
#           - 'layer': target layer module for PyTorch-Grad-CAM
#           - 'baseline_dist': a torch.distributions.Distribution for perturbations
#         """

#         # 1) Đảm bảo inputs và target là leaf tensors
#         inputs = test_images.clone().detach().to(device)
#         if isinstance(class_idx, torch.Tensor):
#             target = class_idx.clone().detach().to(device)
#         else:
#             target = torch.tensor([int(class_idx)], device=device)

#         # 2) Lấy layer và baseline distribution ra khỏi kwargs
#         layer = kwargs.pop("layer", None)
#         baseline_distribution = kwargs.pop("baseline_dist", None)

#         # 3) Wrapper chỉ nhận images & target, và swallow mọi kwargs khác
#         def attribution_wrapper(images: torch.Tensor, target=None, **unused):
#             """
#             images: có thể là [n_samples, batch, C, H, W] hoặc [batch, C, H, W]
#             target: int hoặc Tensor
#             Trả về Tensor [n_perturb_samples, …] trên CPU
#             """
#             # Nếu Captum xếp thêm dim sample đầu tiên, flatten nó
#             if images.ndim == 5:
#                 # [n_samples, batch, C, H, W] → [n_samples*batch, C, H, W]
#                 images = images.flatten(0, 1)

#             results = []
#             dummy_shape = None

#             for img in images:
#                 # img: [C, H, W], unsqueeze thành batch 1
#                 if img.ndim == 3:
#                     img_tensor = img.unsqueeze(0).requires_grad_().to(device)
#                 elif img.ndim == 4:
#                     img_tensor = img.requires_grad_().to(device)
#                 else:
#                     raise RuntimeError(f"Unexpected img.dim()={img.ndim}")

#                 # Chuẩn bị target cho batch này
#                 if isinstance(target, torch.Tensor):
#                     cls_tensor = target.to(device)
#                 else:
#                     cls_tensor = torch.tensor([int(target)], device=device)

#                 # Tính attribution
#                 if hasattr(attribution_method, "attribute"):
#                     # Captum-style
#                     attr = attribution_method.attribute(
#                         input_tensor=img_tensor,
#                         model=model,
#                         layer=layer,
#                         target=cls_tensor,
#                     ).detach().cpu()
#                 else:
#                     # Grad-CAM style
#                     cam_t = [ClassifierOutputTarget(int(cls_tensor.item()))]
#                     sal = attribution_method(input_tensor=img_tensor, targets=cam_t)
#                     sal_np = (
#                         sal.detach().cpu().numpy()
#                         if isinstance(sal, torch.Tensor)
#                         else np.array(sal)
#                     )
#                     attr = torch.from_numpy(sal_np)
#                     if attr.ndim == 3:  # [B, H, W] → [B, 1, H, W]
#                         attr = attr.unsqueeze(1)

#                 dummy_shape = attr.shape
#                 # Nếu không NaN, giữ lại
#                 if not torch.isnan(attr).any():
#                     results.append(attr.cpu())

#             # Nếu không có kết quả, tạo dummy noise lớn
#             if not results:
#                 dummy = torch.randn(dummy_shape) * 1e6
#                 results = [dummy.clone() for _ in range(n_perturb_samples)]

#             # Đảm bảo trả đúng số perturb samples
#             if len(results) < n_perturb_samples:
#                 results += [results[-1]] * (n_perturb_samples - len(results))

#             return torch.cat(results, dim=0)

#         # 4) Chạy sensitivity_max (Captum sẽ deep-copy baseline_distribution, nhưng không deep-copy model/layer)
#         sens = sensitivity_max(
#             attribution_wrapper,
#             inputs,
#             target=target,
#             baseline_distribution=baseline_distribution,
#             n_perturb_samples=n_perturb_samples,
#             step=step,
#         )

#         # 5) Xử lý kết quả
#         if (sens > 100).any():
#             return None
#         if return_mean:
#             return float(sens.mean().detach().cpu())
#         return sens.detach().cpu()





