import torch
from torch import nn

class BaseCAM:
    """
    Base class for Class Activation Mapping (CAM).

    Args:
        model_dict (dict): Must contain:
            - 'arch': the torch.nn.Module model
            - either 'target_layer' (nn.Module) or 'layer_name' (str)
    """
    def __init__(self, model_dict):
        # Retrieve model
        if 'arch' not in model_dict:
            raise KeyError("model_dict must have key 'arch' for the model architecture")
        self.model = model_dict['arch']
        self.model_arch = self.model  # alias for backward compatibility
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        # Determine target layer (module or string)
        layer_spec = model_dict.get('target_layer', None)
        if layer_spec is None:
            layer_spec = model_dict.get('layer_name', None)
        if layer_spec is None:
            raise KeyError("model_dict must have 'target_layer' or 'layer_name'")

        # Resolve module if given by name
        if isinstance(layer_spec, nn.Module):
            self.target_layer = layer_spec
        else:
            self.target_layer = self._find_layer_by_name(self.model, layer_spec)

        # Storage for activations and gradients
        self.activations = {}
        self.gradients = {}

        # Register hooks
        self._register_hooks()

    def _find_layer_by_name(self, model: nn.Module, layer_name: str) -> nn.Module:
        """
        Find a submodule by hierarchical name, e.g. 'layer4.1.conv1' or 'features.3'.
        """
        parts = layer_name.split('.')
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit():
                # numeric index for sequential
                module = list(module.children())[int(part)]
            else:
                raise ValueError(f"Cannot find submodule '{part}' in {module}")
        if not isinstance(module, nn.Module):
            raise ValueError(f"Resolved object '{layer_name}' is not a nn.Module")
        return module

    def _register_hooks(self):
        """
        Attach forward and backward hooks to the target layer.
        """
        def forward_hook(module, input, output):
            # Save output activation
            self.activations['value'] = output

        def backward_hook(module, grad_input, grad_output):
            # Save gradient wrt output
            self.gradients['value'] = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        # use full backward hook for latest PyTorch
        self.target_layer.register_full_backward_hook(backward_hook)

    def forward(self, input_tensor: torch.Tensor, class_idx=None, retain_graph=False):
        """
        Computes CAM; override in subclass.
        """
        raise NotImplementedError

    def __call__(self, input_tensor: torch.Tensor, class_idx=None, retain_graph=False):
        return self.forward(input_tensor, class_idx, retain_graph)
