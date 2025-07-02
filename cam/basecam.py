"""
Part of code borrows from https://github.com/1Konny/gradcam_plus_plus-pytorch
Modified to accept direct target_layer module in model_dict.
"""
import torch
from utils import (
    find_alexnet_layer, find_vgg_layer, find_resnet_layer,
    find_densenet_layer, find_squeezenet_layer, find_layer,
    find_googlenet_layer, find_mobilenet_layer, find_shufflenet_layer
)

class BaseCAM(object):
    """
    Base class for Class activation mapping.

    : Args
        - **model_dict**: Dict with format:
            {
              'type': 'vgg',
              'arch': torchvision.models.vgg16(pretrained=True),
              'layer_name': 'features',      # optional if target_layer given
              'target_layer': <nn.Module>,    # optional, direct module
              'input_size': (224, 224)
            }
    """
    def __init__(self, model_dict):
        # 1) Initialize model
        self.model_arch = model_dict['arch']
        self.model_arch.eval()
        if torch.cuda.is_available():
            self.model_arch.cuda()

        # storage for hooks
        self.gradients = {}
        self.activations = {}

        # define hooks
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = (
                grad_output[0].cuda()
                if torch.cuda.is_available()
                else grad_output[0]
            )
        def forward_hook(module, input, output):
            self.activations['value'] = (
                output.cuda()
                if torch.cuda.is_available()
                else output
            )

        # 2) Determine target_layer: direct or via find_* functions
        if 'target_layer' in model_dict:
            # use the module passed directly
            self.target_layer = model_dict['target_layer']
        else:
            # fallback to legacy layer_name lookup
            model_type = model_dict.get('type', '')
            layer_name = model_dict['layer_name']
            if 'vgg' in model_type.lower():
                self.target_layer = find_vgg_layer(self.model_arch, layer_name)
            elif 'resnet' in model_type.lower():
                self.target_layer = find_resnet_layer(self.model_arch, layer_name)
            elif 'densenet' in model_type.lower():
                self.target_layer = find_densenet_layer(self.model_arch, layer_name)
            elif 'alexnet' in model_type.lower():
                self.target_layer = find_alexnet_layer(self.model_arch, layer_name)
            elif 'squeezenet' in model_type.lower():
                self.target_layer = find_squeezenet_layer(self.model_arch, layer_name)
            elif 'googlenet' in model_type.lower():
                self.target_layer = find_googlenet_layer(self.model_arch, layer_name)
            elif 'shufflenet' in model_type.lower():
                self.target_layer = find_shufflenet_layer(self.model_arch, layer_name)
            elif 'mobilenet' in model_type.lower():
                self.target_layer = find_mobilenet_layer(self.model_arch, layer_name)
            else:
                # generic fallback
                self.target_layer = find_layer(self.model_arch, layer_name)

        # 3) Register hooks on the chosen layer
        self.target_layer.register_forward_hook(forward_hook)
        
        self.target_layer.register_full_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        raise NotImplementedError("Must be implemented in subclasses")

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
