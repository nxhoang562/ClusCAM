# import copy
# import torch
# from torchvision.models.inception import Inception3

# class ReciproCam:
#     '''
#     ReciproCam class contains official implementation of Reciprocal CAM algorithm for CNN architecture 
#     published at CVPR2024 XAI4CV workshop.
#     '''

#     def __init__(self, model, device, target_layer_name=None):
#         '''
#         Creator of ReciproCAM class
        
#         Args:
#             model: CNN architectur pytorch model
#             device: runtime device type (ex, 'cuda', 'cpu')
#             target_layer_name: layer name for understanding the layer's activation
#         '''

#         # self.model = copy.deepcopy(model)
#         # self.model.eval()
#         # self.target_layer_name = target_layer_name
#         # self.device = device
#         # self.feature = None
#         # filter = [[1/16.0, 1/8.0, 1/16.0],
#         #             [1/8.0, 1/4.0, 1/8.0],
#         #             [1/16.0, 1/8.0, 1/16.0]]
#         # self.gaussian = torch.tensor(filter).to(device)
#         # self.softmax = torch.nn.Softmax(dim=1)
#         # self.target_layers = []
#         # self.conv_depth = 0
#         # if self.target_layer_name is not None:
#         #     children = dict(self.model.named_children())
#         #     if self.target_layer_name in children:
#         #         target = children[self.target_layer_name]
#         #         self._find_target_layer(target)
#         #     else:
#         #         self._find_target_layer(self.model)
#         # else:
#         #     self._find_target_layer(self.model)
        
#         self.device = device

#         # --- KHỞI TẠO CHUNG PHẢI CÓ TRƯỚC ---
#         # filter Gaussian sẽ luôn tồn tại
#         filt = [[1/16.0, 1/8.0, 1/16.0],
#                 [1/8.0, 1/4.0, 1/8.0],
#                 [1/16.0, 1/8.0, 1/16.0]]
#         self.gaussian = torch.tensor(filt)     # để batch_test tự .to(device)
#         self.softmax = torch.nn.Softmax(dim=1)
#         self.feature = None

#         # Deep-copy và eval model
#         self.model = copy.deepcopy(model)
#         self.model.eval()

#         # Chỉ dành riêng cho Inception v3: hook vào Mixed_7c
#         if isinstance(model, Inception3):
#             self.target_layers = [ self.model.Mixed_7c ]
#         else:
#             # với CNN khác, dùng cơ chế target_layer_name cũ
#             self.target_layer_name = target_layer_name
#             self.target_layers = []
#             self.conv_depth = 0
#             if self.target_layer_name:
#                 children = dict(self.model.named_children())
#                 if self.target_layer_name in children:
#                     self._find_target_layer(children[self.target_layer_name])
#                 else:
#                     self._find_target_layer(self.model)
#             else:
#                 self._find_target_layer(self.model)

        
#         self.target_layers[-1].register_forward_hook(self._cam_hook())


#     def _find_target_layer(self, m, depth=0):
#         '''
#         Searching target layer by name from given network model as recursive manner.
#         '''

#         children = dict(m.named_children())
#         if not children:
#             if isinstance(m, torch.nn.Conv2d):
#                 self.target_layers.clear()
#                 self.target_layers.append(m)
#                 self.conv_depth = depth
#             elif self.conv_depth == depth and any(self.target_layers) and isinstance(m, torch.nn.BatchNorm2d):
#                 self.target_layers.append(m)
#             elif self.conv_depth == depth and any(self.target_layers) and isinstance(m, torch.nn.ReLU):
#                 self.target_layers.append(m)
#         else:
#             for name, child in children.items():
#                 self._find_target_layer(child, depth+1)


#     def _cam_hook(self):
#         '''
#         Setup hook funtion for generating new masked features for calculating reciprocal activation score 
#         '''

#         def fn(_, input, output):
#             self.feature = output[0].unsqueeze(0)
#             bs, nc, h, w = self.feature.shape
#             new_features = self._mosaic_feature(self.feature, nc, h, w, False)
#             new_features = torch.cat((self.feature, new_features), dim = 0)
#             return new_features

#         return fn


#     def _mosaic_feature(self, feature_map, nc, h, w, is_gaussian=False):
#         '''
#         Generate spatially masked feature map [h*w, nc, h, w] from input feature map [1, nc, h, w].
#         If is_gaussian is true then the spatially masked feature map's value are filtered by 3x3 Gaussian filter.  
#         '''
        
#         if is_gaussian == False:
#             feature_map_repeated = feature_map.repeat(h * w, 1, 1, 1)
#             mosaic_feature_map_mask = torch.zeros(h * w, nc, h, w).to(self.device)
#             spatial_order = torch.arange(h * w).reshape(h, w)
#             for i in range(h):
#                 for j in range(w):
#                     k = spatial_order[i, j]
#                     mosaic_feature_map_mask[k, :, i, j] = torch.ones(nc).to(self.device)
#             new_features = feature_map_repeated * mosaic_feature_map_mask
#         else:
#             new_features = torch.zeros(h*w, nc, h, w).to(self.device)
#             for b in range(h*w):
#                 for i in range(h): #0...h-1
#                     kx_s = max(i-1, 0)
#                     kx_e = min(i+1, h-1)
#                     if i == 0: sx_s = 1
#                     else: sx_s = 0
#                     if i == h-1: sx_e = 1
#                     else: sx_e = 2
#                     for j in range(w): #0...w-1
#                         ky_s = max(j-1, 0)
#                         ky_e = min(j+1, w-1)
#                         if j == 0: sy_s = 1
#                         else: sy_s = 0
#                         if j == w-1: sy_e = 1
#                         else: sy_e = 2
#                         if b == i*w + j:
#                             r_feature_map = feature_map[0,:,i,j].reshape(feature_map.shape[1],1,1)
#                             r_feature_map = r_feature_map.repeat(1,3,3)
#                             score_map = r_feature_map*self.gaussian.repeat(feature_map.shape[1],1,1)
#                             new_features[b,:,kx_s:kx_e+1,ky_s:ky_e+1] = score_map[:,sx_s:sx_e+1,sy_s:sy_e+1]

#         return new_features


#     def _get_class_activaton_map(self, mosaic_predic, index, h, w):
#         '''
#         Calculate class activation map from the prediction result of mosaic feature input.
#         '''
        
#         cam = (mosaic_predic[:,index]).reshape((h, w))
#         cam_min = cam.min()
#         cam = (cam - cam_min) / (cam.max() - cam_min)
    
#         return cam


#     def __call__(self, input, index=None):
#         with torch.no_grad():
#             BS, _, _, _ = input.shape
#             if BS > 1:
#                 cam = []
#                 for b in range(BS):
#                     prediction = self.model(input[b, : , :, :].unsqueeze(0))
#                     prediction = self.softmax(prediction)
#                     if index == None:
#                         index = prediction[0].argmax().item()
#                     _, _, h, w = self.feature.shape
#                     cam.append(self._get_class_activaton_map(prediction[1:, :], index, h, w))
#             else:
#                 prediction = self.model(input)
#                 prediction = self.softmax(prediction)
#                 if index == None:
#                     index = prediction[0].argmax().item()
#                 _, _, h, w = self.feature.shape
#                 cam = self._get_class_activaton_map(prediction[1:, :], index, h, w)

#         return cam, index

import copy
import math
import torch
import torch.nn.functional as F
from torchvision.models.swin_transformer import SwinTransformer

class ReciproCam:
    '''
    Reciprocal CAM class implements Reciprocal CAM algorithm with support for:
      - CNNs (hook + mosaic)
      - ViT (token masking)
      - Swin-Transformer (token masking)
    '''
    def __init__(self, model, device, target_layer_name=None):
        '''
        Args:
            model: PyTorch CNN, ViT, or SwinTransformer
            device: 'cuda' or 'cpu'
            target_layer_name: only for CNN branch
        '''
        # Deepcopy to avoid modifying original
        self.model = copy.deepcopy(model).to(device)
        self.model.eval()
        self.device = device
        self.softmax = torch.nn.Softmax(dim=1)

        # Gaussian filter kernel
        filt = [[1/16.0, 1/8.0, 1/16.0],
                [1/8.0,  1/4.0, 1/8.0],
                [1/16.0, 1/8.0, 1/16.0]]
        self.gaussian = torch.tensor(filt, device=device)

        # Detect ViT vs Swin
        self.is_vit  = hasattr(self.model, '_process_input') and hasattr(self.model, 'encoder')
        self.is_swin = isinstance(self.model, SwinTransformer)

        # CNN branch: find deepest conv layer and register hook
        if not self.is_vit and not self.is_swin:
            self.target_layer_name = target_layer_name
            self.feature = None
            self.target_layers = []
            self.conv_depth = 0
            children = dict(self.model.named_children())
            if target_layer_name and target_layer_name in children:
                self._find_target_layer(children[target_layer_name])
            else:
                self._find_target_layer(self.model)
            self.target_layers[-1].register_forward_hook(self._cam_hook())

        # Cache for masks
        self._cached_masks = {}

    def _find_target_layer(self, module, depth=0):
        children = dict(module.named_children())
        if not children:
            if isinstance(module, torch.nn.Conv2d):
                # found a conv: reset list at this depth
                self.target_layers.clear()
                self.target_layers.append(module)
                self.conv_depth = depth
            elif self.conv_depth == depth and self.target_layers:
                # after deepest conv, include succeeding BN and ReLU
                if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.ReLU):
                    self.target_layers.append(module)
        else:
            for child in children.values():
                self._find_target_layer(child, depth+1)

    def _cam_hook(self):
        def hook_fn(module, input, output):
            # save original feature map
            self.feature = output[0].unsqueeze(0)  # [1, C, H, W]
            bs, nc, h, w = self.feature.shape
            # create mosaic features (no Gaussian)
            new_feats = self._mosaic_feature(self.feature, nc, h, w, is_gaussian=False)
            # concatenate original + mosaic
            return torch.cat((self.feature, new_feats), dim=0)
        return hook_fn

    def _get_mask(self, nc, h, w):
        key = (nc, h, w)
        if key not in self._cached_masks:
            # mask shape: [h*w, C, h, w], each entry m has one spatial cell kept
            idx = torch.arange(h*w, device=self.device).view(h, w)
            mask = torch.zeros(h*w, nc, h, w, device=self.device)
            for i in range(h):
                for j in range(w):
                    mask[idx[i,j], :, i, j] = 1.0
            self._cached_masks[key] = mask
        return self._cached_masks[key]

    def _mosaic_feature(self, feat, nc, h, w, is_gaussian=False):
        if not is_gaussian:
            mask     = self._get_mask(nc, h, w)            # [h*w, C, h, w]
            expanded = feat.unsqueeze(0).expand(h*w, -1, -1, -1)
            return expanded * mask
        else:
            # apply Gaussian blur via grouped conv2d
            kernel  = self.gaussian.view(1,1,3,3).repeat(nc,1,1,1)
            blurred = F.conv2d(feat, kernel, padding=1, groups=nc)
            mask    = self._get_mask(nc, h, w)
            expanded= blurred.unsqueeze(0).expand(h*w, -1, -1, -1)
            return expanded * mask

    def __call__(self, img, index=None):
        img = img.to(self.device)
        with torch.no_grad():
            # --- Swin-Transformer branch ---
            if self.is_swin:
                # 1) patch embedding is the first block in model.features
                x_embed = self.model.features[0](img)       # [B, H', W', C]
                B, Ht, Wt, C = x_embed.shape
                tokens = x_embed.view(B, Ht*Wt, C)          # [B, N, C]

                # 2) define forward on masked tokens
                def forward_masked(tok_flat):
                    # reshape back to spatial grid
                    x_sp = tok_flat.view(B, Ht, Wt, C)
                    # pass through all subsequent stages
                    for stage in self.model.features[1:]:
                        x_sp = stage(x_sp)
                    # final norm, permute, pooling, flatten, head
                    x_sp = self.model.norm(x_sp)            # [B, Hout, Wout, Cout]
                    x_sp = self.model.permute(x_sp)         # [B, Cout, Hout, Wout]
                    x_sp = self.model.avgpool(x_sp)         # [B, Cout, 1, 1]
                    x_sp = self.model.flatten(x_sp)         # [B, Cout]
                    return self.model.head(x_sp)            # [B, num_classes]

                # 3) mask each token and score
                scores = []
                for i in range(Ht*Wt):
                    mask_tok = torch.zeros_like(tokens)
                    mask_tok[:, i] = tokens[:, i]
                    logits_i = forward_masked(mask_tok)     # [B, num_classes]
                    p_i      = self.softmax(logits_i)
                    if index is None:
                        index = p_i.argmax(dim=1).item()
                    scores.append(p_i[0, index])

                # 4) reshape scores into heatmap
                grid = Ht
                cam  = torch.stack(scores).reshape(grid, grid)
                cam  = (cam - cam.min()) / (cam.max() - cam.min())
                return cam.cpu(), index

            # --- ViT branch ---
            if self.is_vit:
                x   = self.model._process_input(img)      # [1, n_patches, C]
                cls = self.model.class_token.expand(1, -1, -1)
                x   = torch.cat([cls, x], dim=1)          # [1, n+1, C]
                enc = self.model.encoder(x)               # [1, n+1, C]
                out = self.model.heads(enc[:, 0])         # [1, num_classes]
                prob= self.softmax(out)
                if index is None:
                    index = prob.argmax(dim=1).item()

                n_tokens = x.shape[1]
                grid     = int(math.sqrt(n_tokens - 1))
                scores   = []
                for i in range(1, n_tokens):
                    xm     = torch.zeros_like(x)
                    xm[:,0] = x[:,0]                       # keep cls
                    xm[:,i] = x[:,i]                       # keep patch i
                    enc_i  = self.model.encoder(xm)
                    out_i  = self.model.heads(enc_i[:, 0])
                    scores.append(self.softmax(out_i)[0, index])
                cam = torch.stack(scores).reshape(grid, grid)
                cam = (cam - cam.min()) / (cam.max() - cam.min())
                return cam.cpu(), index

            # --- CNN branch ---
            BS = img.size(0)
            if BS > 1:
                cams, cls_idx = [], index
                for b in range(BS):
                    pred = self.model(img[b:b+1])
                    pred = self.softmax(pred)
                    if cls_idx is None:
                        cls_idx = pred[0].argmax().item()
                    _, nc, h, w = self.feature.shape
                    cams.append(self._get_class_activation_map(pred[1:], cls_idx, h, w))
                return cams, cls_idx
            else:
                pred = self.model(img)
                pred = self.softmax(pred)
                if index is None:
                    index = pred[0].argmax().item()
                _, nc, h, w = self.feature.shape
                cam = self._get_class_activation_map(pred[1:], index, h, w)
                return cam, index

    def _get_class_activation_map(self, mosaic_pred, index, h, w):
        # mosaic_pred: [h*w, num_classes]
        cam = mosaic_pred[:, index].view(h, w)
        return (cam - cam.min()) / (cam.max() - cam.min())