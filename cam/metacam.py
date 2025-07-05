import torch
import torch.nn.functional as F
from cam.basecam import BaseCAM
from sklearn.cluster import KMeans


class ClusterScoreCAM(BaseCAM):
    """
    Score-CAM với clustering, tuỳ chỉnh zero-out và temperature đối với một class nhất định.

    Args:
        model_dict: dict giống BaseCAM
        num_clusters: số cụm K
        zero_ratio: tỉ lệ phần trăm cluster nhỏ nhất bị loại (0-1)
        temperature_dict: dict class_idx, temperature
        default_temperature:
    """
    def __init__(
        self,
        model_dict,
        num_clusters=10,
        zero_ratio=0.5,
        temperature_dict=None,
        temperature=0.5
    ):
        super().__init__(model_dict)
        self.K = num_clusters
        self.zero_ratio = zero_ratio
        self.temperature_dict = temperature_dict or {}
        self.temperature = temperature

    def forward(self, input, class_idx=None, retain_graph=False):
        # Input: (1,C,H,W)
        b, c, h, w = input.size()

        # 1) Forward pass + chọn class
        logits = self.model_arch(input)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        # nếu class_idx là tensor, chuyển về int
        elif isinstance(class_idx, torch.Tensor):
            class_idx = int(class_idx)
        base_score = logits[0, class_idx]

        # 2) Backprop lấy activation maps (low-res)
        self.model_arch.zero_grad()
        base_score.backward(retain_graph=retain_graph)
        activations = self.activations['value'][0]  # (nc, u, v)
        nc, u, v = activations.shape

        # 3) Upsample & normalize mỗi activation map lên input size
        up_maps = []
        for i in range(nc):
            a = activations[i:i+1].unsqueeze(0)  # (1,1,u,v)
            a_up = F.interpolate(
                a, size=(h, w), mode='bilinear', align_corners=False
            )[0, 0]
            if a_up.max() != a_up.min():
                a_up = (a_up - a_up.min()) / (a_up.max() - a_up.min())
            up_maps.append(a_up)
        up_maps = torch.stack(up_maps, dim=0)  # (nc, h, w)

        # 4) Flatten upsampled maps và clustering
        flat_maps = up_maps.reshape(nc, -1).detach().cpu().numpy()  # (nc, h*w)
        kmeans = KMeans(n_clusters=self.K, init='k-means++', random_state=0)
        print(f"[ClusterScoreCAM] Running KMeans++ with {self.K} clusters...")
        kmeans.fit(flat_maps)
        rep_maps = torch.from_numpy(
            kmeans.cluster_centers_.reshape(self.K, h, w)
        ).to(activations.device)
        
        #them de visualize 
        self.rep_maps = rep_maps  # tensor (K, h, w)
        self.base_score = base_score  # cũng lưu nếu cần debug

        # 5) Tính score difference mỗi mask
        diffs = torch.zeros(self.K, device=activations.device)
        with torch.no_grad():
            for k in range(self.K):
                mask = rep_maps[k:k+1].unsqueeze(0)  # (1,1,h,w)
                out = self.model_arch(input * mask)
                diffs[k] = out[0, class_idx] - base_score

        # 6) Zero-out cluster nhỏ nhất
        num_zero = int(self.zero_ratio * self.K)
        if num_zero > 0:
            lowest = torch.argsort(diffs)[:num_zero]
            diffs[lowest] = float('-inf')

        # 7) Softmax với nhiệt độ class
        T = self.temperature_dict.get(class_idx, self.temperature)
        weights = F.softmax(diffs / T, dim=0)

        # 8) Kết hợp saliency map
        sal = torch.zeros(1,1,h,w, device=activations.device)
        for k in range(self.K):
            sal += weights[k] * rep_maps[k:k+1].unsqueeze(0)

        # 9) Post-process + normalize
        sal = F.relu(sal)
        mn, mx = sal.min(), sal.max()
        if mn == mx:
            return None
        sal = (sal - mn) / (mx - mn)
        
        self.last_saliency = sal  # tensor (1,1,h,w)

        return sal
# khong chay voi batch
    # def __call__(self, input, class_idx=None, retain_graph=False):
    #     return self.forward(input, class_idx, retain_graph)
    def __call__(self, input: torch.Tensor, class_idx=None, retain_graph=False):
        """
         Hỗ trợ cả single (1,C,H,W) và batch (B,C,H,W).
         Nếu batch, sẽ gọi forward() cho từng sample và
         trả về saliency maps dạng (B,1,H,W).
         """
        # batch đầu tiên
        if input.dim() == 4 and input.size(0) > 1:
            B = input.size(0)
            print(f"[ClusterScoreCAM] Detected batch size: {B}")
            # chuyển class_idx về list có độ dài B
            if class_idx is None:
                cls_list = [None] * B
            elif isinstance(class_idx, torch.Tensor):
                cls_list = class_idx.detach().cpu().tolist()
            elif isinstance(class_idx, (list, tuple)):
                cls_list = list(class_idx)
            else:
                # cùng 1 label cho tất cả
                cls_list = [int(class_idx)] * B

            outs = []
            for i in range(B):
                print(f"[ClusterScoreCAM] Processing sample {i+1}/{B}, class_idx={cls_list[i]}")
                inp_i = input[i : i+1]
                ci = cls_list[i]
                sal_i = self.forward(inp_i, ci, retain_graph)
                # forward trả về (1,1,H,W)
                outs.append(sal_i)
            # ghép lại thành (B,1,H,W)
            return torch.cat(outs, dim=0)
        # không phải batch, gọi bình thường
        return self.forward(input, class_idx, retain_graph)
