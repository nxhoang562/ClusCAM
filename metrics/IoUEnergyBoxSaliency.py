import torch

def Local_Error(bbox, saliency_map, eps=1e-6):
    """
    Tính energy‑based IoU giữa bounding‑box và saliency map.

    Args:
        bbox (tuple or list of int): (x1, y1, x2, y2) – tọa độ góc trên trái và góc dưới phải.
        saliency_map (torch.Tensor): Tensor kích thước (W, H), giá trị ≥ 0 (float).
        eps (float): hệ số nhỏ tránh chia cho 0.

    Returns:
        torch.Tensor: giá trị IoU năng lượng (scalar).
    """
    # Unpack bbox
    x1, y1, x2, y2 = bbox
    W, H = saliency_map.shape

    # 1. Tạo mask bounding‑box (kiểu float để nhân với saliency)
    mask_box = torch.zeros((W, H), dtype=saliency_map.dtype, device=saliency_map.device)
    mask_box[x1:x2, y1:y2] = 1.0

    # 2. Tính năng lượng giao (intersection energy)
    #    sum saliency_map trong vùng box
    inter_energy = (mask_box * saliency_map).sum()

    # 3. Tính năng lượng hợp (union energy)
    #    bằng tổng năng lượng box + saliency – inter
    union_energy = (mask_box.sum() + saliency_map.sum() - inter_energy).clamp(min=eps)

    # 4. Tỷ số IoU năng lượng
    iou_energy = inter_energy / union_energy

    return 1 - iou_energy

def Local_Error_Binary(bbox, saliency_map, thr=0.5, eps=1e-6):
    """
    Tính 1 - IoU (nhị phân) giữa bbox và vùng saliency >= thr.

    Args:
        bbox: (x1, y1, x2, y2)
        saliency_map: tensor (W,H), giá trị float ≥ 0
        thr: ngưỡng chuyển saliency_map thành 0/1
        eps: tránh chia cho 0

    Returns:
        torch.Tensor: lỗi = 1 - IoU_binary
    """
    x1, y1, x2, y2 = bbox
    W, H = saliency_map.shape

    # mask cho bbox
    mask_box = torch.zeros((W, H), device=saliency_map.device)
    mask_box[x1:x2, y1:y2] = 1.0

    # mask nhị phân saliency
    mask_sal = (saliency_map >= thr).float()

    # giao và hợp (binary)
    inter = (mask_box * mask_sal).sum()
    union = (mask_box + mask_sal - mask_box*mask_sal).clamp(min=eps).sum()

    iou = inter / union
    return 1 - iou


def Local_Error_EnergyThreshold(bbox, saliency_map, thr=0.5, eps=1e-6):
    """
    Tính 1 - (sum saliency>=thr trong box) / (tổng saliency>=thr + box - giao)
    """
    x1, y1, x2, y2 = bbox
    W, H = saliency_map.shape

    # mask bbox
    mask_box = torch.zeros((W, H), device=saliency_map.device)
    mask_box[x1:x2, y1:y2] = 1.0

    # chỉ lấy saliency cao hơn ngưỡng
    sal_th = torch.where(saliency_map >= thr, saliency_map, torch.zeros_like(saliency_map))

    inter_energy = (mask_box * sal_th).sum()
    union_energy = (mask_box.sum() + sal_th.sum() - inter_energy).clamp(min=eps)

    energy_iou = inter_energy / union_energy
    return 1 - energy_iou


