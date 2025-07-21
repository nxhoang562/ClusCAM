import torch

def IoUEnergyBoxSaliency(bbox, saliency_map, eps=1e-6):
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
