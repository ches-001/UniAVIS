import torch

def compute_2d_ciou(preds_2dbox: torch.Tensor, targets_2dbox: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    """
    Complete Intersection Over Union for 3D boxes.

    preds_2dbox: (N, ..., 4)
    targets_2dbox: (N, ..., 4)

    [center_x, center_y, width, height]
    """
    assert preds_2dbox.shape[-1] == targets_2dbox.shape[-1] and targets_2dbox.shape[-1] == 4

    preds_w = preds_2dbox[..., 2:3]
    preds_h = preds_2dbox[..., 3:4]
    preds_x1 = preds_2dbox[..., 0:1] - (preds_w / 2)
    preds_y1 = preds_2dbox[..., 1:2] - (preds_h / 2)
    preds_x2 = preds_x1 + preds_w
    preds_y2 = preds_y1 + preds_h

    targets_w = targets_2dbox[..., 2:3]
    targets_h = targets_2dbox[..., 3:4]
    targets_x1 = targets_2dbox[..., 0:1] - (targets_w / 2)
    targets_y1 = targets_2dbox[..., 1:2] - (targets_h / 2)
    targets_x2 = targets_x1 + targets_w
    targets_y2 = targets_y1 + targets_h

    intersection_w = (torch.min(preds_x2, targets_x2) - torch.max(preds_x1, targets_x1)).clip(min=0)
    intersection_h = (torch.min(preds_y2, targets_y2) - torch.max(preds_y1, targets_y1)).clip(min=0)
    intersection = intersection_w * intersection_h
    union = (preds_w * preds_h) + (targets_w * targets_h) - intersection
    iou = intersection / (union + eps)

    cw = (torch.max(preds_x2, targets_x2) - torch.min(preds_x1, targets_x1))
    ch = (torch.max(preds_y2, targets_y2) - torch.min(preds_y1, targets_y1))
    c_sq = cw.pow(2) + ch.pow(2) + eps
    v = (4 / (torch.pi**2)) * (torch.arctan(targets_w / targets_h) - torch.arctan(preds_w / preds_h)).pow(2)
    d_sq = (preds_2dbox[..., :1] - targets_2dbox[..., :1]).pow(2) + (preds_2dbox[..., 1:2] - targets_2dbox[..., 1:2]).pow(2)

    with torch.no_grad():
        a = v / (v - iou + (1 + eps))

    ciou = iou - ((d_sq / c_sq) + (a * v))
    return ciou.squeeze(-1)


def compute_3d_ciou(preds_3dbox: torch.Tensor, targets_3dbox: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    """
    Complete Intersection Over Union for 3D boxes.

    preds_3dbox: (N, ..., 6)
    targets_3dbox: (N, ..., 6)

    [center_x, center_y, center_z, length, width, height]
    """
    assert preds_3dbox.shape[-1] == targets_3dbox.shape[-1] and targets_3dbox.shape[-1] == 6

    preds_l = preds_3dbox[..., 3:4]
    preds_w = preds_3dbox[..., 4:5]
    preds_h = preds_3dbox[..., 5:6]
    preds_x1 = preds_3dbox[..., 0:1] - (preds_l / 2)
    preds_y1 = preds_3dbox[..., 1:2] - (preds_w / 2)
    preds_z1 = preds_3dbox[..., 2:3] - (preds_h / 2)
    preds_x2 = preds_x1 + preds_l
    preds_y2 = preds_y1 + preds_w
    preds_z2 = preds_z1 + preds_h

    targets_l = targets_3dbox[..., 3:4]
    targets_w = targets_3dbox[..., 4:5]
    targets_h = targets_3dbox[..., 5:6]
    targets_x1 = targets_3dbox[..., 0:1] - (targets_l / 2)
    targets_y1 = targets_3dbox[..., 1:2] - (targets_w / 2)
    targets_z1 = targets_3dbox[..., 2:3] - (targets_h / 2)
    targets_x2 = targets_x1 + targets_l
    targets_y2 = targets_y1 + targets_w
    targets_z2 = targets_z1 + targets_h

    intersection_l = (torch.min(preds_x2, targets_x2) - torch.max(preds_x1, targets_x1)).clip(min=0)
    intersection_w = (torch.min(preds_y2, targets_y2) - torch.max(preds_y1, targets_y1)).clip(min=0)
    intersection_h = (torch.min(preds_z2, targets_z2) - torch.max(preds_z1, targets_z1)).clip(min=0)

    intersection = intersection_l * intersection_w * intersection_h
    union = ((preds_l * preds_w * preds_h) + (targets_l * targets_w * targets_h)) - intersection
    iou = intersection / (union + eps)

    cl = (torch.max(preds_x2, targets_x2) - torch.min(preds_x1, targets_x1))
    cw = (torch.max(preds_y2, targets_y2) - torch.min(preds_y1, targets_y1))
    ch = (torch.max(preds_z2, targets_z2) - torch.min(preds_z1, targets_z1))
    c_sq = cl.pow(2) + cw.pow(2) + ch.pow(2) + eps
    v = (4 / (3 * torch.pi**2)) * (
        (torch.arctan(targets_l / targets_w) - torch.arctan(preds_l / preds_w)) +
        (torch.arctan(targets_w / targets_h) - torch.arctan(preds_w / preds_h)) +
        (torch.arctan(targets_h / targets_l) - torch.arctan(preds_h / preds_l))
    ).pow(2)
    d_sq = (
        (preds_3dbox[..., :1] - preds_3dbox[..., :1]).pow(2) + 
        (preds_3dbox[..., 1:2] - preds_3dbox[..., 1:2]).pow(2) + 
        (preds_3dbox[..., 2:3] - preds_3dbox[..., 2:3]).pow(2)
    )

    with torch.no_grad():
        a = v / (v - iou + (1 + eps))
        
    ciou = iou - ((d_sq / c_sq) + (a * v))
    return ciou.squeeze(-1)