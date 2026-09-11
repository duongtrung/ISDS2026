"""Region similarity J (IoU) and boundary F-measure (maxpool-based, px tolerance)."""
import torch
import torch.nn.functional as F


def iou(pred, gt, thresh=0.5, eps=1e-6):
    p = (pred > thresh).float()
    g = (gt > 0.5).float()
    inter = (p * g).sum(dim=(1, 2, 3))
    union = ((p + g) > 0).float().sum(dim=(1, 2, 3))
    return (inter + eps) / (union + eps)


def _boundary(mask, k=3):
    er = -F.max_pool2d(-mask, k, stride=1, padding=k // 2)
    return (mask - er).clamp(0, 1)


def boundary_f(pred, gt, thresh=0.5, tol=2, eps=1e-6):
    p = (pred > thresh).float()
    g = (gt > 0.5).float()
    pb, gb = _boundary(p), _boundary(g)
    kd = 2 * tol + 1
    gd = F.max_pool2d(gb, kd, stride=1, padding=tol)
    pd = F.max_pool2d(pb, kd, stride=1, padding=tol)
    prec = (pb * gd).sum(dim=(1, 2, 3)) / (pb.sum(dim=(1, 2, 3)) + eps)
    rec = (gb * pd).sum(dim=(1, 2, 3)) / (gb.sum(dim=(1, 2, 3)) + eps)
    return 2 * prec * rec / (prec + rec + eps)
