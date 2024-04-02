import sys
sys.path.append("/home/meribejayson/Desktop/Projects/SharkCNN/training_models/YOLO3D/utils")

import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class YOLO_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    """
        Example box output: torch.Size([1, 5, 68, 120]) 
        Example class prob output: torch.Size([1, 1, 68, 120])
    """
    def forward(self, xbox, xbox_hat, xclass, xclass_hat, n_pos):
        xbox = xbox.view(-1, 5)
        xbox_hat = xbox_hat.view(-1, 5)

        xclass = xclass.flatten()
        xclass_hat = xclass_hat.flatten()
        
        bbox_loss = bbox(pred_bboxes=xbox, target_bboxes=xbox_hat, target_scores=xclass, target_scores_sum=n_pos, fg_mask=(xclass > 0.5) )
        class_prob_loss = self.bce_loss(xclass, xclass_hat)

        return (bbox_loss, class_prob_loss)


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin
    

def probiou(obb1, obb2, CIoU=True, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor): A tensor of shape (N, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, ) representing obb similarities.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou

def bbox(pred_bboxes, target_bboxes, target_scores, target_scores_sum, fg_mask):
    """IoU loss."""
    iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])

    return ((1.0 - iou) * target_scores).sum() / target_scores_sum