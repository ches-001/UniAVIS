import torch
import torch.nn.functional as F
from .track_loss import TrackLoss
from typing import Tuple

class VectorMapLoss(TrackLoss):
    """
    This is an implementation of the loss function of the "VectorMapNet: End-to-end Vectorized HD Map Learning"
    paper as described in section 3.5 of the paper https://arxiv.org/pdf/2206.08920
    """
    def __init__(
            self,
            cls_lambda: float, 
            iou_lambda: float, 
            l1_lambda: float,
            polygen_lambda: float,
        ):
        super(VectorMapLoss, self).__init__(cls_lambda, iou_lambda, l1_lambda, angle_lambda=0)
        self.polygen_lambda = polygen_lambda
        del self.angle_lambda


    def forward(
            self, 
            pred_polylines: torch.Tensor, 
            pred_boxes: torch.Tensor,
            target_polylines: torch.Tensor, 
            target_boxes: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        pred_polylines: (N, num_elements, num_vertices, 2, num_grid)
            last axis corresponds to x, y

        :pred_boxes: (N, num_elements, 4 + num_classes)

        target_polylines: (N, num_elements, num_vertices, 2), expected to be long tensor (int64)
            last axis corresponds to x, y
                    
        :target_boxes: (N, num_elements, 5) (5 = 4 box data (x, y, w, h) and 1 class label)
        """
        assert target_polylines.dtype == torch.int64
        
        device = pred_polylines.device
        batch_size = pred_polylines.shape[0]
        num_elements = pred_polylines.shape[1]

        batch_indexes = torch.arange(batch_size, device=device)[:, None].tile(1, num_elements)

        pred_xywh = pred_boxes[:, :, 0:4]
        pred_cls_logits = pred_boxes[:, :, 4:]

        target_xywh = target_boxes[:, :, 0:4]
        target_cls = target_boxes[:, :, 4]

        pred_indexes, target_indexes, det_loss, polygen_loss_mask = self._forward_match(
            pred_xywh, target_xywh, pred_cls_logits, target_cls, batch_indexes, num_cls=pred_cls_logits.shape[-1]
        )

        pred_polylines = pred_polylines[batch_indexes, pred_indexes, ...][polygen_loss_mask]
        target_polylines = target_polylines[batch_indexes, target_indexes, ...][polygen_loss_mask]

        pred_polylines = pred_polylines.flatten(start_dim=0, end_dim=2)
        target_polylines = target_polylines.flatten(start_dim=0, end_dim=2)

        polygen_loss = self._cls_loss(pred_polylines, target_polylines, torch.arange(target_polylines.shape[0], device=device))
        polygen_loss = polygen_loss.mean()

        loss = det_loss + (self.polygen_lambda * polygen_loss)
        return loss, pred_indexes, target_indexes, polygen_loss_mask
    

    def _forward_match(
            self, 
            pred_xywh: torch.Tensor, 
            target_xywh: torch.Tensor, 
            pred_cls_logits: torch.Tensor, 
            target_cls: torch.Tensor,
            batch_indexes: torch.Tensor,
            num_cls: int
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # bg (background class must be the last class)
        bg_cls_idx = num_cls - 1
        pred_xywh = pred_xywh[:, :, None, :]
        target_xywh = target_xywh[:, None, :, :]

        ciou_cost_mat, l1_box_cost_mat = self._box_ciou_and_l1_loss(pred_xywh, target_xywh, is_3d=False)
        
        pred_cls_logits = pred_cls_logits[:, :, None, :]
        target_cls_proba = self._make_cls_match_targets(target_cls, batch_indexes, num_cls)
        target_cls_proba = target_cls_proba[:, None, :, :]
        cls_loss, cls_cost_mat = self._cls_loss_and_cls_match_cost(pred_cls_logits, target_cls_proba)

        costs_mat = (
            (self.cls_lambda * cls_cost_mat) +
            (self.iou_lambda * ciou_cost_mat) + 
            (self.l1_lambda * l1_box_cost_mat)
        )
        match_indexes = self._optimal_linear_assign(costs_mat)
        pred_indexes = match_indexes[:, :, 0]
        target_indexes = match_indexes[:, :, 1]

        target_cls = target_cls[batch_indexes, target_indexes]
        polygen_loss_mask = (target_cls != bg_cls_idx)

        # see the loss_mask in TrackLoss._first_frame_forward method (in track_loss.py) to 
        # understand this loss_mask
        det_loss_mask = torch.zeros_like(ciou_cost_mat, dtype=torch.bool)
        det_loss_mask[batch_indexes, pred_indexes, target_indexes] = 1

        cls_loss = cls_loss[det_loss_mask].mean()
        det_loss_mask = torch.masked_fill(det_loss_mask, (target_cls == bg_cls_idx)[:, None, :], 0)

        ciou_loss = ciou_cost_mat[det_loss_mask].mean()
        l1_box_loss = l1_box_cost_mat[det_loss_mask].mean()

        det_loss = (
            (self.cls_lambda * cls_loss) +
            (self.iou_lambda * ciou_loss) + 
            (self.l1_lambda * l1_box_loss) 
        )

        return pred_indexes, target_indexes, det_loss, polygen_loss_mask
    

class RasterMapLoss(VectorMapLoss):
    def __init__(self, *args, **kwargs):
        super(RasterMapLoss, self).__init__(*args, **kwargs)

    def forward(
            self, 
            protos: torch.Tensor, 
            pred_boxes: torch.Tensor,
            target_polylines: torch.Tensor,
            target_boxes: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        protos (N, k, H, W), where k is number of segmentation coefficients

        pred_boxes (N, num_elements, 4 + k + num_classes) (k = num segmentation proto coefficients)

        target_polylines: (N, num_elements, num_vertices, 2), expected to be long tensor (int64)
            last axis corresponds to x, y

        target_boxes: (N, num_elements, 5) (5 = 4 box data (x, y, w, h) and 1 class label)
        """
        assert target_polylines.dtype == torch.int64

        device = protos.device
        batch_size = pred_boxes.shape[0]
        num_elements = pred_boxes.shape[1]
        num_seg_coefs = protos.shape[1]

        batch_indexes = torch.arange(batch_size, device=device)[:, None].tile(1, num_elements)

        pred_xywh = pred_boxes[:, :, 0:4]
        pred_cls_logits = pred_boxes[:, :, 4 + num_seg_coefs:]

        target_xywh = target_boxes[:, :, 0:4]
        target_cls = target_boxes[:, :, 4]

        pred_indexes, target_indexes, det_loss, polygen_loss_mask = self._forward_match(
            pred_xywh, target_xywh, pred_cls_logits, target_cls, batch_indexes, num_cls=pred_cls_logits.shape[-1]
        )

        # here, we get the predicted segmentation coefficients and their corresponding target polylines.
        # then compute the predicted segmentation masks by matrix multipying the coefficients with the
        # protos map, then we create a target seg_mask to compare with the predicted masks and compute loss
        pred_seg_coefs = pred_boxes[:, :, 4:4 + num_seg_coefs]
        pred_seg_coefs =  pred_seg_coefs[batch_indexes, pred_indexes, :]
        
        pred_seg_masks = torch.einsum("nmk,nkhw->nmhw", pred_seg_coefs, protos)
        pred_seg_masks = pred_seg_masks[polygen_loss_mask]

        target_polylines = target_polylines[batch_indexes, target_indexes, ...][polygen_loss_mask]
        bidx = torch.arange(target_polylines.shape[0], device=device)[:, None].tile(1, target_polylines.shape[1])

        target_seg_masks = torch.zeros_like(pred_seg_masks)
        target_seg_masks[bidx, target_polylines[:, :, 1], target_polylines[:, :, 0]] = 1

        _, h, w = pred_seg_masks.shape
        pos_sum = target_seg_masks.sum(dim=(1, 2), keepdim=True)
        neg_sum = (h * w) - pos_sum
        eps = torch.finfo(pred_seg_masks.dtype).eps
        pos_weight = (neg_sum / (pos_sum + eps))

        polygen_loss = F.binary_cross_entropy_with_logits(pred_seg_masks, target_seg_masks, pos_weight=pos_weight, reduction="mean")
        loss = det_loss + (self.polygen_lambda * polygen_loss)
        return loss, pred_indexes, target_indexes, polygen_loss_mask
