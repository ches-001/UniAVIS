import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from utils.metric_utils import compute_2d_ciou, compute_3d_ciou
from typing import Tuple, Optional

class TrackLoss(nn.Module):
    """
    This is an implementation of the loss function of the TrackFormer paper as described in section 3.3
    of the paper https://arxiv.org/pdf/2101.02702
    """
    def __init__(
            self, 
            cls_lambda: float, 
            iou_lambda: float, 
            l1_lambda: float, 
            angle_lambda: float, 
        ):
        super(TrackLoss, self).__init__()

        self.cls_lambda = cls_lambda
        self.iou_lambda = iou_lambda
        self.l1_lambda = l1_lambda
        self.angle_lambda = angle_lambda

    def forward(
            self, 
            preds: torch.Tensor, 
            targets: torch.Tensor, 
            prev_track_ids: Optional[torch.Tensor]=None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        preds:          (N, num_dets, d)
        targets:        (N, num_dets, d)
        prev_track_ids: (N, num_dets)
        """
        if prev_track_ids is None:
            return self._first_frame_forward(preds, targets)
        return self._next_frame_forward(preds, targets, prev_track_ids)
        

    def _first_frame_forward(
            self, 
            preds: torch.Tensor, 
            targets: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        device = preds.device

        assert preds.shape[:2] == targets.shape[:2]

        if targets.shape[-1] == 5:
            ciou_cost_mat, l1_box_cost_mat = self._box_ciou_and_l1_loss(
                preds[:, :, None, 1:5], targets[:, None, :, 1:-2], is_3d=False
            )
            pred_angle_idx, pred_cls_start_idx = 4, 5
            target_angle_idx, target_cls_idx = 5, 6

        elif targets.shape[-1] == 8:
            ciou_cost_mat, l1_box_cost_mat = self._box_ciou_and_l1_loss(
                preds[:, :, None, 1:7], targets[:, None, :, 1:-2], is_3d=True
            )
            pred_angle_idx, pred_cls_start_idx = 6, 7
            target_angle_idx, target_cls_idx = 7, 8

        else:
            raise ValueError

        batch_indexes = torch.arange(targets.shape[0], device=device)[:, None].tile(1, targets.shape[1])

        angle_cost_mat = self._angle_loss(preds[:, :, None, pred_angle_idx], targets[:, None, :, target_angle_idx])

        pred_cls_logits = preds[:, :, None, pred_cls_start_idx:]
        target_cls = targets[:, :, target_cls_idx]
        target_cls_proba = self._make_cls_match_targets(target_cls, batch_indexes, num_cls=preds.shape[-1])
        target_cls_proba = target_cls_proba[:, None, :, :]
        cls_loss, cls_cost_mat = self._cls_loss_and_cls_match_cost(pred_cls_logits, target_cls_proba)

        costs_mat = (
            (self.cls_lambda * cls_cost_mat) +
            (self.iou_lambda * ciou_cost_mat) + 
            (self.l1_lambda * l1_box_cost_mat) + 
            (self.angle_lambda * angle_cost_mat)
        )

        # row indexes of the match indexes, which corresponds to prediction indexes are already sorted
        match_indexes = self._optimal_linear_assign(costs_mat)
        pred_indexes = match_indexes[:, :, 0]
        target_indexes = match_indexes[:, :, 1]

        track_ids = targets[batch_indexes, target_indexes, 0]

        pred_track_mask = track_ids.clone()
        pred_track_mask[pred_track_mask != -999] = 1
        pred_track_mask[pred_track_mask == -999] = 0
        pred_track_mask = pred_track_mask.bool()

        # The loss mask is a num_obj x num_obj matrix where the only matched pred and target indexes are
        # set, to compute loss for only successful matches
        loss_mask = torch.zeros_like(costs_mat, dtype=torch.bool)
        loss_mask[batch_indexes, pred_indexes, target_indexes] = 1
        
        cls_loss = cls_loss[loss_mask].mean()

        # After computing the cls_loss, the loss_mask is further edited to ensure that indexes that match
        # a prediction to a pading target is excluded from the other loss computation 
        loss_mask[torch.ones_like(track_ids)[:, :, None] & (track_ids == -999)[:, None, :]] = 0

        ciou_loss = ciou_cost_mat[loss_mask].mean()
        l1_box_loss = l1_box_cost_mat[loss_mask].mean()
        angle_loss = angle_cost_mat[loss_mask].mean()

        loss = (
            (self.cls_lambda * cls_loss) +
            (self.iou_lambda * ciou_loss) + 
            (self.l1_lambda * l1_box_loss) + 
            (self.angle_lambda * angle_loss)
        )

        return loss, pred_indexes, target_indexes, track_ids, pred_track_mask
    

    def _next_frame_forward(
            self, 
            preds: torch.Tensor, 
            targets: torch.Tensor, 
            prev_track_ids: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        

        assert preds.shape[:2] == targets.shape[:2]
        
        if targets.shape[-1] == 5:
            is_3d = False
            pred_box_start, pred_box_end = 0, 4
            target_box_start, target_box_end = 1, -2
            pred_angle_idx, pred_cls_start_idx = 4, 5
            target_angle_idx, target_cls_idx = 5, 6

        elif targets.shape[-1] == 8:
            is_3d = True
            pred_box_start, pred_box_end = 0, 6
            target_box_start, target_box_end = 1, -2
            pred_angle_idx, pred_cls_start_idx = 6, 7
            target_angle_idx, target_cls_idx = 7, 8

        else:
            raise ValueError
        
        device = preds.device
        num_cls = preds.shape[-1] - pred_cls_start_idx
        bg_cls_idx = num_cls - 1

        track_ids = targets[:, :, 0]
        prev_track_ids = self._reorder_prev_track_ids(prev_track_ids)
        output_track_ids = torch.zeros_like(track_ids)
        output_track_ids.fill_(-999)

        # track ids from previous frame present in current frame and vise versa
        match_mask = (prev_track_ids[:, :, None] == track_ids[:, None, :]) & (prev_track_ids[:, :, None] != -999)
        match_pred_mask = match_mask.any(dim=-1)

        # track is from previous frame not present in current frame (out of frame or occluded)
        # and ones from current frame not present in previous frame (new detections)
        nomatch_mask = (prev_track_ids[:, :, None] != track_ids[:, None, :])
        
        flat_pred_idx = None
        flat_target_idx = None

        if torch.any(match_mask):
            b_idx, pred_idx, target_idx = torch.where(match_mask)

            output_track_ids[match_pred_mask] = prev_track_ids[match_pred_mask]

            # flattened indexes when pred and targets are flattened from (N, num_dets, d) to (N * num_dets, d)
            flat_pred_idx = (b_idx * match_mask.shape[1]) + pred_idx
            flat_target_idx = (b_idx * match_mask.shape[1]) + target_idx

            mp_preds = preds.flatten(start_dim=0, end_dim=1)[flat_pred_idx]
            mp_targets = targets.flatten(start_dim=0, end_dim=1)[flat_target_idx]

            mp_angle_loss = self._angle_loss(mp_preds[..., pred_angle_idx], mp_targets[..., target_angle_idx])
            mp_ciou_loss, mp_l1_box_loss = self._box_ciou_and_l1_loss(
                mp_preds[..., pred_box_start:pred_box_end], mp_targets[..., target_box_start:target_box_end], is_3d
            )
            mp_pred_cls = mp_preds[..., pred_cls_start_idx:]
            mp_target_cls = mp_targets[..., target_cls_idx].long()
            mp_cls_loss = self._cls_loss(mp_pred_cls, mp_target_cls, torch.arange(mp_target_cls.shape[0], device=device))

            mp_loss = (
                (self.cls_lambda * mp_cls_loss) +
                (self.iou_lambda * mp_ciou_loss) + 
                (self.l1_lambda * mp_l1_box_loss) + 
                (self.angle_lambda * mp_angle_loss)
            )
            loss = loss + mp_loss.mean()

        # previous tracks not matched to any current target tracks due to out-of-frame or occlusion
        disappeared_mask = nomatch_mask.all(dim=-1)
        disappeared_preds = preds[disappeared_mask]

        if torch.any(disappeared_mask):
            bg_targets = torch.zeros(*disappeared_preds.shape[0], device=device, dtype=torch.int64)
            bg_targets.fill_(bg_cls_idx)
            bg_loss = self._cls_loss(disappeared_preds, bg_targets, torch.arange(bg_targets.shape[0], device=device))
            loss = loss + bg_loss.mean()

        # current target tracks not matched to any previous track (new detection)
        nomatch_target_mask = nomatch_mask.all(dim=-2)
        nomatch_pred_mask = (~match_pred_mask) & (~disappeared_mask)

        if torch.any(nomatch_target_mask):
            new_preds = torch.zeros_like(preds)
            new_targets = torch.zeros_like(targets)
            new_targets.fill_(-999)
            new_preds[nomatch_pred_mask] = preds[nomatch_pred_mask]
            new_targets[nomatch_target_mask] = targets[nomatch_target_mask]
            (
                new_loss, pred_indexes, target_indexes, new_track_ids, new_track_mask
            ) = self._first_frame_forward(new_preds, new_targets)

            output_track_ids[new_track_mask] = new_track_ids[new_track_mask]
            loss = loss + new_loss.mean()

        # set the matching pred and target indexes
        if flat_pred_idx is not None:
            pred_indexes = pred_indexes.flatten(start_dim=0, end_dim=1)
            target_indexes = target_indexes.flatten(start_dim=0, end_dim=1)

            # FYI, the next two lines are correct. pred_indexes is sorted and target_indexes
            # is ordered according to how pred_indexes is sorted
            pred_indexes[flat_pred_idx] = flat_pred_idx % preds.shape[1]
            target_indexes[flat_pred_idx] = flat_target_idx % targets.shape[1]

            pred_indexes = pred_indexes.unflatten(dim=0, sizes=preds.shape)
            target_indexes = target_indexes.unflatten(dim=0, sizes=targets.shape)

        pred_track_mask = output_track_ids.clone()
        pred_track_mask[pred_track_mask != -999] = 1
        pred_track_mask[pred_track_mask == -999] = 0
        pred_track_mask = pred_track_mask.bool()

        return loss, pred_indexes, target_indexes, output_track_ids, pred_track_mask


    def _optimal_linear_assign(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        match_indexes = []
        for i in range(0, cost_matrix.shape[0]):
            row_i, col_i = linear_sum_assignment(cost_matrix[i].detach().cpu(), maximize=False)
            indexes = np.stack([row_i, col_i], axis=1)
            match_indexes.append(indexes)

        match_indexes = np.stack(match_indexes, axis=0)
        match_indexes = torch.from_numpy(match_indexes).to(cost_matrix.device)
        return match_indexes


    def _box_ciou_and_l1_loss(
            self, 
            preds: torch.Tensor, 
            targets: torch.Tensor, 
            is_3d: bool=True
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        if is_3d:
            ciou_loss = 1- compute_2d_ciou(preds, targets)
        else:
            ciou_loss = 1 - compute_3d_ciou(preds, targets)
        # the l1_los function might cause a warning, when predictions and targets of different shapes,
        # specifically when computing the matching cost matrix with prediction and target shapes of:
        # [N, N_pred, 1] and [N, 1, N_target] respectively. It is absolutely in this situation.
        l1_loss = F.l1_loss(preds, targets, reduction="none")
        return ciou_loss, l1_loss


    def _angle_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = (1 - torch.cos(preds - targets)) / 2
        return losses
    
    def _make_cls_match_targets(
            self, 
            targets: torch.Tensor, 
            batch_indexes: torch.Tensor,
            num_cls: int
        ) -> torch.Tensor:

        device = targets.device
        target_cls = torch.zeros(targets.shape[0], targets.shape[1], num_cls, dtype=torch.int64, device=device)
        det_indexes = torch.arange(targets.shape[1], device=device)[None, :].tile(targets.shape[0], 1)
        cls_indexes = targets.long()
        target_cls[batch_indexes, det_indexes, cls_indexes] = 1
        return target_cls
    
    def _cls_loss_and_cls_match_cost(
            self, 
            pred_logits: torch.Tensor, 
            targets: torch.Tensor, 
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        log_pred_proba = F.log_softmax(pred_logits, dim=-1)
        cls_loss = -(targets * log_pred_proba)
        cls_loss = cls_loss.sum(dim=-1)

        pred_probs = F.softmax(pred_logits, dim=-1)
        cls_cost_mat = -(targets * pred_probs)
        cls_cost_mat = cls_cost_mat.sum(dim=-1)
        return cls_loss, cls_cost_mat

    
    def _cls_loss(
            self, 
            pred_logits: torch.Tensor, 
            targets: torch.Tensor,
            batch_indexes: torch.Tensor
        ) -> torch.Tensor:

        log_pred_proba = F.log_softmax(pred_logits, dim=-1)
        log_pred_proba = log_pred_proba[batch_indexes, targets]
        cls_loss = -log_pred_proba
        return cls_loss
    

    def _reorder_prev_track_ids(self, prev_track_ids: torch.Tensor) -> torch.Tensor:
        # reorder prev_track_ids so that the valid tracks come first and the pad or background ones come later
        # see the first for loop of the TrackFormer.forward method code in trackformer.py to understand why.
        device = prev_track_ids.device
        mask = (prev_track_ids != -999)
        _, sorted_indexes = mask.sort(dim=1, descending=True)
        batch_indexes = torch.arange(mask.shape[0], device=device)[:, None].tile(1, mask.shape[1])
        prev_track_ids = prev_track_ids[batch_indexes, sorted_indexes]
        return prev_track_ids