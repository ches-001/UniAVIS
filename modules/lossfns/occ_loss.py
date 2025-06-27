import torch
import torch.nn as nn
import torch.nn.functional as F

class OccLoss(nn.Module):
    def __init__(self, bce_lambda: float, dice_lambda: float, main_lambda: float, aux_lambda: float, eps: float=1e-5):
        super(OccLoss, self).__init__()
        self.eps = eps
        self.bce_lambda = bce_lambda
        self.dice_lambda = dice_lambda
        self.main_lambda = main_lambda
        self.aux_lambda = aux_lambda

    def forward(self, occ_pred: torch.Tensor, attn_mask: torch.Tensor, occ_target: torch.Tensor) -> torch.Tensor:
        """
        occ_pred: (N, num_agents, T, H, W)

        attn_mask: (N, num_agents, T, H / s, W / s)

        occ_target: (N, num_agents, T, H, W)
        """
        main_loss = self._lossfn(occ_pred, occ_target)
        
        occ_target = torch.flatten(occ_target, start_dim=0, end_dim=1)
        occ_target = F.interpolate(occ_target, size=attn_mask.shape[-2:], mode="bilinear", align_corners=False)
        occ_target = torch.unflatten(occ_target, dim=0, sizes=attn_mask.shape[:2])
        occ_target = occ_target.round()
        
        attn_loss = self._lossfn(attn_mask, occ_target)

        return (self.main_lambda * main_loss) + (self.aux_lambda * attn_loss)


    def _lossfn(self, occ_pred: torch.Tensor, occ_target: torch.Tensor) -> torch.Tensor:
        valid_target_mask = (occ_target != -999).all(dim=(-3, -2, -1))

        h, w = occ_pred.shape[-2:]
        pos_sum = occ_target.sum(dim=(-1, -2), keepdim=True)
        neg_sum = (h * w) - pos_sum
        pos_weight = (neg_sum / pos_sum)

        bce_loss = F.binary_cross_entropy_with_logits(occ_pred, occ_target, reduction="none", pos_weight=pos_weight)
        bce_loss = bce_loss[valid_target_mask].mean()

        dice_loss = 1 - self._dice_score(occ_pred, occ_target)
        dice_loss = dice_loss[valid_target_mask].mean()

        loss = (self.bce_lambda * bce_loss) + (self.dice_lambda * dice_loss)
        return loss


    def _dice_score(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        a = 2 * (pred_mask * target_mask).sum(dim=(-2. -1))
        b = pred_mask.sum(dim=(-2. -1)) + target_mask.sum(dim=(-2. -1))
        dice_score = (a + self.eps) / (b + self.eps)
        return dice_score