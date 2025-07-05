import torch
import torch.nn as nn
from utils.img_utils import transform_points
from typing import Optional, Tuple


class MotionLoss(nn.Module):
    def __init__(self, cls_lambda: float, reg_lambda: float, as_gmm: bool=False, collapsed_gmm: bool=False):
        super(MotionLoss, self).__init__()
        """
        cls_lambda: mode classification weight, only applicable if as_gmm is True

        reg_lambda: regression weight, only applicable if as_gmm is True

        as_gmm: as Gaussian mixture model, if set to True, the k predicted modes are treated like a Gaussian mixture model.
            If set to False, the mode  whose mean has the least distance from the target is selected as the distribution to use
            when computing the log likelihood of the target belonging to that distribution

        collapsed_gmm: applicable only if as_gmm is True, this argument controls whether the GMM is collapsed before computing the
            likelihood or if the weighted sum of likelihood of each individual mode is computed.
        """
        self.cls_lambda = cls_lambda
        self.reg_lambda = reg_lambda
        self.as_gmm = as_gmm
        self.collapsed_gmm = collapsed_gmm

    
    def forward(
            self, 
            pred_motion_modes: torch.Tensor, 
            pred_mode_proba: torch.Tensor, 
            target_motion: torch.Tensor,
            ego_pred_motion_modes: Optional[torch.Tensor]=None,
            ego_pred_mode_proba: Optional[torch.Tensor]=None,
            ego_target_motion: Optional[torch.Tensor]=None,
            transform: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        """
        pred_motion_modes: (N, num_agents, k, T, 5), where k and T are number of modes and timesteps

        pred_mode_proba: (N, num_agents, k)

        target_motion: (N, num_agents, T, 2), this motion data is in scene (ego) level

        ego_pred_motion_modes: (N, k, T, 5), ego motion predictions

        ego_pred_mode_proba: (N, k), ego motion predictions

        ego_target_motion: (N, T, 2), ego target motion

        transform: (N, num_agents, 4, 4), transformation matrix from agent level to scene (ego) level
            if this is None, the function assumes that the multiagents_trajs is already projected to scene level.
        """
        if transform is not None:
            assert (
                transform.shape[-1] == 4 and
                transform.shape[-2] == transform.shape[-1]
            )

        if (
            ego_pred_motion_modes is not None
            and ego_pred_mode_proba is not None
            and ego_target_motion is not None
        ):
            pred_motion_modes = torch.concat([ego_pred_mode_proba[:, None], pred_motion_modes], dim=1)
            pred_mode_proba = torch.concat([ego_pred_mode_proba[:, None], pred_mode_proba], dim=1)
            target_motion = torch.concat([ego_target_motion[:, None], target_motion], axis=1)

            if transform is not None:
                ego_transform = torch.eye(transform.shape[-1])[None, None, :, :]
                ego_transform = ego_transform.tile(transform.shape[0], 1, 1, 1)
                transform = torch.concat([ego_transform, transform], axis=1)
        
        if self.as_gmm:
            if self.collapsed_gmm:
                return self._collapsed_mixture_forward(pred_motion_modes, pred_mode_proba, target_motion, transform)
            else:
                return self._uncollapsed_mixture_forward(pred_motion_modes, pred_mode_proba, target_motion, transform)
        return self._no_mixture_forward(pred_motion_modes, pred_mode_proba, target_motion, transform)


    def _collapsed_mixture_forward(
            self,
            pred_motion_modes: torch.Tensor, 
            pred_mode_proba: torch.Tensor, 
            target_motion: torch.Tensor,
            transform: Optional[torch.Tensor]=None
        ) -> torch.Tensor:

        pred_mode_proba = pred_mode_proba[..., None, None, None]
        sigma = torch.exp(pred_motion_modes[..., [2, 3]])
        rho = pred_motion_modes[..., 4]
        
        input = target_motion[..., None]
        mu = pred_motion_modes[..., [0, 1], None]
        
        covar = torch.stack([
            sigma[..., 0].pow(2),
            rho * sigma[..., 0] * sigma[..., 1],
            rho * sigma[..., 0] * sigma[..., 1],
            sigma[..., 1].pow(2)
        ], dim=-1)

        covar = torch.unflatten(covar, dim=-1, sizes=(2, 2))
        
        mixture_mu = (pred_mode_proba * mu).sum(dim=2)

        mu_div = mu - mixture_mu[:, :, None]

        mixture_covar = pred_mode_proba * (covar + (mu_div @ mu_div.transpose(-1, -2)))
        mixture_covar = mixture_covar.sum(dim=2)

        loss_mask = (target_motion != -999).all(dim=-1)

        if transform is not None:
            mixture_mu, mixture_covar = self._transform_mu_and_covar(mixture_mu, mixture_covar, transform)

        log_proba = self._compute_dist_log_proba(input, mixture_mu, mixture_covar)
        log_proba = log_proba[loss_mask]
        log_proba = (-log_proba).mean()
        return log_proba
    

    def _uncollapsed_mixture_forward(
            self,
            pred_motion_modes: torch.Tensor, 
            pred_mode_proba: torch.Tensor, 
            target_motion: torch.Tensor,
            transform: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        
        pred_mode_proba = pred_mode_proba[..., None]
        sigma = torch.exp(pred_motion_modes[..., [2, 3]])
        rho = pred_motion_modes[..., 4]

        input = target_motion[:, :, None, :, :, None]
        mu = pred_motion_modes[..., [0, 1], None]

        covar = torch.stack([
            sigma[..., 0].pow(2),
            rho * sigma[..., 0] * sigma[..., 1],
            rho * sigma[..., 0] * sigma[..., 1],
            sigma[..., 1].pow(2)
        ], dim=-1)

        covar = torch.unflatten(covar, dim=-1, sizes=(2, 2))

        loss_mask = (target_motion != -999).all(dim=-1)

        if transform is not None:
            mu, covar = self._transform_mu_and_covar(mu, covar, transform[:, :, None])

        log_pdf = self._compute_dist_log_proba(input, mu, covar)
        log_proba = torch.log(pred_mode_proba)

        loss = -torch.logsumexp(log_proba + log_pdf, dim=2)
        loss = loss[loss_mask].mean()
        return loss


    def _no_mixture_forward(
            self,
            pred_motion_modes: torch.Tensor, 
            pred_mode_proba: torch.Tensor, 
            target_motion: torch.Tensor,
            transform: Optional[torch.Tensor]
        ) -> torch.Tensor:
        
        num_modes = pred_motion_modes.shape[2]
        num_timesteps = pred_motion_modes.shape[3]
        pred_dim = pred_motion_modes.shape[4]

        # select the indexes of the mode whose mean values are of the least distance 
        # from the target trajectory

        valid_target_mask = (target_motion != -999).all(dim=-1)
        modes_mu = pred_motion_modes[..., [0, 1]]

        with torch.no_grad():
            dist = modes_mu - target_motion[:, :, None]
            dist = dist.pow(2).sum(dim=-1).sqrt()
            dist[~valid_target_mask[:, :, None, :].tile(1, 1, num_modes, 1)] = 0

            num_traj_points = valid_target_mask.sum(dim=-1)
            num_traj_points = num_traj_points[:, :, None]
            dist = dist.sum(dim=-1) / num_traj_points
            dist[torch.isnan(dist)] = torch.inf

            best_indexes = dist.argmin(dim=-1)
            best_indexes = best_indexes[:, :, None]

        best_mode_proba = torch.gather(pred_mode_proba, dim=2, index=best_indexes)[:, :, 0]

        best_indexes = best_indexes[..., None, None].tile(1, 1, 1, num_timesteps, pred_dim)
        pred_motion = torch.gather(pred_motion_modes, dim=2, index=best_indexes)[:, :, 0]
        
        input = target_motion[..., None]
        mu = pred_motion[..., [0, 1], None]
        sigma = torch.exp(pred_motion[..., [2, 3]])
        rho = pred_motion[..., 4]

        covar = torch.stack([
            sigma[..., 0].pow(2),
            rho * sigma[..., 0] * sigma[..., 1],
            rho * sigma[..., 0] * sigma[..., 1],
            sigma[..., 1].pow(2)
        ], dim=-1)
        covar = torch.unflatten(covar, dim=-1, sizes=(2, 2))

        motion_loss_mask = valid_target_mask
        mode_loss_mask = motion_loss_mask.any(dim=-1)
        
        if transform is not None:
            mu, covar = self._transform_mu_and_covar(mu, covar, transform)
            
        log_pdf = self._compute_dist_log_proba(input, mu, covar)
        log_pdf = log_pdf[motion_loss_mask]
        log_pdf = log_pdf.mean()

        log_proba = torch.log(best_mode_proba[mode_loss_mask]).mean()
        loss = -((self.cls_lambda * log_proba) + (self.reg_lambda * log_pdf))
        return loss
    

    def _transform_mu_and_covar(
            self, 
            mu: torch.Tensor, 
            covar: torch.Tensor, 
            transform: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        mu = transform_points(mu[..., 0], transform_matrix=transform)[..., None]
        
        R = transform[..., :2, :2]
        R_T = R.transpose(-1, -2)
        covar = R[..., None, :, :] @ covar @ R_T[..., None, :, :]
        return mu, covar


    def _compute_dist_log_proba(
            self, 
            input: torch.Tensor, 
            mu: torch.Tensor, 
            covar: torch.Tensor
        ) -> torch.Tensor:
        # d = input.shape[-2] and not d = input.shape[-1] because last dimension has a size of 1, 
        # so last two dimensions are of shape [d, 1]
        d = input.shape[-2]
        div = input - mu
        covar_inv = torch.linalg.inv(covar)
        covar_det = torch.linalg.det(covar)
        two_pi = torch.tensor(2 * torch.pi, device=input.device)
        mahal_dist = (div.transpose(-1, -2) @ covar_inv @ div)
        mahal_dist = mahal_dist[..., 0, 0]
        log_proba = -0.5 * (mahal_dist + torch.log(covar_det) + (d * torch.log(two_pi)))
        return log_proba