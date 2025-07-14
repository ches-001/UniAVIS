import torch
import torch.linalg
import torch.nn as nn
from utils.img_utils import transform_points
from typing import Optional, Tuple


class MotionLoss(nn.Module):
    def __init__(self, cls_lambda: float, reg_lambda: float, as_gmm: bool=False, collapsed_gmm: bool=False):
        super(MotionLoss, self).__init__()
        """
        cls_lambda: mode classification weight, only applicable if as_gmm is False

        reg_lambda: regression weight, only applicable if as_gmm is True

        as_gmm: as Gaussian mixture model, if set to True, the k predicted modes are treated like a Gaussian mixture model.
            If set to False, the mode  whose mean has the least distance from the target is selected as the distribution to use
            when computing the log likelihood of the target belonging to that distribution

        collapsed_gmm: applicable only if as_gmm is True, this argument controls whether the GMM is collapsed to a single 
            distribution parameterized with its composite first and second moments before computing the likelihood
            or if the weighted sum of likelihood of each individual mode is computed.
        """
        self.cls_lambda = cls_lambda
        self.reg_lambda = reg_lambda
        self.as_gmm = as_gmm
        self.collapsed_gmm = collapsed_gmm

    
    def forward(
            self, 
            pred_motions: torch.Tensor, 
            pred_mode_scores: torch.Tensor, 
            target_motions: torch.Tensor,
            ego_pred_motions: Optional[torch.Tensor]=None,
            ego_pred_mode_scores: Optional[torch.Tensor]=None,
            ego_target_motions: Optional[torch.Tensor]=None,
            transform: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        """
        pred_motions: (N, num_agents, k, T, 5), where k and T are number of modes and timesteps

        pred_mode_scores: (N, num_agents, k)

        target_motions: (N, num_agents, T, 2), this motion data is in scene (ego) level

        ego_pred_motions: (N, k, T, 5), ego motion predictions

        ego_pred_mode_scores: (N, k), ego motion predictions

        ego_target_motions: (N, T, 2), ego target motion

        transform: (N, num_agents, 4, 4), transformation matrix from agent level to scene (ego) level
            if this is None, the function assumes that the multiagents_trajs is already projected to scene level.
        """
        if transform is not None:
            assert (
                transform.shape[-1] == 4 and
                transform.shape[-2] == transform.shape[-1]
            )

        if (
            ego_pred_motions is not None
            and ego_pred_mode_scores is not None
            and ego_target_motions is not None
        ):
            pred_motions = torch.concat([ego_pred_mode_scores[:, None], pred_motions], dim=1)
            pred_mode_scores = torch.concat([ego_pred_mode_scores[:, None], pred_mode_scores], dim=1)
            target_motions = torch.concat([ego_target_motions[:, None], target_motions], axis=1)

            if transform is not None:
                ego_transform = torch.eye(transform.shape[-1])[None, None, :, :]
                ego_transform = ego_transform.tile(transform.shape[0], 1, 1, 1)
                transform = torch.concat([ego_transform, transform], axis=1)
        
        if self.as_gmm:
            if self.collapsed_gmm:
                return self._collapsed_mixture_forward(pred_motions, pred_mode_scores, target_motions, transform)
            else:
                return self._uncollapsed_mixture_forward(pred_motions, pred_mode_scores, target_motions, transform)
        return self._no_mixture_forward(pred_motions, pred_mode_scores, target_motions, transform)


    def _collapsed_mixture_forward(
            self,
            pred_motions: torch.Tensor, 
            pred_mode_scores: torch.Tensor, 
            target_motions: torch.Tensor,
            transform: Optional[torch.Tensor]=None
        ) -> torch.Tensor:

        pred_mode_scores = pred_mode_scores[..., None, None, None]
        sigma = torch.exp(pred_motions[..., [2, 3]])
        rho = pred_motions[..., 4]
        
        input = target_motions[..., None]
        mu = pred_motions[..., [0, 1], None]
        
        covar = torch.stack([
            sigma[..., 0].pow(2),
            rho * sigma[..., 0] * sigma[..., 1],
            rho * sigma[..., 0] * sigma[..., 1],
            sigma[..., 1].pow(2)
        ], dim=-1)

        covar = torch.unflatten(covar, dim=-1, sizes=(2, 2))
        
        mixture_mu = (pred_mode_scores * mu).sum(dim=2)

        mu_div = mu - mixture_mu[:, :, None]

        mixture_covar = pred_mode_scores * (covar + (mu_div @ mu_div.transpose(-1, -2)))
        mixture_covar = mixture_covar.sum(dim=2)

        loss_mask = (target_motions != -999).all(dim=-1)

        if transform is not None:
            mixture_mu, mixture_covar = self._transform_mu_and_covar(mixture_mu, mixture_covar, transform)

        log_proba = self._compute_dist_log_proba(input, mixture_mu, mixture_covar)
        log_proba = log_proba[loss_mask]
        log_proba = (-log_proba).mean()
        return log_proba
    

    def _uncollapsed_mixture_forward(
            self,
            pred_motions: torch.Tensor, 
            pred_mode_scores: torch.Tensor, 
            target_motions: torch.Tensor,
            transform: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        
        pred_mode_scores = pred_mode_scores[..., None]
        sigma = torch.exp(pred_motions[..., [2, 3]])
        rho = pred_motions[..., 4]

        input = target_motions[:, :, None, :, :, None]
        mu = pred_motions[..., [0, 1], None]

        covar = torch.stack([
            sigma[..., 0].pow(2),
            rho * sigma[..., 0] * sigma[..., 1],
            rho * sigma[..., 0] * sigma[..., 1],
            sigma[..., 1].pow(2)
        ], dim=-1)

        covar = torch.unflatten(covar, dim=-1, sizes=(2, 2))

        loss_mask = (target_motions != -999).all(dim=-1)

        if transform is not None:
            mu, covar = self._transform_mu_and_covar(mu, covar, transform[:, :, None])

        log_pdf = self._compute_dist_log_proba(input, mu, covar)
        log_proba = torch.log(pred_mode_scores)

        loss = -torch.logsumexp(log_proba + log_pdf, dim=2)
        loss = loss[loss_mask].mean()
        return loss


    def _no_mixture_forward(
            self,
            pred_motions: torch.Tensor, 
            pred_mode_scores: torch.Tensor, 
            target_motions: torch.Tensor,
            transform: Optional[torch.Tensor]
        ) -> torch.Tensor:
        
        num_modes = pred_motions.shape[2]
        num_timesteps = pred_motions.shape[3]
        pred_dim = pred_motions.shape[4]

        # select the indexes of the mode whose mean values are of the least distance 
        # from the target trajectory

        valid_target_mask = (target_motions != -999).all(dim=-1)
        modes_mu = pred_motions[..., [0, 1]]

        with torch.no_grad():
            dist = modes_mu - target_motions[:, :, None]
            dist = dist.pow(2).sum(dim=-1).sqrt()
            dist[~valid_target_mask[:, :, None, :].tile(1, 1, num_modes, 1)] = 0

            num_traj_points = valid_target_mask.sum(dim=-1)
            num_traj_points = num_traj_points[:, :, None]
            dist = dist.sum(dim=-1) / num_traj_points
            dist[torch.isnan(dist)] = torch.inf

            best_indexes = dist.argmin(dim=-1)
            best_indexes = best_indexes[:, :, None]

        best_mode_proba = torch.gather(pred_mode_scores, dim=2, index=best_indexes)[:, :, 0]

        best_indexes = best_indexes[..., None, None].tile(1, 1, 1, num_timesteps, pred_dim)
        pred_motion = torch.gather(pred_motions, dim=2, index=best_indexes)[:, :, 0]
        
        input = target_motions[..., None]
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
    

    @staticmethod
    def multishooting_solver(
        motions: torch.Tensor, 
        pred_pos: torch.Tensor, 
        t_splits: int=3, 
        dt: float=0.5,
        lambda_xy: float=0.01,
        lambda_goal: float=0.05,
        lambda_kinematics: float=0.1,
        iter_steps: int=10,
        lr=1e-3,
        verbose: bool=False,
        **kwargs
    ) -> torch.Tensor:
        """
        motions: (N, num_agents, T, 6), [x, y, x', y', x'', y'']
            The extra point in the trajectory corresponds to the target positions, velocities and accelerations
            at t = 0.

        initial_pos: (N, num_agents, 2), [x, y], predicted (imprecise) positions

        t_splits: number of segments to split the time horizon

        dt: time between two consecutive points
        """
        h_motions = torch.concat([motions, torch.ones_like(motions[..., [0]])], dim=-1)
        X = h_motions[..., :-1, :]
        X_T = X.transpose(-1, -2)
        Y = h_motions[..., 1:, :2]

        M = torch.linalg.inv(X_T @ X) @ (X_T @ Y)
        ERRS = (X @ M) - Y
        M_ERR = torch.linalg.inv(X_T @ X) @ (X_T @ ERRS)

        dynamics = lambda x : (x @ M) - (x @ M_ERR)

        t_horizon = h_motions.shape[2] // t_splits
        h_motions[..., 0, :2] = pred_pos.detach()

        # TODO: include other kinematic factors curvature and curvature rate and jerk. Factor them both
        # here and in the cost function
        initial_pos = nn.Parameter(h_motions[..., t_horizon::t_horizon, None, :2].clone(), requires_grad=True)
        initial_vel = nn.Parameter(h_motions[..., t_horizon::t_horizon, None, 2:4].clone(), requires_grad=True)
        initial_accel = nn.Parameter(h_motions[..., t_horizon::t_horizon, None, 4:6].clone(), requires_grad=True)

        optimizer = torch.optim.Adam([initial_pos, initial_vel, initial_accel], lr=lr, **kwargs)

        best_shots = None
        best_step = None
        best_cost = torch.inf

        for step in range(0, iter_steps):
            num_points = h_motions.shape[-2]
            shots = torch.concat([h_motions[..., [0], None, :2], initial_pos], dim=-3)
            shots_vel = torch.concat([h_motions[..., [0], None, 2:4], initial_vel], dim=-3)
            shots_accel = torch.concat([h_motions[..., [0], None, 4:6], initial_accel], dim=-3)
            ones = h_motions[..., ::t_horizon, None, [6]]

            for i in range(1, t_horizon+1):
                pos_xy_t_minus_1 = shots[..., [i-1], :]
                vel_xy_t_minus_1 = shots_vel[..., [i-1], :]
                accel_xy_t_minus_1 = shots_accel[..., [i-1], :]

                input = torch.concat([pos_xy_t_minus_1, vel_xy_t_minus_1, accel_xy_t_minus_1, ones], dim=-1)

                current_shot = dynamics(input)
                current_vel = (current_shot - pos_xy_t_minus_1) / dt
                current_accel = (current_vel - vel_xy_t_minus_1) / dt

                shots = torch.concat([shots, current_shot], dim=-2)
                shots_vel = torch.concat([shots_vel, current_vel], dim=-2)
                shots_accel = torch.concat([shots_accel, current_accel], dim=-2)

            shots = torch.concat([shots[..., 0, [0], :], shots[..., 1:, :].flatten(-3, -2)], dim=-2)
            shots_vel = torch.concat([shots_vel[..., 0, [0], :], shots_vel[..., 1:, :].flatten(-3, -2)], dim=-2)
            shots_accel = torch.concat([shots_accel[..., 0, [0], :], shots_accel[..., 1:, :].flatten(-3, -2)], dim=-2)

            t_indexes = list(range(t_horizon, num_points))[::t_horizon]
            last_t = num_points - 1
            if last_t not in t_indexes:
                t_indexes.append(last_t)

            shots = shots[..., :num_points, :]
            shots_vel = shots_vel[..., :num_points, :]
            shots_accel = shots_accel[..., :num_points, :]

            targets = h_motions[..., :2]
            targets_vel = h_motions[..., 2:4]
            targets_accel = h_motions[..., 4:6]

            pos_cost = lambda_xy * nn.functional.mse_loss(shots, targets, reduction="mean")
            goal_cost = lambda_goal * nn.functional.mse_loss(shots[..., t_indexes, :], targets[..., t_indexes, :], reduction="mean")
            vel_cost = nn.functional.mse_loss(shots_vel[..., t_indexes, :], targets_vel[..., t_indexes, :], reduction="mean")
            accel_cost = nn.functional.mse_loss(shots_accel[..., t_indexes, :], targets_accel[..., t_indexes, :], reduction="mean")

            cost = pos_cost + goal_cost + (lambda_kinematics * (vel_cost + accel_cost))
            cost.backward()
            optimizer.step()

            if cost < best_cost:
                best_cost = cost.item()
                best_shots = shots.detach()
                best_step = step

            if verbose:
                print(f"multishooting iter step: {step}, cost: {cost :.4f}")
        
        if verbose:
            print(f"\nbest iter step: {best_step}, best cost: {best_cost :.4f}")

        return best_shots