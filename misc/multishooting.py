import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

def compute_curvature(traj: torch.Tensor):
    """
    Input
    ------------------------------------------
    :traj: (..., T, 2)

    Formula
    ------------------------------------------
    curvature: k = 2 * area / (norm(a) * norm(b) * norm(c))
    area: 1 / 2 * ((b_x - a_x) * (c_y - a_y) - (c_x - a_x) * (b_y - a_y))

    Return
    ------------------------------------------
    :curvature: (..., T)
    """
    p_prev = traj[..., :-2, :]
    p_curr = traj[..., 1:-1, :]
    p_next = traj[..., 2:, :]

    a = p_prev - p_curr
    b = p_next - p_curr
    c = p_next - p_prev

    eps = torch.finfo(traj.dtype).eps

    numerator = torch.abs(
        (p_curr[..., 0] - p_prev[..., 0]) * (p_next[..., 1] - p_prev[..., 1]) -
        (p_curr[..., 1] - p_prev[..., 1]) * (p_next[..., 0] - p_prev[..., 0])
    )
    denominator = (torch.norm(a, dim=-1) * torch.norm(b, dim=-1) * torch.norm(c, dim=-1)) + eps

    curvature = numerator / denominator
    return curvature


def multishooting_solver(
        motions: torch.Tensor, 
        pred_pos: torch.Tensor, 
        t_segments: int, 
        dt: float=0.5,
        lambda_xy: float=0.05,
        lambda_goal: float=0.3,
        lambda_kinematics: float=0.3,
        lambda_smoothness: float=3.0,
        lambda_curvature: float=0.05,
        alpha: float=0.8,
        iter_steps: int=20,
        lr=0.0001,
        xy_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]=None,
        verbose: bool=False,
        pad_mask: Optional[torch.Tensor]=None,
        **kwargs
    ) -> torch.Tensor:
        """
        Input
        ------------------------------------------
        :motions: (N, num_agents, T+1, 6), [x, y, x', y', x'', y'']
            The extra timestep point in the trajectory corresponds to the target positions, velocities and 
            accelerations at t = 0.

        :pred_pos: (N, num_agents, 2), [x, y], predicted (imprecise) positions

        :t_segments: number of segments to split the time horizon

        :dt: time between two consecutive points

        :lambda_xy: weight value for MSE (Mean Squared Error) between estimated and target x, y positions

        :lambda_goal: weight value for MSE (Mean Squared Error) between estimated and target positions at 
            last timestep of each segment

        :lambda_kinematics: weight value for MSE (Mean Squared Error) between estimated and target 
            velocity and acceleration at last timestep of each segment
        
        :lambda_curvature: weight value for MSE (Mean Squared Error) between estimated abd target 
            curvature and curvature rates

        :alpha: weight contribution of simulated trajectory to initial guess, consequentially the 
            contribution of target trajectory to initial guess is 1 - alpha

        :iter_steps: Number of optimization steps

        :lr: Learning rate

        :xy_range: range of x, y values used for scaling

        :verbose: if True, log training steps

        :pad_mask: (N, num_agents, T+1) padding mask, for parts to of the motions tensor to ignore, 
            1 corresponds to values to not ignore 0 for values to ignore

        :kwargs: Adam optimizer key word arguments

        Return
        ------------------------------------------
        :shots: (N, num_agents, T+1, 2), aligned motion trajectory
        """
        assert t_segments > 0 and t_segments <= motions.shape[-2]
        assert alpha >= 0 and alpha <= 1

        if pad_mask is not None:
            assert pad_mask.dtype == torch.bool
            assert pad_mask.shape == motions.shape[:-1]

        xy_range = xy_range or ((-51.2, 51.2), (-51.2, 51.2))
        xy_range = torch.tensor(xy_range, dtype=motions.dtype, device=motions.device)
        xyva_range = xy_range.tile(3, 1)

        motions = 2 * ((motions - xyva_range[:, 0]) / (xyva_range[:, 1] - xyva_range[:, 0])) - 1
        pred_pos = 2 * ((pred_pos - xy_range[:, 0]) / (xy_range[:, 1] - xy_range[:, 0])) - 1

        h_motions = torch.concat([motions, torch.ones_like(motions[..., [0]])], dim=-1)
        X = h_motions[..., :-1, :]
        X_T = X.transpose(-1, -2)
        Y = h_motions[..., 1:, :2]

        M = torch.linalg.inv(X_T @ X) @ (X_T @ Y)
        ERRS = (X @ M) - Y
        M_ERR = torch.linalg.inv(X_T @ X) @ (X_T @ ERRS)

        dynamics = lambda x : (x @ M) - (x @ M_ERR)

        t_horizon = h_motions.shape[-2] // t_segments

        # replace current position in targets motion with current predicted position from upstream model
        # (TrackFormer) while sustaining kinematic factors (velocity and acceleration) as if it were the
        # target motion.
        h_motions[..., 0, :2] = pred_pos.detach()

        sim_pos = h_motions[..., None, [0], :2]
        sim_vel = h_motions[..., None, [0], 2:4]
        sim_accel = h_motions[..., None, [0], 4:6]
        ones = torch.ones_like(sim_accel[..., [0]])

        for _ in range(1, h_motions.shape[-2]):
            pos_t_minus_1 = sim_pos[..., [-1], :]
            vel_t_minus_1 = sim_vel[..., [-1], :]
            accel_t_minus_1 = sim_accel[..., [-1], :]

            input = torch.concat([
                pos_t_minus_1, 
                vel_t_minus_1, 
                accel_t_minus_1, 
                ones
            ], dim=-1)

            pos = dynamics(input)
            vel = (pos - pos_t_minus_1) / dt
            accel = (vel - vel_t_minus_1) / dt

            sim_pos = torch.concat([sim_pos, pos], dim=-2)
            sim_vel = torch.concat([sim_vel, vel], dim=-2)
            sim_accel = torch.concat([sim_accel, accel], dim=-2)

        sim_pos = sim_pos[..., 0, t_horizon::t_horizon, None, :]
        sim_vel = sim_vel[..., 0, t_horizon::t_horizon, None, :]
        sim_accel = sim_accel[..., 0, t_horizon::t_horizon, None, :]

        tgt_pos = h_motions[..., t_horizon::t_horizon, None, :2].clone()
        tgt_vel = h_motions[..., t_horizon::t_horizon, None, 2:4].clone()
        tgt_accel = h_motions[..., t_horizon::t_horizon, None, 4:6].clone()

        initial_pos = nn.Parameter((alpha * sim_pos) + ((1 - alpha) * tgt_pos), requires_grad=True)
        initial_vel = nn.Parameter((alpha * sim_vel) + ((1 - alpha) * tgt_vel), requires_grad=True)
        initial_accel = nn.Parameter((alpha * sim_accel) + ((1 - alpha) * tgt_accel), requires_grad=True)

        optimizer = torch.optim.Adam([initial_pos, initial_vel, initial_accel], lr=lr, **kwargs)

        best_shots = None
        best_step = None
        best_cost = torch.inf
        num_points = h_motions.shape[-2]
        t_indexes = list(range(t_horizon, num_points))[::t_horizon]

        if pad_mask is not None:
            pos_mask = ~pad_mask
        else:
            pos_mask = torch.zeros_like(motions[..., 0], dtype=torch.bool)
            
        goal_mask = pos_mask[..., t_indexes]

        for step in range(0, iter_steps):
            shots = torch.concat([h_motions[..., [0], None, :2], initial_pos], dim=-3)
            shots_vel = torch.concat([h_motions[..., [0], None, 2:4], initial_vel], dim=-3)
            shots_accel = torch.concat([h_motions[..., [0], None, 4:6], initial_accel], dim=-3)
            ones = h_motions[..., ::t_horizon, None, [6]]

            for i in range(1, t_horizon+1):
                pos_t_minus_1 = shots[..., [i-1], :]
                vel_t_minus_1 = shots_vel[..., [i-1], :]
                accel_t_minus_1 = shots_accel[..., [i-1], :]

                input = torch.concat([pos_t_minus_1, vel_t_minus_1, accel_t_minus_1, ones], dim=-1)

                current_shot = dynamics(input)
                current_vel = (current_shot - pos_t_minus_1) / dt
                current_accel = (current_vel - vel_t_minus_1) / dt

                shots = torch.concat([shots, current_shot], dim=-2)
                shots_vel = torch.concat([shots_vel, current_vel], dim=-2)
                shots_accel = torch.concat([shots_accel, current_accel], dim=-2)

            shots = torch.concat([shots[..., 0, [0], :], shots[..., 1:, :].flatten(-3, -2)], dim=-2)
            shots_vel = torch.concat([shots_vel[..., 0, [0], :], shots_vel[..., 1:, :].flatten(-3, -2)], dim=-2)
            shots_accel = torch.concat([shots_accel[..., 0, [0], :], shots_accel[..., 1:, :].flatten(-3, -2)], dim=-2)

            last_t = num_points - 1
            if last_t not in t_indexes:
                t_indexes.append(last_t)

            shots = shots[..., :num_points, :]
            shots_vel = shots_vel[..., :num_points, :]
            shots_accel = shots_accel[..., :num_points, :]
            shots_curvature = compute_curvature(shots)
            shots_curvature_rate = (shots_curvature[..., 1:] - shots_curvature[..., :-1]) / dt

            targets = h_motions[..., :2]
            targets_vel = h_motions[..., 2:4]
            targets_accel = h_motions[..., 4:6]
            targets_curvature = compute_curvature(targets)
            targets_curvature_rate = (targets_curvature[..., 1:] - targets_curvature[..., :-1]) / dt
        
            pos_cost = F.mse_loss(shots, targets, reduction="none")
            pos_cost = torch.masked_fill(pos_cost, pos_mask[..., None], value=0.0)
            pos_cost = lambda_xy * pos_cost.mean()

            goal_cost = F.mse_loss(shots[..., t_indexes, :], targets[..., t_indexes, :], reduction="none")
            vel_cost = F.mse_loss(shots_vel[..., t_indexes, :], targets_vel[..., t_indexes, :], reduction="none")
            accel_cost = F.mse_loss(shots_accel[..., t_indexes, :], targets_accel[..., t_indexes, :], reduction="none")
            kinematics_cost = (lambda_goal * goal_cost) + (lambda_kinematics * (vel_cost + accel_cost))
            kinematics_cost = torch.masked_fill(kinematics_cost, goal_mask[..., None], value=0.0)
            kinematics_cost = kinematics_cost.mean()

            curvature_cost = F.mse_loss(shots_curvature, targets_curvature, reduction="none")
            curvature_cost = torch.masked_fill(curvature_cost, pos_mask[..., 2:], value=0)
            curvature_cost = curvature_cost.mean()

            curvature_rate_cost = F.mse_loss(shots_curvature_rate, targets_curvature_rate, reduction="none")
            curvature_rate_cost = torch.masked_fill(curvature_rate_cost, pos_mask[..., 3:], value=0.0)
            curvature_rate_cost = curvature_rate_cost.mean()

            curvature_cost = lambda_curvature * (curvature_cost + curvature_rate_cost)

            pos_smoothness_cost = (shots[..., 2:, :] - (2 * shots[..., 1:-1, :]) + shots[..., :-2, :]).pow(2)
            pos_smoothness_cost = torch.masked_fill(pos_smoothness_cost, pos_mask[..., 2:, None], value=0.0)
            pos_smoothness_cost = pos_smoothness_cost.mean()

            vel_smoothness_cost = (shots_vel[..., 1:, :] - shots_vel[..., :-1, :]).pow(2)
            accel_smoothness_cost = (shots_accel[..., 1:, :] - shots_accel[..., :-1, :]).pow(2)
            kinematic_smoothness_cost = vel_smoothness_cost + accel_smoothness_cost
            kinematic_smoothness_cost = torch.masked_fill(kinematic_smoothness_cost, pos_mask[..., 1:, None], value=0.0)
            kinematic_smoothness_cost = kinematic_smoothness_cost.mean()

            smoothness_cost = lambda_smoothness * (pos_smoothness_cost + kinematic_smoothness_cost)

            cost = pos_cost + kinematics_cost + smoothness_cost + curvature_cost
            
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

        best_shots = ((xy_range[:, 1] - xy_range[:, 0]) * (best_shots + 1) / 2) + xy_range[:, 0]
        return best_shots