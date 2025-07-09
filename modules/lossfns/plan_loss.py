import torch
import torch.nn as nn
from utils.metric_utils import compute_2d_ciou
from utils.img_utils import transform_points
from typing import Optional, List, Tuple

class PlanLoss(nn.Module):
    def __init__(
            self, 
            dist_lambda: float, 
            col_lambda: float, 
            weight_tilda_pair: Optional[List[Tuple[float, float]]]=None
        ):
        super(PlanLoss, self).__init__()

        """
        dist_lambda: weight value for distance loss between predicted and target trajectories

        col_lambda: weight value for collision loss

        weight_tilda_pair: weight and pad value pair (w, tilda) for collision loss. For each pair, the multi-agent boxes
            are spaced by the corresponding tilda value and a weighted sum of each is performeed, with w as weights
        """
        self.dist_lambda = dist_lambda
        self.col_lambda = col_lambda
        self.weight_tilda_pair = weight_tilda_pair or [(1.0, 0.0), (0.4, 0.5), (0.1, 1.0)]


    def forward(
            self, 
            pred_motion: torch.Tensor, 
            target_motion: torch.Tensor,
            ego_size: torch.Tensor,
            multiagent_size: torch.Tensor,
            multiagents_motions: torch.Tensor,
            transform: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        """
        pred_motion: (N, T, 2), predicted trajectory (x, y)

        target_motion: (N, T, 2), target trajectory (x, y)

        ego_size: (N, 2) box size (w, h) of ego vehicle

        multiagent_size: (N, num_agents, 2), box sizes (w, h) of other agents (no ego trajectory)

        multiagents_motions: (N, num_agents, T, 2), trajectory of other agents (no ego trajectory)
            NOTE: This trajectory must be in ego (scene-level) vehicle frame

        transform: (N, num_agents, 3, 3), transformation matrix from agent level to scene (ego) level
            if this is None, the function assumes that the multiagents_motions is already projected to scene level.
        """
        device = pred_motion.device
        num_timesteps = pred_motion.shape[1]
        num_tildas = len(self.weight_tilda_pair)

        valid_traj_point_mask = (target_motion != -999).all(dim=-1)

        dist_loss = (pred_motion - target_motion).pow(2).sum(dim=-1).sqrt()
        dist_loss = dist_loss[valid_traj_point_mask].mean()

        ego_size = ego_size[:, None, :].tile(1, num_timesteps, 1)
        ego_box_traj = torch.concat([pred_motion, ego_size], dim=-1)
        ego_box_traj = ego_box_traj[:, None, None, :, :]

        weights_and_tilda = torch.tensor(self.weight_tilda_pair, dtype=multiagent_size.dtype, device=device)
        weights = weights_and_tilda[None, None, :, 0]
        tildas = weights_and_tilda[None, None, :, 1, None, None]
        
        if transform is not None:
            assert (
                transform.shape[-1] == 4 and
                transform.shape[-2] == transform.shape[-1]
            )
            multiagents_motions = transform_points(multiagents_motions, transform_matrix=transform)

        multiagent_size = multiagent_size[:, :, None, :].tile(1, 1, num_timesteps, 1)
        multiagent_size = multiagent_size[:, :, None, :, :] + tildas
        multiagents_motions = multiagents_motions[:, :, None, :, :].tile(1, 1, num_tildas, 1, 1)

        multiagent_box_traj = torch.concat([multiagents_motions, multiagent_size], dim=-1)

        col_loss = self._collision_loss(ego_box_traj, multiagent_box_traj, weights, valid_traj_point_mask)
        col_loss = col_loss.mean()

        loss = (self.dist_lambda * dist_loss) + (self.col_lambda * col_loss)
        return loss

    
    def _collision_loss(
            self, 
            ego_box_traj: torch.Tensor, 
            multiagent_box_traj: torch.Tensor, 
            weights: torch.Tensor,
            mask: torch.Tensor
        ) -> torch.Tensor:
        """
        ego_box_traj: (N, 1, 1, T, 4)
        
        multiagent_box_traj: (N, num_agents, num_tilda, T, 4)

        weights: (1, 1, num_weights), num_tilda is same as num_weights

        mask: (N, T)
        """
        num_agents = multiagent_box_traj.shape[1]
        num_tilda = multiagent_box_traj.shape[2]

        mask = mask[:, None, None, :].tile(1, num_agents, num_tilda, 1)
        ciou = compute_2d_ciou(ego_box_traj, multiagent_box_traj)
        ciou = torch.where(mask, ciou, 0.0)

        # NOTE: division by 0 ought to be impossible here, because that would imply that a given ego
        # vehicle of a given batchdoes not have any valid points but pad values in its trajectory.
        ciou = ciou.sum(dim=-1) / mask.sum(dim=-1)
        ciou = (ciou * weights).sum(dim=-1)
        return ciou