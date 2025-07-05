import torch
import torch.nn as nn
from .base import BaseFormer
from .common import AddNorm, PosEmbedding1D, PosEmbedding2D, SpatialSinusoidalPosEmbedding, SimpleMLP
from .attentions import MultiHeadedAttention, DeformableAttention
from utils.img_utils import transform_points
from typing import Tuple, Dict, Optional


class MotionFormerDecoderLayer(nn.Module):
    def __init__(
            self,
            num_heads: int, 
            embed_dim: int,
            num_ref_points: int,
            dim_feedforward: int=512, 
            dropout: float=0.1,
            offset_scale: float=1.0,
            bev_feature_hw: Tuple[int, int]=(200, 200),
        ):
        super(MotionFormerDecoderLayer, self).__init__()

        self.num_heads          = num_heads
        self.embed_dim          = embed_dim
        self.num_ref_points     = num_ref_points
        self.dim_feedforward    = dim_feedforward
        self.dropout            = dropout
        self.offset_scale       = offset_scale
        self.bev_feature_hw  = bev_feature_hw

        self.self_attention        = MultiHeadedAttention(self.num_heads, self.embed_dim, self.dropout)
        self.self_addnorm          = AddNorm(self.embed_dim)

        self.agent_cross_attention = MultiHeadedAttention(self.num_heads, self.embed_dim, self.dropout)
        self.agent_cross_addnorm   = AddNorm(self.embed_dim)

        self.map_cross_attention   = MultiHeadedAttention(self.num_heads, self.embed_dim, self.dropout)
        self.map_cross_addnorm     = AddNorm(self.embed_dim)

        self.deformable_attention  = DeformableAttention(
            self.num_heads, 
            embed_dim, 
            num_ref_points=self.num_ref_points,
            dropout=self.dropout,
            offset_scale=self.offset_scale,
            num_fmap_levels=1,
            concat_vq_for_offset=False
        )
        self.mlp                  = SimpleMLP(self.embed_dim * 3, self.embed_dim, self.dim_feedforward)

    def forward(
            self, 
            bev_features: torch.Tensor,
            queries: torch.Tensor,
            agent_queries: torch.Tensor, 
            map_queries: torch.Tensor,
            ref_points: torch.Tensor,
            agent_pad_mask: Optional[torch.Tensor]=None,
            agent_agent_attn_mask: Optional[torch.Tensor] = None,
            agent_map_attn_mask: Optional[torch.Tensor] = None,
            agent_bev_attn_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:

        """
        Input
        --------------------------------
        :bev_features: (N, H_bev * W_bev, (C_bev or embed_dim)) Bird eye view features from BEVFormer

        :queries: (N, k, num_agents, embed_dim), context query from previous layer + corresponding positional embedding
                 (k = num_modes or number of modes)

        :agent_queries: (N, 1, num_agents, embed_dim) Agent queries from the TrackFormer network

        :map_queries: (N, 1, num_map_elements, embed_dim) Map queries from the Mapformer network

        :ref_points: (N, k, num_agents, 1, 2), reference points for the deformable attention

        :agent_pad_mask: bool type (N, 1, num_agents), bool padding mask for agent queries (0 if pad value, else 1):

        :agent_agent_attn_mask: bool type (N, 1, num_agents, num_agents) (0 if pad value, else 1):

        :agent_map_attn_mask: bool type (N, 1, num_agents, num_map_elements) (0 if pad value, else 1):
        
        :agent_bev_attn_mask: bool type (N, num_agents * k, 1) (0 if pad value, else 1):

        Returns
        --------------------------------
        output: (N, k, num_agents, embed_dim), output queries to be fed into the next layer
        """
        if agent_pad_mask is not None:
            assert agent_pad_mask.dtype == torch.bool

        if agent_agent_attn_mask is not None:
            assert agent_agent_attn_mask.dtype == torch.bool

        if agent_map_attn_mask is not None:
            assert agent_map_attn_mask.dtype == torch.bool

        if agent_bev_attn_mask is not None:
            assert agent_bev_attn_mask.dtype == torch.bool

        self_attn_queries  = self.self_attention(
            queries, 
            queries, 
            queries, 
            attention_mask=agent_agent_attn_mask
        )
        self_attn_queries  = self.self_addnorm(self_attn_queries, self._apply_mask(queries, agent_pad_mask))

        agent_ctx_queries  = self.agent_cross_attention(
            self_attn_queries, 
            agent_queries, 
            agent_queries, 
            attention_mask=agent_agent_attn_mask
        )
        agent_ctx_queries  = self.agent_cross_addnorm(agent_ctx_queries, self_attn_queries)

        map_ctx_queries    = self.map_cross_attention(
            self_attn_queries,
            map_queries,
            map_queries,
            attention_mask=agent_map_attn_mask
        )
        map_ctx_queries    = self.map_cross_addnorm(map_ctx_queries, self_attn_queries)

        bev_spatial_shape  = torch.tensor([self.bev_feature_hw], device=bev_features.device, dtype=torch.int64)

        goal_point_queries = self.deformable_attention(
            queries.flatten(start_dim=1, end_dim=2),
            ref_points.flatten(start_dim=1, end_dim=2),
            bev_features,
            bev_spatial_shape,
            attention_mask=agent_bev_attn_mask,
            normalize_ref_points=False
        )
        goal_point_queries = torch.unflatten(goal_point_queries, dim=1, sizes=queries.shape[1:3])
        ctx_queries        = torch.concat([agent_ctx_queries, map_ctx_queries, goal_point_queries], dim=-1)
        output             = self.mlp(ctx_queries)
        return output

    def _apply_mask(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if mask is None:
            return x
        mask = mask[..., None]
        x = torch.masked_fill(x, ~mask, 0.0)
        return x
    

class MotionFormer(BaseFormer):
    def __init__(
            self,
            max_num_agents: int,
            max_num_map_elements: int,
            num_heads: int=4, 
            embed_dim: int=128,
            num_layers: int=3,
            num_modes: int=6,
            num_ref_points: int=4,
            pred_horizon: int=12,
            dim_feedforward: int=512, 
            dropout: float=0.1,
            offset_scale: float=1.0,
            learnable_pe: bool=True,
            bev_feature_hw: Tuple[int, int]=(200, 200),
            grid_xy_res: Tuple[float, float]=(0.512, 0.512),
        ):

        super(MotionFormer, self).__init__()

        self.max_num_agents       = max_num_agents + 1 # +1 because of ego vehicle queries
        self.max_num_map_elements = max_num_map_elements
        self.num_heads            = num_heads
        self.embed_dim            = embed_dim
        self.num_layers           = num_layers
        self.num_modes            = num_modes
        self.num_ref_points       = num_ref_points
        self.pred_horizon         = pred_horizon
        self.dim_feedforward      = dim_feedforward
        self.dropout              = dropout
        self.offset_scale         = offset_scale
        self.learnable_pe         = learnable_pe
        self.bev_feature_hw       = bev_feature_hw
        self.grid_xy_res          = grid_xy_res

        self.spatial_pos_emb     = SpatialSinusoidalPosEmbedding(self.embed_dim)
        self.agent_query_pos_emb = PosEmbedding1D(
            self.max_num_agents, 
            self.embed_dim, 
            learnable=False
        )
        self.map_query_pos_emb   = PosEmbedding1D(
            self.max_num_map_elements, 
            self.embed_dim, 
            learnable=False
        )
        self.ctx_query_emb       = PosEmbedding2D(
            self.num_modes,
            self.max_num_agents,  
            self.embed_dim, 
            learnable=self.learnable_pe
        )
        self.agent_anchor_fc = nn.Linear(self.embed_dim, self.embed_dim)
        self.scene_anchor_fc = nn.Linear(self.embed_dim, self.embed_dim)
        self.current_pos_fc  = nn.Linear(self.embed_dim, self.embed_dim)
        self.goal_pos_fc     = nn.Linear(self.embed_dim, self.embed_dim)

        self.decoder_modules = self._create_decoder_layers()

        self.agent_query_mlp = SimpleMLP(self.embed_dim, self.embed_dim, self.dim_feedforward)
        self.map_query_mlp   = SimpleMLP(self.embed_dim, self.embed_dim, self.dim_feedforward)
        self.mode_score_mlp  = SimpleMLP(self.embed_dim, 1, self.dim_feedforward, final_activation=nn.Softmax(dim=-1))
        self.trajectory_mlp  = nn.Sequential(
            SimpleMLP(self.embed_dim, self.pred_horizon * 5, self.dim_feedforward),
            nn.Unflatten(dim=-1, unflattened_size=(self.pred_horizon, 5))
        )

    def _create_decoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            MotionFormerDecoderLayer(
                num_heads=self.num_heads,
                embed_dim=self.embed_dim,
                num_ref_points=self.num_ref_points,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                offset_scale=self.offset_scale,
                bev_feature_hw=self.bev_feature_hw

            ) for _ in range(0, self.num_layers)
        ])

    def forward(
            self, 
            agent_current_pos: torch.Tensor,
            agent_anchors: torch.Tensor,
            bev_features: torch.Tensor,
            ego_query: torch.Tensor,
            agent_queries: torch.Tensor,
            map_queries: torch.Tensor,
            transform_matrices: torch.Tensor,
            agent_pad_mask: Optional[torch.Tensor]=None,
            map_pad_mask: Optional[torch.Tensor]=None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        """
        Input
        --------------------------------
        :agent_current_pos: (N, num_agents, 2) batch of current positions of multiple agents

        :agent_anchors: (K, 2) a cluster of agent level trajectory endpoints for K-modalities, 

        :bev_features: (N, H_bev * W_bev, embed_dim) Bird eye view features from BEVFormer

        :ego_query: (N, embed_dim), ego vehicle queries from the TrackFormer

        :agent_queries: (N, num_agents, embed_dim) Agent queries from the TrackFormer network 
            (num_agents = max number of agents)

        :map_queries: (N, num_map_elements, embed_dim) Map queries from the MapFormer network 
            (num_map_elements = max number of mapped areas)

        :transform_matrices: (N, num_agents, 4, 4) transformation (2D rotation and translation combined) matrix that 
            transforms each agent from agent-level coordinates to scene-level coordinates (in homogeneous coordinates)
            NOTE: Scene level coordinate in this case refers to coordinates of ego-vehicle, as in the BEV
            scene, The ego vehicle center is the reference point
        
        :agent_pad_mask: (N, num_agents), bool padding mask for invalid agent queries (0 if pad value, else 1):

        :agent_pad_mask: (N, num_map_elements), bool padding mask for invalid map element queriess (0 if pad value, else 1):

        Returns
        --------------------------------
        :motion_queries: (N, num_agents, k, embed_dim), MotionFormer final context query output

        :mode_traj: (N, num_agents, k, T, 5), batch of estimated k-mode T-long trajectories for each agent
            in agent level frame (k = number of modes, T = length of trajectory). Last dim corresponds to 
            (u_x, u_y, log_sigma_x, log_sigma_y, x_y_corr)

        :mode_scores: (N, num_agents, K), batch of estimated probability scores for each k-mode

        :ego_data: Dict of tensors as follows:

        ---------------------------------
            ego_query     (N, k, embed_dim)

            ego_mode_traj (N, k, T, 5)
            
            mode_scores   (N, k)
        """
        assert transform_matrices.shape[-1] == 4
        assert transform_matrices.shape[-2] == transform_matrices.shape[-1]
        
        batch_size = bev_features.shape[0]
        k          = agent_anchors.shape[0]
        device     = bev_features.device

        # include ego vehicle related data (including ego query) to the rest of the queries
        ego_current_pos    = torch.zeros_like(agent_current_pos[:, [0], :])
        agent_current_pos  = torch.concat([ego_current_pos, agent_current_pos], dim=1)

        agent_queries      = torch.concat([ego_query[:, None, :], agent_queries], dim=1)

        ego_transform      = torch.eye(transform_matrices.shape[-1], device=device)
        ego_transform      = ego_transform[None, None, :, :].tile(batch_size, 1, 1, 1)
        transform_matrices = torch.concat([transform_matrices, ego_transform], dim=1)

        if agent_pad_mask is not None:
            agent_pad_mask = torch.concat([
                torch.ones_like(agent_pad_mask[:, [0]]), agent_pad_mask
            ], dim=1)
            agent_pad_mask = agent_pad_mask[:, None, ...]

        if map_pad_mask is not None:
            map_pad_mask = map_pad_mask[:, None, ...]

        num_agents        = agent_queries.shape[1]
        num_map_elements  = map_queries.shape[1]
        agent_pos_indexes = torch.arange(num_agents, device=device)[None, None, :].tile(batch_size, 1, 1)
        map_pos_indexes   = torch.arange(num_map_elements, device=device)[None, None, :].tile(batch_size, 1, 1)
        agent_queries     = self.agent_query_mlp(agent_queries[:, None, :, :] + self.agent_query_pos_emb(agent_pos_indexes))
        map_queries       = self.map_query_mlp(map_queries[:, None, :, :] + self.map_query_pos_emb(map_pos_indexes))
        
        # scene level anchors:
        # I^s = R_i \cdot I^a + T_i, where R_i and T_i are agent specific rotation and translation matrices used for
        # transforming from agent level coordinates to scene level coordinates. In this implementation, the rotation (2x2)
        # and translation (2x1) matrices are combined to make a 4 x 4 transformation matrix P_i, and the agent level anchors 
        # are converted to homogeneous coordinates so that I^s = P_i \cdot I^a, Where I^a = agent level anchors and 
        # I^s = scene level anchors
        # NOTE: here, we only really need the last timestep of the agent level anchors and the scene level anchors.
        # agent_anchors shape: (1, k, 1, 2)
        # scene_anchors shape: (N, k, num_agents, 2)
        transform_matrices = transform_matrices[:, None]
        agent_anchors      = agent_anchors[None, :, None, None, :]
        scene_anchors      = transform_points(agent_anchors, transform_matrix=transform_matrices)
        agent_anchors      = agent_anchors[..., 0, :]
        scene_anchors      = scene_anchors[..., 0, :]

        agent_current_pos     = agent_current_pos[:, None, ...]
        agent_anchors_emb     = self.agent_anchor_fc(self.spatial_pos_emb(agent_anchors))
        scene_anchors_emb     = self.scene_anchor_fc(self.spatial_pos_emb(scene_anchors))
        agent_current_pos_emb = self.current_pos_fc(self.spatial_pos_emb(agent_current_pos))
        agent_goal_pos_emb    = self.goal_pos_fc(self.spatial_pos_emb(scene_anchors))
        query_pos_emb         = agent_anchors_emb + scene_anchors_emb + agent_current_pos_emb + agent_goal_pos_emb

        ctx_queries           = self.ctx_query_emb(None, agent_pos_indexes[:, 0, :]).permute(0, 2, 3, 1)
        queries               = ctx_queries + query_pos_emb
        grid_xy_res           = torch.tensor(self.grid_xy_res, device=device)
        bev_wh                = torch.tensor([self.bev_feature_hw[1], self.bev_feature_hw[0]], device=device)
        min_real_xy           = (-bev_wh / 2) * grid_xy_res
        max_real_xy           = -min_real_xy

        agent_agent_attn_mask = None
        agent_map_attn_mask   = None
        agent_bev_attn_mask   = None

        if agent_pad_mask is not None:
            agent_agent_attn_mask = agent_pad_mask[..., None] & agent_pad_mask[..., None, :]
            agent_bev_attn_mask   = agent_pad_mask[..., None].tile(1, self.num_modes, 1, 1)
            agent_bev_attn_mask   = agent_bev_attn_mask.flatten(start_dim=1, end_dim=2)
        
        if map_pad_mask is not None:
            agent_map_attn_mask = agent_pad_mask[..., None] & map_pad_mask[..., None, :]
        
        agent_goal_pos = scene_anchors[..., None, :]

        for i in range(0, self.num_layers):
            # reference points are computed from the agent goal position (final point in trajectory prediction).
            # The reference points are first converted to scene level frame since they are originally in agent
            # level frame and we need it to correspond to the BEV grid.
            ref_points  = 2 * ((agent_goal_pos - min_real_xy) / (max_real_xy - min_real_xy)) - 1

            ctx_queries = self.decoder_modules[i](
                bev_features=bev_features,
                queries=queries,
                agent_queries=agent_queries, 
                map_queries=map_queries,
                ref_points=ref_points,
                agent_pad_mask=agent_pad_mask,
                agent_agent_attn_mask=agent_agent_attn_mask,
                agent_map_attn_mask=agent_map_attn_mask,
                agent_bev_attn_mask=agent_bev_attn_mask
            )

            mode_traj   = self.trajectory_mlp(ctx_queries)
            mode_scores = self.mode_score_mlp(ctx_queries)

            # The agent current position is 0 because each agent is treated as its own reference point,
            # so to convert the trajectory to scene level just compute the appropriate transformation 
            # matrix and use it to transform the points. The points are intially (v x dt = dx) from which
            # we compute the trajectory via a cummulative sum operation over the timestep axis.
            xy_traj            = torch.cumsum(mode_traj[..., :2], dim=3)
            mode_traj          = torch.concat([xy_traj, mode_traj[..., 2:4], torch.tanh(mode_traj[..., 4:])], dim=-1)
            agent_goal_pos     = mode_traj[..., [-1], :2]
            agent_goal_pos     = transform_points(agent_goal_pos, transform_matrix=transform_matrices)
            agent_goal_pos_emb = self.goal_pos_fc(self.spatial_pos_emb(agent_goal_pos[..., 0, :]))
            query_pos_emb      = agent_anchors_emb + scene_anchors_emb + agent_current_pos_emb + agent_goal_pos_emb
            queries            = ctx_queries + query_pos_emb

        ctx_queries = torch.transpose(ctx_queries, 1, 2)
        mode_traj   = torch.transpose(mode_traj, 1, 2)
        mode_scores = torch.transpose(mode_scores, 1, 2)

        ego_data    = dict(
            ego_query=ctx_queries[:, 0, :], 
            ego_mode_traj=mode_traj[:, 0, ...], 
            ego_mode_score=mode_scores[:, 0, ...]
        )
        ctx_queries = ctx_queries[:, 1:, :]
        mode_traj   = mode_traj[:, 1:, ...]
        mode_scores = mode_scores[:, 1:, ...] 
        return ctx_queries, mode_traj, mode_scores, ego_data