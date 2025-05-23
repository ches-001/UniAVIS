import torch
import torch.nn as nn
from .common import AddNorm, PosEmbedding1D, SpatialSinusoidalPosEmbedding, SimpleMLP
from .attentions import MultiHeadedAttention, DeformableAttention
from typing import *


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
        ) -> torch.Tensor:

        """
        Input
        --------------------------------
        :bev_features: (N, H_bev * W_bev, (C_bev or embed_dim)) Bird eye view features from BEVFormer

        :queries: (N, max_num_agents * k, embed_dim), context query from previous layer + corresponding positional embedding
                 (k = num_modes or number of modes)

        :agent_queries: (N, max_num_agents, embed_dim) Agent queries from the TrackFormer network

        :map_queries: (N, max_num_maps, embed_dim) Map queries from the Mapformer network

        :proj_matrix: (N, max_num_agents, 3, 3) projection matrix that projects each agent from agent-level coordinates to 
                      global-level coordinates (in homogeneous coordinates) in the real world

        Returns
        --------------------------------
        output: (N, num_queries, embed_dim), output queries to be fed into the next layer
        """
        self_attn_queries  = self.self_attention(queries, queries, queries)
        self_attn_queries  = self.self_addnorm(self_attn_queries, queries)

        agent_ctx_queries  = self.agent_cross_attention(self_attn_queries, agent_queries, agent_queries)
        agent_ctx_queries  = self.agent_cross_addnorm(agent_ctx_queries, self_attn_queries)

        map_ctx_queries    = self.agent_cross_attention(self_attn_queries, map_queries, map_queries)
        map_ctx_queries    = self.map_cross_addnorm(map_ctx_queries, self_attn_queries)

        bev_spatial_shape  = torch.tensor([self.bev_feature_hw], device=bev_features.device, dtype=torch.int64)
        goal_point_queries = self.deformable_attention(
            queries, ref_points, bev_features, bev_spatial_shape, normalize_ref_points=False
        )
        
        ctx_queries        = torch.concat([agent_ctx_queries, map_ctx_queries, goal_point_queries], dim=-1)
        output             = self.mlp(ctx_queries)
        return output
    

class MotionFormer(nn.Module):
    def __init__(
            self,
            num_heads: int, 
            embed_dim: int,
            max_num_agents: int,
            max_num_maps: int,
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

        self.num_heads          = num_heads
        self.embed_dim          = embed_dim
        self.max_num_agents     = max_num_agents
        self.max_num_maps       = max_num_maps
        self.num_layers         = num_layers
        self.num_modes          = num_modes
        self.num_ref_points     = num_ref_points
        self.pred_horizon       = pred_horizon
        self.dim_feedforward    = dim_feedforward
        self.dropout            = dropout
        self.offset_scale       = offset_scale
        self.learnable_pe       = learnable_pe
        self.bev_feature_hw  = bev_feature_hw
        self.grid_xy_res        = grid_xy_res

        self.spatial_pos_emb     = SpatialSinusoidalPosEmbedding(self.embed_dim)
        self.agent_query_pos_emb = PosEmbedding1D(
            self.max_num_agents, 
            self.embed_dim, 
            learnable=False
        )
        self.map_query_pos_emb   = PosEmbedding1D(
            self.max_num_maps, 
            self.embed_dim, 
            learnable=False
        )
        self.ctx_query_emb       = PosEmbedding1D(
            self.max_num_agents * self.num_modes, 
            self.embed_dim, 
            learnable=self.learnable_pe
        )
        self.agent_anchor_fc = nn.Linear(self.embed_dim, self.embed_dim)
        self.scene_anchor_fc = nn.Linear(self.embed_dim, self.embed_dim)
        self.current_pos_fc  = nn.Linear(self.embed_dim, self.embed_dim)
        self.goal_pos_fc     = nn.Linear(self.embed_dim, self.embed_dim)

        self.decoder_modules    = self._create_decoder_layers()

        self.agent_query_mlp    = SimpleMLP(self.embed_dim, self.embed_dim, self.dim_feedforward)
        self.map_query_mlp      = SimpleMLP(self.embed_dim, self.embed_dim, self.dim_feedforward)
        self.mode_score_mlp     = SimpleMLP(self.embed_dim, 1, self.dim_feedforward, final_activation=nn.Softmax(dim=-1))
        self.trajectory_mlp     = SimpleMLP(self.embed_dim, self.pred_horizon * 5, self.dim_feedforward)

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
            agent_queries: torch.Tensor,
            map_queries: torch.Tensor,
            proj_matrix: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Input
        --------------------------------
        :agent_current_pos: (N, max_num_agents, 2) batch of current positions of multiple agents

        :agent_anchors: (K, T, 2) a cluster of agent level trajectory points for K-modalities, 
                              T is length of trajectory / prediction horizon. Agent level means
                              that the point is relative to the given agent as reference point.

        :bev_features: (N, H_bev * W_bev, (C_bev or embed_dim)) Bird eye view features from BEVFormer

        :agent_queries: (N, max_num_agents, embed_dim) Agent queries from the TrackFormer network 
                        (max_num_agents = max number of agents)

        :map_queries: (N, max_num_maps, embed_dim) Map queries from the MapFormer network 
                    (max_num_maps = max number of mapped areas)

        :proj_matrix: (N, max_num_agents, 3, 3) projection (2D rotation and translation combined) matrix that projects 
                      each agent from agent-level coordinates to global-level coordinates (in homogeneous coordinates)
                      in the real world. NOTE: Typically. 

        Returns
        --------------------------------
        :motion_queries: (N, max_num_agents, k, embed_dim), MotionFormer final context query output

        :mode_trajectories: (N, max_num_agents, k, T, 5), batch of estimated k-mode T-long trajectories for each agent
                            (k = number of modes, T = length of trajectory). 
                            Last dim corresponds to (u_x, u_y, log_sigma_x, log_sigma_y, x_y_corr)

        :mode_scores: (N, max_num_agents, K), batch of estimated probability scores for each k-mode
        """
        batch_size, *_       = bev_features.shape
        k, *_                = agent_anchors.shape
        _, max_num_agents, _ = agent_queries.shape
        device               = bev_features.device
        agent_queries        = self.agent_query_mlp(agent_queries + self.agent_query_pos_emb())
        map_queries          = self.map_query_mlp(map_queries + self.map_query_pos_emb())
        
        # scene level anchors are points global coordinates that are not specific to any agent level coordinates
        # since we wish to generate scene level anchors for each agent level anchor, we generate them like so:
        # I^s = R_i \cdot I^a + T_i, where R_i and T_i are agent specific rotation and translation matrices used for
        # projecting from agent level coordinates to scene level coordinates. In this implementation, the rotation (2x2)
        # and translation (2x1) matrices are combined to make a 3x3 projection matrix P_i, and the agent level anchors 
        # are converted to homogeneous coordinates so that I^s = P_i \cdot I^a, Where I^a = agent level anchors and 
        # I^s = scene level anchors
        # NOTE: here, we only really need the last timestep of the agent level anchors and the scene level anchors.
        # agent_anchors shape: (1, 1,   k, 2)
        # scene_anchors shape: (N, max_num_agents, k, 2)
        agent_anchors = agent_anchors[:, -1, :]
        agent_anchors = agent_anchors[None, None, :, :]
        ones          = torch.ones(*agent_anchors.shape[:-1], 1, dtype=agent_anchors.dtype, device=device)
        scene_anchors = torch.concat([agent_anchors, ones], dim=-1)
        scene_anchors = torch.einsum("natii,ttcik->nacik", proj_matrix[:, :, None], scene_anchors[..., None])
        scene_anchors = scene_anchors[..., :2, 0]

        # initialize the agent goal position to the last timestep of the scene level anchor for the first decoder layer
        # and expand the dimensions of the agent_current_pos to (N, max_num_agents, 1, 2) to match the rest
        agent_goal_pos        = scene_anchors

        agent_current_pos     = agent_current_pos[..., None, :]
        
        agent_anchors_emb     = self.agent_anchor_fc(self.spatial_pos_emb(agent_anchors))
        scene_anchors_emb     = self.scene_anchor_fc(self.spatial_pos_emb(scene_anchors))
        agent_current_pos_emb = self.current_pos_fc(self.spatial_pos_emb(agent_current_pos))
        agent_goal_pos_emb    = self.goal_pos_fc(self.spatial_pos_emb(agent_goal_pos))
        query_pos_emb         = agent_anchors_emb + scene_anchors_emb + agent_current_pos_emb + agent_goal_pos_emb
        query_pos_emb         = query_pos_emb.reshape(batch_size, max_num_agents * k, self.embed_dim)

        # initialize context query to a learned positional embedding
        ctx_queries           = self.ctx_query_emb()

        queries               = ctx_queries + query_pos_emb

        # reshape agent_goal_pos to (N, max_num_agents, 2) and create a ones tensor to set it and the next
        # agent_goal_pos to homogeneous coordinate and tile k-times along the second axis
        agent_goal_pos        = agent_goal_pos[..., 0, :].tile(1, k, 1)
        ones                  = torch.ones(*agent_goal_pos.shape[:-1], 1, dtype=agent_anchors.dtype, device=device)
        grid_xy_res           = torch.tensor(self.grid_xy_res, device=device)
        bev_wh                = torch.tensor([self.bev_feature_hw[1], self.bev_feature_hw[0]], device=device)
        min_real_xy           = (-bev_wh / 2) * grid_xy_res
        max_real_xy           = -min_real_xy

        # tile the projection matrix because the projection also needs to happen for each agent across each modality
        proj_matrix           = proj_matrix.tile(1, k, 1, 1)

        for i in range(0, self.num_layers):
            # we set the reference points for the deformable attention between the queries and the bev features to
            # scene-level goal positions predicted in the previous layer of each agent, we then convert the scene
            # level coordinates to BEV grid space coordinates. Since the reference points are computed from the goal
            # positions (which have a shape of (N, max_num_agents, 2)), we tile the second dimension by k, ensuring
            # that each modality has the same reference points, and to also match the reference points to the context
            # query which has a shape of (N, max_num_agents * k, num_embed)
            
            # TODO: Revisit this ref_points conversion
            ref_points  = torch.concat([agent_goal_pos, ones], dim=-1)
            ref_points  = torch.einsum("naii,naik->naik", proj_matrix, ref_points[..., None])
            ref_points  = ref_points[..., :2, 0][..., None, :]
            ref_points  = 2 * ((ref_points - min_real_xy) / (max_real_xy - min_real_xy)) - 1

            ctx_queries = self.decoder_modules[i](
                bev_features=bev_features,
                queries=queries,
                agent_queries=agent_queries, 
                map_queries=map_queries,
                ref_points=ref_points,
            )

            mode_trajectories = self.trajectory_mlp(ctx_queries)
            mode_scores       = self.mode_score_mlp(ctx_queries)
            mode_trajectories = mode_trajectories.reshape(batch_size, max_num_agents, k, self.pred_horizon, 5)

            # we use cumsum method here because the agent estimates velocities, given that velocity is the rate of
            # change of distance overtime, a cumulative sum of velocities will yield an actual distance trajectory
            xy_traj           = mode_trajectories[..., :2].cumsum(dim=3)
            mode_trajectories = torch.concat([xy_traj, mode_trajectories[..., 2:]], dim=-1)
            mode_scores       = mode_scores.reshape(batch_size, max_num_agents, k)

            # update positional embedding based on predicted goal positions for each agent.
            agent_goal_pos     = mode_trajectories[..., -1, :2]
            agent_goal_pos_emb = self.goal_pos_fc(self.spatial_pos_emb(agent_goal_pos))
            agent_goal_pos     = agent_goal_pos.reshape(batch_size, max_num_agents * k, 2)
            query_pos_emb      = agent_anchors_emb + scene_anchors_emb + agent_current_pos_emb + agent_goal_pos_emb
            queries            = ctx_queries + query_pos_emb.reshape(batch_size, max_num_agents * k, self.embed_dim)

        ctx_queries = ctx_queries.reshape(batch_size, max_num_agents, k, self.embed_dim)
        return ctx_queries, mode_trajectories, mode_scores