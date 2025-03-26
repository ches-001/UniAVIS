import torch
import torch.nn as nn
from .common import PosEmbedding1D, PosEmbedding2D
from .attentions import DeformableAttention
from .trackformer import TrackFormerDecoderLayer
from typing import *


class PlanFormerDecoderLayer(TrackFormerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super(PlanFormerDecoderLayer, self).__init__(*args, **kwargs)


class PlanFormer(nn.Module):
    def __init__(
            self,
            num_heads: int, 
            embed_dim: int,
            num_commands: int,
            num_layers: int=3,
            num_modes: int=6,
            num_ref_points: int=4,
            pred_horizon: int=6,
            dim_feedforward: int=512,
            dropout: float=0.1,
            offset_scale: float=1.0,
            learnable_pe: bool=True,
            bev_feature_shape: Tuple[int, int]=(200, 200)
        ):
        super(PlanFormer, self).__init__()

        self.num_heads            = num_heads
        self.embed_dim            = embed_dim
        self.num_commands         = num_commands
        self.num_layers           = num_layers
        self.num_modes            = num_modes
        self.num_ref_points       = num_ref_points
        self.pred_horizon         = pred_horizon
        self.dim_feedforward      = dim_feedforward
        self.dropout              = dropout
        self.offset_scale         = offset_scale
        self.learnable_pe         = learnable_pe
        self.bev_feature_shape    = bev_feature_shape
        
        self.commands_emb         = PosEmbedding1D(
            self.num_commands, 
            embed_dim=self.embed_dim, 
            learnable=learnable_pe
        )
        self.plan_pos_emb         = PosEmbedding1D(
            1, 
            embed_dim=self.embed_dim, 
            learnable=learnable_pe
        )
        self.bev_pos_emb          = PosEmbedding2D(
            x_dim=self.bev_feature_shape[1], 
            y_dim=self.bev_feature_shape[0], 
            embed_dim=self.embed_dim, 
            learnable=False
        )
        self.plan_queries_mlp     = nn.Sequential(
            nn.Linear(self.embed_dim, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.embed_dim),
            nn.MaxPool2d(kernel_size=(self.num_modes, 1), stride=1)
        )

        self.trajectory_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.pred_horizon * 2),
        )

        self.decoder_modules       = self._create_decoder_layers()

    def _create_decoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([PlanFormerDecoderLayer(
            num_heads=self.num_heads, 
            embed_dim=self.embed_dim, 
            num_ref_points=self.num_ref_points, 
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            offset_scale=self.offset_scale,
            bev_feature_shape=self.bev_feature_shape
        ) for _ in range(self.num_layers)])

    def forward(
            self,
            commands: torch.LongTensor, 
            bev_features: torch.Tensor, 
            track_queries: torch.Tensor, 
            motion_queries: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input
        --------------------------------
        :commands: (N, 1) Integer encoded commands that encode the direction intentions (turn left, turn right, etc)

        :bev_features: (N, H_bev * W_bev, (C_bev or embed_dim)) Bird eye view features from BEVFormer

        :track_queries: (N, embed_dim) Agent track queries from the TrackFormer

        :motion_queries: (N, k, embed_dim), Motion queries from the MotionFormer"

        Returns
        --------------------------------
        :plan_queries: (N, embed_dim), PlanFormer / Planner final context query output

        :trajectories: (N, T, 2), batch of estimated trajectory of waypoints
                            Last dim corresponds to (x, y) of waypoints
        """

        batch_size = track_queries.shape[0]
        device     = track_queries.device

        commands_emb   = self.commands_emb()[0][commands.squeeze()][:, None, :]
        plan_pos_emb   = self.plan_pos_emb()
        bev_pos_emb    = self.bev_pos_emb(flatten=True)
        
        track_queries  = track_queries[:, None, :]
        plan_queries   = self.plan_queries_mlp(commands_emb + track_queries + motion_queries)
        plan_queries   = plan_queries + plan_pos_emb
        bev_features   = bev_features + bev_pos_emb

        ref_points     = DeformableAttention.generate_standard_ref_points(
            self.bev_feature_shape, 
            batch_size=batch_size,
            device=device,
            normalize=False,
            n_sample=1
        )
        ref_points     = ref_points[..., None, :]

        for i in range(0, self.num_layers):
            plan_queries = self.decoder_modules[i](
                queries=plan_queries, 
                bev_features=bev_features, 
                ref_points=ref_points
            )

        trajectories = self.trajectory_mlp(plan_queries)
        trajectories = trajectories.reshape(batch_size, self.pred_horizon, 2)
        trajectories = trajectories.cumsum(dim=1)
        
        return plan_queries[:, 0, :], trajectories