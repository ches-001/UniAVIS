import torch
import torch.nn as nn
from .common import PosEmbedding1D, PosEmbedding2D, SimpleMLP, AddNorm
from .attentions import MultiHeadedAttention, DeformableAttention
from typing import *


class PlanFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        num_ref_points: int=4,
        dim_feedforward: int=512, 
        dropout: float=0.1,
        offset_scale: float=1.0,
        bev_feature_hw: Tuple[int, int]=(200, 200),
    ):
        super(PlanFormerDecoderLayer, self).__init__()

        self.num_heads         = num_heads
        self.embed_dim         = embed_dim
        self.num_ref_points    = num_ref_points
        self.dim_feedforward   = dim_feedforward
        self.dropout           = dropout
        self.offset_scale      = offset_scale
        self.bev_feature_hw = bev_feature_hw

        self.self_attention      = MultiHeadedAttention(
            self.num_heads, 
            self.embed_dim, 
            dropout=self.dropout, 
        )
        self.addnorm1            = AddNorm(input_dim=self.embed_dim)
        self.deform_attention    = DeformableAttention(
            self.num_heads,
            self.embed_dim, 
            num_ref_points=num_ref_points, 
            dropout=self.dropout, 
            offset_scale=self.offset_scale,
            num_fmap_levels=1,
            concat_vq_for_offset=False,
        )
        self.addnorm2            = AddNorm(input_dim=self.embed_dim)
        self.mlp                 = SimpleMLP(self.embed_dim, self.embed_dim, self.dim_feedforward)

    def forward(
            self, 
            queries: torch.Tensor,
            bev_features: torch.Tensor,
            ref_points: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Input
        --------------------------------
        :queries: (N, num_queries, det_embeds) input queries (num_queries = num_detections + num_track)

        :bev_features: (N, W_bev * H_bev, (C_bev or embed_dim))

        :ref_points: (N, num_queries, 1, 2), reference points for the deformable attention

        Returns
        --------------------------------
        :plan_queries: (N, num_queries, embed_dim), output queries to be fed into the next layer
        """
        H_bev, W_bev = self.bev_feature_hw
        assert bev_features.shape[1] == H_bev * W_bev
        assert bev_features.shape[2] == queries.shape[2] and bev_features.shape[2] == self.embed_dim

        bev_spatial_shape = torch.LongTensor([[H_bev, W_bev]], device=queries.device)

        out1 = self.self_attention(queries, queries, queries)
        out2 = self.addnorm1(queries, out1)
        out3 = self.deform_attention(
            out2, 
            ref_points, 
            bev_features, 
            bev_spatial_shape, 
            normalize_ref_points=False
        )
        out4 = self.addnorm2(out2, out3)
        out5 = self.mlp(out4)
        return out5


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
            bev_feature_hw: Tuple[int, int]=(200, 200)
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
        self.bev_feature_hw    = bev_feature_hw
        
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
            x_dim=self.bev_feature_hw[1], 
            y_dim=self.bev_feature_hw[0], 
            embed_dim=self.embed_dim, 
            learnable=False
        )
        self.plan_queries_mlp     = SimpleMLP(
            self.embed_dim, 
            self.embed_dim, 
            self.dim_feedforward, 
            final_activation=nn.MaxPool2d(kernel_size=(self.num_modes, 1), stride=1)
        )
        self.trajectory_mlp       = SimpleMLP(self.embed_dim, self.pred_horizon * 2, self.dim_feedforward)
        self.decoder_modules      = self._create_decoder_layers()

    def _create_decoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([PlanFormerDecoderLayer(
            num_heads=self.num_heads, 
            embed_dim=self.embed_dim, 
            num_ref_points=self.num_ref_points, 
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            offset_scale=self.offset_scale,
            bev_feature_hw=self.bev_feature_hw
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

        :track_queries: (N, embed_dim) Ego vehicle track queries from the TrackFormer

        :motion_queries: (N, k, embed_dim), Ego vehicle motion queries from the MotionFormer"

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
            self.bev_feature_hw, 
            batch_size=batch_size,
            device=device,
            normalize=True,
            n_sample=1
        )
        ref_points     = ref_points[..., None, :]

        for i in range(0, self.num_layers):
            plan_queries = self.decoder_modules[i](
                queries=plan_queries, 
                bev_features=bev_features, 
                ref_points=ref_points
            )
        trajectories = self.trajectory_mlp(plan_queries).reshape(batch_size, self.pred_horizon, 2).cumsum(dim=1)
        
        return plan_queries[:, 0, :], trajectories