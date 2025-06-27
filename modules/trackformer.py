import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseFormer
from .attentions import MultiHeadedAttention, DeformableAttention
from .common import AddNorm, PosEmbedding1D, DetectionHead, SimpleMLP
from typing import Optional, Tuple, Dict


class TrackFormerDecoderLayer(nn.Module):
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
        super(TrackFormerDecoderLayer, self).__init__()

        self.num_heads       = num_heads
        self.embed_dim       = embed_dim
        self.num_ref_points  = num_ref_points
        self.dim_feedforward = dim_feedforward
        self.dropout         = dropout
        self.offset_scale    = offset_scale
        self.bev_feature_hw  = bev_feature_hw

        self.self_attention   = MultiHeadedAttention(
            self.num_heads, 
            self.embed_dim, 
            dropout=self.dropout, 
        )
        self.addnorm1         = AddNorm(input_dim=self.embed_dim)
        self.deform_attention = DeformableAttention(
            self.num_heads,
            self.embed_dim, 
            num_ref_points=self.num_ref_points, 
            dropout=self.dropout, 
            offset_scale=self.offset_scale,
            num_fmap_levels=1,
            concat_vq_for_offset=False,
        )
        self.addnorm2 = AddNorm(input_dim=self.embed_dim)
        self.mlp      = SimpleMLP(self.embed_dim, self.embed_dim, self.dim_feedforward)
        self.addnorm3 = AddNorm(input_dim=self.embed_dim)

    def forward(
            self, 
            queries: torch.Tensor,
            bev_features: torch.Tensor,
            ref_points: torch.Tensor, 
            orig_det_queries: Optional[torch.Tensor]=None,
        ) -> torch.Tensor:
        """
        Input
        --------------------------------

        :queries: (N, num_queries, embed_dim) input queries

        :bev_features: (N, W_bev * H_bev, (C_bev or embed_dim))

        :ref_points: (N, num_queries, 1, 2), reference points for the deformable attention

        :orig_det_queries: (N, num_queries, embed_dim), original object / detection queries


        Returns
        --------------------------------
        :track_queries: (N, num_queries, embed_dim), output queries to be fed into the next layer
        """
        H_bev, W_bev = self.bev_feature_hw
        assert bev_features.shape[1] == H_bev * W_bev
        assert bev_features.shape[2] == queries.shape[2] and bev_features.shape[2] == self.embed_dim

        bev_spatial_shape = torch.tensor([[H_bev, W_bev]], device=queries.device, dtype=torch.int64)
        
        q_and_k = queries
        if orig_det_queries is not None:
            q_and_k = q_and_k + orig_det_queries

        out1 = self.self_attention(q_and_k, q_and_k, queries)

        out2 = self.addnorm1(queries, out1)

        if orig_det_queries is not None:
            deform_attn_queries = out2 + orig_det_queries
        else:
            deform_attn_queries = out2
        
        out3 = self.deform_attention(
            deform_attn_queries, 
            ref_points, 
            bev_features, 
            bev_spatial_shape, 
            normalize_ref_points=False
        )
        out3 = self.addnorm2(out2, out3)
        out4 = self.mlp(out3)
        out5 = self.addnorm3(out3, out4)
        return out5


class TrackFormer(BaseFormer):
    def __init__(
            self,
            num_classes: int,
            num_heads: int=8, 
            embed_dim: int=256,
            num_layers: int=6,
            num_ref_points: int=4,
            dim_feedforward: int=512, 
            dropout: float=0.1,
            offset_scale: float=1.0,
            max_detections: int=900,
            learnable_pe: bool=True,
            bev_feature_hw: Tuple[int, int]=(200, 200),
            det_3d: bool=True
        ):
        super(TrackFormer, self).__init__()

        self.num_classes     = num_classes
        self.num_heads       = num_heads
        self.embed_dim       = embed_dim
        self.num_layers      = num_layers
        self.num_ref_points  = num_ref_points
        self.dim_feedforward = dim_feedforward
        self.dropout         = dropout
        self.offset_scale    = offset_scale
        self.max_detections  = max_detections
        self.learnable_pe    = learnable_pe
        self.bev_feature_hw  = bev_feature_hw
        self.det_3d          = det_3d

        self.detection_pos_emb  = PosEmbedding1D(
            self.max_detections, 
            embed_dim=self.embed_dim, 
            learnable=learnable_pe
        )
        self.ego_query_emb      = PosEmbedding1D(
            1,
            embed_dim=self.embed_dim,
            learnable=True
        )
        self.decoder_modules    = self._create_decoder_layers()
        self.detection_module   = DetectionHead(
            embed_dim=self.embed_dim, 
            num_classes=self.num_classes, 
            det_3d=self.det_3d,
        )

    def _create_decoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([TrackFormerDecoderLayer(
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
            bev_features: torch.Tensor, 
            track_queries: Optional[torch.Tensor]=None,
            track_queries_mask: Optional[torch.BoolTensor]=None,
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        """
        Input
        --------------------------------
        :bev_features: (N, H_bev*W_bev, (C_bev or embed_dim)), BEV features from the BevFormer encoder

        :track_queries: (N, num_queries, embed_dim), embedding output of TrackFormer decoder at previous timestep (t-1)

        :track_queries_mask: (N, num_queries), boolean mask, indicating the tracks with valid and invalid detections
                            NOTE: Detection is invalid if score <= track_threshold (1 if valid, else 0)

        Returns
        --------------------------------
        :queries: (N, num_queries, embed_dim) batch of output context query for each segmented item (including invalid detections).

        :detections: (N, num_queries, det_params) | (N, num_queries, det_params) tensor contains 2d or 3d box detections
        """
        assert bev_features.shape[-1] == self.embed_dim

        batch_size        = bev_features.shape[0]
        orig_det_queries  = None
        
        if track_queries is not None:
            # if track queries are available, combine them with the static detection queries
            # the detection queries are responsible for detecting new objects that enter the frame, the 
            # track queries on the other hand are queries used to persist a detection across multiple frames
            # for as long as said detection is alive, inotherwords, tracking.
            assert track_queries.shape[-1] == self.embed_dim
            assert track_queries.shape[1] == track_queries_mask.shape[1]
            
            orig_det_queries = self.detection_pos_emb().tile(batch_size, 1, 1)
            queries          = []
            for i in range(0, batch_size):
                track_mask  = track_queries_mask[i]
                track_query = track_queries[i][track_mask]
                query        = torch.concat([track_query, orig_det_queries[0, track_query.shape[0]:]], dim=0)
                queries.append(query)
            queries = torch.stack(queries, dim=0)

        else:
            queries = self.detection_pos_emb().tile(batch_size, 1, 1)
        
        ref_points = DeformableAttention.generate_standard_ref_points(
            self.bev_feature_hw,
            batch_size=batch_size, 
            device=bev_features.device, 
            normalize=True, 
            n_sample=queries.shape[1] + 1 # (+1 because of the ego queries)
        )
        ref_points = ref_points[:, :, None, :]

        # make ego query and combine then with detection / track queries, model then in the 
        # decoder layers and split them up into agent queries and ego queries
        ego_query = self.ego_query_emb().tile(batch_size, 1, 1)
        queries = torch.concat([ego_query, queries], dim=1)
        if orig_det_queries is not None:
            orig_det_queries = torch.concat([
                torch.zeros_like(orig_det_queries[:, [0], :]), orig_det_queries
            ], dim=1)

        
        for decoder_idx in range(0, len(self.decoder_modules)):
            queries = self.decoder_modules[decoder_idx](
                queries=queries,
                bev_features=bev_features,
                ref_points=ref_points,
                orig_det_queries=orig_det_queries,
            )

        detections = self.detection_module(queries)
        ego_data   = dict(ego_query=queries[:, 0, :], ego_detection=detections[:, 0, :])
        queries    = queries[:, 1:, :]
        detections = detections[..., 1:, :]
        
        return queries, detections, ego_data