import torch
import torch.nn as nn
from .attentions import MultiHeadedAttention, DeformableAttention
from .common import AddNorm, PosEmbedding1D, DetectionHead
from typing import *


class TrackFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int, 
        embed_dim: int,
        num_ref_points: int=4,
        dim_feedforward: int=512, 
        dropout: float=0.1,
        offset_scale: float=1.0,
        bev_feature_shape: Tuple[int, int]=(200, 200),
    ):
        super(TrackFormerDecoderLayer, self).__init__()

        self.num_heads         = num_heads
        self.embed_dim         = embed_dim
        self.num_ref_points    = num_ref_points
        self.dim_feedforward   = dim_feedforward
        self.dropout           = dropout
        self.offset_scale      = offset_scale
        self.bev_feature_shape = bev_feature_shape

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
        self.feedforward_module  = nn.Sequential(
            nn.Linear(self.embed_dim, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.embed_dim),
        )
        self.addnorm3            = AddNorm(input_dim=self.embed_dim)

    def forward(
            self, 
            queries: torch.Tensor,
            bev_features: torch.Tensor,
            ref_points: torch.Tensor, 
        ) -> torch.Tensor:
        """
        :queries: (N, num_queries, det_embeds) list of tensors of shape

        :bev_features: (N, W_bev * H_bev, (C_bev or embed_dim))

        :ref_points: (N, num_queries, 1, 2), reference points for the deformable attention
        """
        H_bev, W_bev = self.bev_feature_shape
        assert bev_features.shape[1] == H_bev * W_bev
        assert bev_features.shape[2] == queries.shape[2] and bev_features.shape[2] == self.embed_dim

        out1              = self.self_attention(queries, queries, queries)
        out2              = self.addnorm1(queries, out1)
        bev_spatial_shape = torch.LongTensor([[H_bev, W_bev]], device=queries.device)
        out3              = self.deform_attention(out2, ref_points, bev_features, bev_spatial_shape)
        out4              = self.addnorm2(out2, out3)
        out5              = self.feedforward_module(out4)
        out6              = self.addnorm3(out4, out5)
        return out6


class TrackFormer(nn.Module):
    def __init__(
            self,
            num_heads: int, 
            embed_dim: int,
            num_layers: int,
            num_obj_classes: int,
            num_ref_points: int=4,
            dim_feedforward: int=512, 
            dropout: float=0.1,
            offset_scale: float=1.0,
            max_objs: int=100,
            learnable_pe: bool=True,
            bev_feature_shape: Tuple[int, int]=(200, 200),
            track_threshold: float=0.5,
            det_3d: bool=True
        ):
        super(TrackFormer, self).__init__()

        self.num_heads         = num_heads
        self.embed_dim         = embed_dim
        self.num_layers        = num_layers
        self.num_obj_classes   = num_obj_classes
        self.num_ref_points    = num_ref_points
        self.dim_feedforward   = dim_feedforward
        self.dropout           = dropout
        self.offset_scale      = offset_scale
        self.max_objs          = max_objs
        self.learnable_pe      = learnable_pe
        self.bev_feature_shape = bev_feature_shape
        self.track_threshold   = track_threshold
        self.det_3d            = det_3d

        self.detection_pos_emb  = PosEmbedding1D(
            self.max_objs, 
            embed_dim=self.embed_dim, 
            learnable=learnable_pe
        )
        self.decoder_modules    = self._create_decoder_layers()
        self.detection_module   = DetectionHead(
            embed_dim=self.embed_dim, 
            num_obj_class=self.num_obj_classes, 
            det_3d=self.det_3d,
            num_seg_coefs=None
        )

    def _create_decoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([TrackFormerDecoderLayer(
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
            bev_features: torch.Tensor, 
            track_queries: Optional[torch.Tensor]=None,
            track_queries_mask: Optional[torch.Tensor]=None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:

        """
        :bev_features: (N, H_bev*W_bev, (C_bev or embed_dim)), BEV features from the BevFormer encoder

        :track_queries: (N, max_objs, embed_dim), embedding output of TrackFormer decoder at previous timestep (t-1)

        :track_queries_mask: (N, max_objs) or (N, max_objs, 1), mask of valid track queries. This will be used to 
                            replace initialized detection queries at t > 0 timesteps with valid track queries
                            (track queries with class scores greater than some threshold)
        """
        is_track_queries      = track_queries is not None
        is_track_queries_mask = track_queries_mask is not None
        assert (
            (is_track_queries and is_track_queries_mask) 
            or ((not is_track_queries) and (not is_track_queries))
        )
        assert bev_features.shape[-1] == self.embed_dim

        batch_size        = bev_features.shape[0]
        ref_points        = DeformableAttention.generate_standard_ref_points(
            self.bev_feature_shape, 
            batch_size=batch_size, 
            device=bev_features.device, 
            normalize=False, 
            n_sample=self.max_objs
        )
        ref_points        = ref_points.unsqueeze(dim=-2)
        detection_queries = self.detection_pos_emb()
        detection_queries = detection_queries.tile(batch_size, 1, 1)
        
        # the idea behind this if-else statement is quite interesting, there are two kinds of queries used
        # in this module, the detection and the track queries, the former is used for identifying new objects
        # that just entered the scene, and the latter is for re-identifying and tracking already detected objects.
        # For spatiotemporal input (like a sequence of frames or a video) processing, at the first timestep 
        # (t = 0) there are no track queries, and the detection queries is required to be capable of detecting
        # new objects in the scene, at subsequent timesteps (t > 0), the output (specfically the embedding)
        # of the last layer of the trackformer decoder module will serve as the query for already detected objects
        # that are to be tracked, as long as the track queries represent valid objects in the scene.
        # Here a given track query represents a valid object if after being projected by the final detection head
        # (MLP), the classification score is above a certain stipulated threshold (hyperparameter). 
        # If there are new detections after (t = 0), then the remaining initialized detection queries (not already
        # replaced by track queries) will be responsible for detecting those, hence it is crucial to ensure that
        # value for `max_objs` should be sufficient enough to accomodate for both the detection queries and the 
        # track queries.
        if track_queries is not None:
            assert track_queries.shape[-1] == self.embed_dim
            assert (
                track_queries_mask.ndim == 2 
                or (track_queries_mask.ndim == 3 and track_queries_mask.shape[2] == 1)
            )
            if track_queries_mask.ndim == 2:
                track_queries_mask = track_queries_mask[..., None]
            queries = torch.where(track_queries_mask, track_queries, detection_queries)
        else:
            queries = detection_queries

        output = queries
        
        layer_results = []
        for decoder_idx in range(0, len(self.decoder_modules)):
            output = self.decoder_modules[decoder_idx](
                queries=output + queries,
                bev_features=bev_features,
                ref_points=ref_points
            )
            layer_results.append(self.detection_module(output))
            
        layer_results = torch.stack(layer_results, dim=0)
        detections    = layer_results[-1]
        track_mask    = detections[..., 0] >= self.track_threshold
        return output, detections, track_mask, layer_results