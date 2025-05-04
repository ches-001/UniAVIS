import torch
import torch.nn as nn
from .trackformer import TrackFormer
from .common import ProtoSegModule, DetectionHead
from typing import *


class MapFormer(TrackFormer):

    # This is not a standard panoptic segformer like the one in this Segformer paper (https://arxiv.org/pdf/2109.03814),
    # this is a custom designed architecture that combines deformable DETR decoder (https://arxiv.org/pdf/2010.04159)
    # with YOLACT (https://arxiv.org/pdf/1904.02689) for instance segmentation.

    def __init__(self, *args, num_seg_coeffs: int=32, seg_c_h: int=256, **kwargs):
        super(MapFormer, self).__init__(*args, **kwargs)

        assert num_seg_coeffs > 0
        
        self.num_seg_coeffs = num_seg_coeffs
        self.seg_c_h        = seg_c_h

        self.proto_seg_module = ProtoSegModule(
            in_channels=self.embed_dim, 
            out_channels=num_seg_coeffs, 
            c_h=seg_c_h, 
        )

        self.detection_module = DetectionHead(
            embed_dim=self.embed_dim, 
            num_classes=self.num_classes, 
            det_3d=self.det_3d,
            num_seg_coefs=self.num_seg_coeffs
        )

    def forward(
            self, 
            bev_features: torch.Tensor, 
            track_queries: Optional[torch.Tensor]=None,
            track_queries_mask: Optional[torch.BoolTensor]=None,
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        """
        Input
        --------------------------------
        :bev_features: (N, H_bev*W_bev, (C_bev or embed_dim)), BEV features from the BevFormer encoder

        :track_queries: (N, num_queries, embed_dim), embedding output of TrackFormer or MapFormer decoder at previous 
                        timestep (t-1)

        :track_queries_mask: (N, num_queries), boolean mask, indicating the tracks with valid and invalid detections
                            NOTE: Detection is invalid if score <= track_threshold (1 if valid, else 0)

        Returns
        --------------------------------
        if training:
            :queries: (N, num_queries, embed_dim) batch of output context query for each segmented item
                        (including invalid detections)

            :layers_detections: (num_layers, N, num_queries, embed_dim), output context query of each layer

            :layer_seg_masks: (num_layers, N, num_queries, H_bev, W_bev), batch of multi-item segmentations of each layer

        else:
            :detections: (N, num_queries, embed_dim), output context query of laast layer

            :layer_seg_masks: (num_layers, N, num_queries, H_bev, W_bev), batch of multi-item segmentations of last layer          
        """
        batch_size = bev_features.shape[0]

        protos = self.proto_seg_module(
            bev_features.permute(0, 2, 1).reshape(batch_size, self.embed_dim, *self.bev_feature_hw)
        )

        if self.training:
            queries, layers_detections = super(MapFormer, self).forward(bev_features, track_queries, track_queries_mask)

            coef_dim_trunc    = layers_detections.shape[-1] - self.num_seg_coeffs
            layers_mask_coefs = layers_detections[..., coef_dim_trunc:]
            layers_detections = layers_detections[..., :coef_dim_trunc]
            layer_seg_masks   = torch.einsum("lnast,tnshw->lnahw", layers_mask_coefs[..., None], protos[None])
            return queries, layers_detections, layer_seg_masks
        
        else:
            detections = super(MapFormer, self).forward(bev_features, track_queries, track_queries_mask)
            coef_dim_trunc = detections.shape[-1] - self.num_seg_coeffs
            mask_coefs     = detections[..., coef_dim_trunc:]
            detections     = detections[..., :coef_dim_trunc]
            seg_masks      = torch.einsum("nast,nshw->nahw", mask_coefs[..., None], protos)
            return detections, seg_masks