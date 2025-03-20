import torch
import torch.nn as nn
from .trackformer import TrackFormer
from .common import ConvBNorm, DetectionHead
from typing import *

class ProtoSegModule(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int=32, 
            c_h: int=256, 
            upsample_mode: str="bilinear"
        ):
        super(ProtoSegModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvBNorm(self.in_channels, c_h, kernel_size=3)
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.conv2 = ConvBNorm(c_h, c_h, kernel_size=3)
        self.conv3 = ConvBNorm(c_h, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.upsample(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class MapFormer(TrackFormer):

    # This is not a standard panoptic segformer like the one in this Segformer paper (https://arxiv.org/pdf/2109.03814),
    # this is a custom designed architecture that combines deformable DETR decoder (https://arxiv.org/pdf/2010.04159)
    # with YOLACT (https://arxiv.org/pdf/1904.02689) for instance segmentation.

    def __init__(self, *args, num_seg_coeffs: int=32, seg_c_h: int=256, upsample_mode: str="bilinear", **kwargs):
        super(MapFormer, self).__init__(*args, **kwargs)

        assert num_seg_coeffs > 0
        
        self.num_seg_coeffs = num_seg_coeffs
        self.seg_c_h        = seg_c_h
        self.upsample_mode  = upsample_mode

        self.proto_seg_module = ProtoSegModule(
            self.embed_dim, 
            out_channels=num_seg_coeffs, 
            c_h=seg_c_h, 
            upsample_mode=upsample_mode
        )

        self.detection_module = DetectionHead(
            embed_dim=self.embed_dim, 
            num_obj_class=self.num_obj_classes, 
            det_3d=self.det_3d,
            num_seg_coefs=self.num_seg_coeffs
        )

    def forward(
            self, 
            bev_features: torch.Tensor, 
            track_queries: Optional[torch.Tensor]=None,
            track_queries_mask: Optional[torch.Tensor]=None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:

        """
        :bev_features: (N, H_bev*W_bev, (C_bev or embed_dim)), BEV features from the BevFormer encoder

        :track_queries: (N, max_objs, embed_dim), embedding output of TrackFormer decoder at previous timestep (t-1)

        :track_queries_mask: (N, max_objs) or (N, max_objs, 1), mask of valid track queries. This will be used to 
                            replace initialized detection queries at t > 0 timesteps with valid track queries
                            (track queries with class scores greater than some threshold)
        """
        batch_size = bev_features.shape[0]

        output, detections, track_mask, layer_results = super(MapFormer, self).forward(
            bev_features, track_queries, track_queries_mask
        )

        bev_features = bev_features.permute(0, 2, 1).reshape(batch_size, self.num_seg_coeffs, *self.bev_feature_shape)
        protos       = self.proto_seg_module(bev_features)
        return output, detections, protos, track_mask, layer_results