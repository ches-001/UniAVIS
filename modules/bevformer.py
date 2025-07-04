import torch
import torch.nn as nn
from torchvision.models import resnet
from .base import BaseFormer
from .backbone import ResNetBackBone
from .attentions import (
    DeformableAttention, 
    MultiViewDeformableAttention, 
    TemporalSelfAttention, 
    SpatialCrossAttention
)
from .common import AddNorm, PosEmbedding2D, SimpleMLP
from typing import Optional, Tuple, Type, Union
    

class BEVFormerEncoderLayer(nn.Module):
    def __init__(
            self, 
            num_heads: int, 
            embed_dim: int,
            num_ref_points: int=4, 
            num_z_ref_points: int=4,
            dim_feedforward: int=512,
            dropout: float=0.1,
            offset_scale: float=1.0,
            num_views: int=6,
            num_fmap_levels: int=4,
        ):
        super(BEVFormerEncoderLayer, self).__init__()

        self.num_heads        = num_heads
        self.embed_dim        = embed_dim
        self.num_ref_points   = num_ref_points
        self.num_z_ref_points = num_z_ref_points
        self.dim_feedforward  = dim_feedforward
        self.offset_scale     = offset_scale
        self.num_views        = num_views
        self.num_fmap_levels  = num_fmap_levels

        self.temporal_self_attn = DeformableAttention(
            num_heads=self.num_heads, 
            embed_dim=self.embed_dim, 
            num_ref_points=self.num_ref_points, 
            dropout=dropout, 
            offset_scale=self.offset_scale,
            num_fmap_levels=1,
            concat_vq_for_offset=True
        )
        self.add_norm1          = AddNorm(self.embed_dim)
        self.spatial_cross_attn = MultiViewDeformableAttention(
            num_heads=self.num_heads, 
            embed_dim=self.embed_dim, 
            num_ref_points=self.num_ref_points, 
            num_z_ref_points=self.num_z_ref_points,
            dropout=dropout, 
            offset_scale=self.offset_scale,
            num_views=self.num_views,
            num_fmap_levels=self.num_fmap_levels,
            concat_vq_for_offset=False
        )
        self.add_norm2          = AddNorm(self.embed_dim)
        self.mlp                = SimpleMLP(self.embed_dim, self.embed_dim, self.dim_feedforward)
        self.add_norm3          = AddNorm(self.embed_dim)

    def forward(
            self,
            bev_queries: torch.Tensor,
            bev_spatial_shape: torch.LongTensor,
            multiscale_fmaps: torch.Tensor,
            sca_ref_points: torch.Tensor,
            sca_attention_mask: torch.Tensor,
            multiscale_fmap_shapes: torch.Tensor,
            transition_matrices: torch.Tensor,
            tsa_ref_points: torch.Tensor,
            bev_histories: Optional[torch.Tensor]=None,
        ) -> torch.Tensor:

        """
        Input
        --------------------------------
        :bev_queries: (N, H_bev * W_bev, C_bev), BEV queries

        :bev_spatial_shape: (1, 2) shape of each spatial feature map [[H_bev, W_bev]]

        :multiscale_fmaps:  (N, num_views, \sum{i=0}^{L-1} H_i \cdot W_i, C), reshaped and concatenated (along dim=1)
                        feature maps from different pyramid levels / scales

        :sca_ref_points: (N, query_len, num_views, L, z_refs, 2) for reference points for 
            value (multiscale feature maps) eg: [[x0, y0], ...[xn, yn]]

        :sca_attention_mask: For boolean masks, (0 if not to attend to, else 1) if mask is float type 
            with continuous values it is directly multiplied with the attention weights

        :multiscale_fmap_shapes: (L, 2) shape of each spatial feature across levels [[H0, W0], ...[Hn, Wn]]

        :transition_matrices: (N, 3, 3), Ego vehicle Motion matrix that transitions the vehicle position at t-1 to t

        :tsa_ref_points: (N, query_len, 1, 2)

        :bev_histories: (N, H_bev * W_bev, C_bev), BEV features from previous timestep t-1

        Returns
        --------------------------------
        :output: (N, H_bev * W_bev, C_bev), output BEV queries to be fed into the next layer
        """

        if bev_histories is None:
            bev_histories = bev_queries

        out1 = self.temporal_self_attn(
            queries=bev_queries, 
            ref_points=tsa_ref_points, 
            value=bev_queries,
            value_spatial_shapes=bev_spatial_shape,
            normalize_ref_points=False 
        )

        out2 = self.add_norm1(out1, bev_queries)

        out3 = self.spatial_cross_attn(
            queries=bev_queries,
            ref_points=sca_ref_points,
            value=multiscale_fmaps,
            value_spatial_shapes=multiscale_fmap_shapes,
            attention_mask=sca_attention_mask,
            normalize_ref_points=False
        )

        out4 = self.add_norm2(out3, out2)
        out5 = self.mlp(out4)
        out6 = self.add_norm3(out5, out4)
        return out6


class BEVFormer(BaseFormer):
    def __init__(
            self, 
            in_img_channels: int=3,
            bb_block: Union[str, Type] = resnet.BasicBlock,
            bb_block_layers: Tuple[int, int, int, int]=(3, 4, 6, 3),
            num_layers: int=6,
            num_heads: int=4,
            embed_dim: int=128, 
            num_ref_points: int=4, 
            dim_feedforward: int=512,
            dropout: float=0.1,
            offset_scale: float=1.0,
            num_z_ref_points: int=4,
            num_views: int=6,
            num_fmap_levels: int=4,
            grid_xy_res: Tuple[float, float]=(0.512, 0.512),
            bev_query_hw: Tuple[int, int]=(200, 200),
            z_ref_range: Tuple[float, float]=(-5.0, 3.0),
            learnable_pe: bool=False
        ):
        
        super(BEVFormer, self).__init__()
    
        self.in_img_channels   = in_img_channels
        self.bb_block          = getattr(resnet, bb_block) if isinstance(bb_block, str) else bb_block
        self.bb_block_layers   = bb_block_layers
        self.num_layers        = num_layers
        self.num_heads         = num_heads
        self.embed_dim         = embed_dim
        self.num_ref_points    = num_ref_points
        self.dim_feedforward   = dim_feedforward
        self.dropout           = dropout
        self.num_z_ref_points  = num_z_ref_points
        self.num_views         = num_views
        self.num_fmap_levels   = num_fmap_levels
        self.grid_xy_res       = grid_xy_res
        self.bev_query_hw      = bev_query_hw
        self.z_ref_range       = z_ref_range
        self.learnable_pe      = learnable_pe
        self.offset_scale      = offset_scale

        self.backbone          = ResNetBackBone(self.in_img_channels, embed_dim, self.bb_block, self.bb_block_layers)
        self.bev_pos_emb       = PosEmbedding2D(*bev_query_hw, embed_dim=self.embed_dim, learnable=self.learnable_pe)
        self.encoder_modules   = self._create_encoder_layers()
        self.register_buffer("z_refs", torch.linspace(*z_ref_range, steps=self.num_z_ref_points))

    def _create_encoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([BEVFormerEncoderLayer(
            num_heads=self.num_heads, 
            embed_dim=self.embed_dim, 
            num_ref_points=self.num_ref_points, 
            num_z_ref_points=self.num_z_ref_points,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            offset_scale=self.offset_scale,
            num_views=self.num_views,
            num_fmap_levels=self.num_fmap_levels,
        ) for _ in range(self.num_layers)])
    

    def forward(
            self, 
            imgs: torch.Tensor,
            lidar_features: torch.Tensor,
            transition_matrices: torch.Tensor,
            cam_proj_matrices: torch.Tensor,
            bev_histories: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        """
        Input
        --------------------------------
        :imgs: (N, V, C, H, W) batch of multiview images

        :lidar_features: (N, C, H, W) batch of LIDAR features from the PointPillarNet
            NOTE: For your custom design, this is not necessary, in the original BEVFormer paper, BEV queries
            were learned and used in the same way that this lidar_features is used, so you can do that too.

        :transition_matrices: (N, 3, 3), Ego vehicle Motion matrix that transitions the vehicle position at t-1 to t

        :cam_proj_matrices: (V, 3, 4) camera intrinsic matrices for each view, for projecting from real world
                            coordinate to

        :bev_histories: (N, H_bev*W_bev, C_bev), BEV features from previous timestep t-1

        NOTE: Do note that the BEV grid space is merely a discretized global (real world) coordinate space.

        Returns
        --------------------------------
        :output: (N, H_bev * W_bev, C_bev), output BEV features
        """
        batch_size, num_views, C_img, H_img, W_img = imgs.shape
        H_bev, W_bev = self.bev_query_hw
    
        device               = imgs.device
        imgs                 = imgs.reshape(batch_size * num_views, C_img, H_img, W_img)
        multiscale_fmaps     = self.backbone(imgs)

        assert len(multiscale_fmaps) == self.num_fmap_levels
        assert all([fmap.shape[1] == self.embed_dim for fmap in multiscale_fmaps])

        multiscale_fmap_shapes = torch.zeros(self.num_fmap_levels, 2, device=device, dtype=torch.int64)

        for i in range(0, len(multiscale_fmaps)):
            fmap               = multiscale_fmaps[i]
            *_, H_fmap, W_fmap = fmap.shape
            fmap               = fmap.reshape(batch_size, num_views, self.embed_dim, H_fmap * W_fmap)

            multiscale_fmap_shapes[i][0] = H_fmap
            multiscale_fmap_shapes[i][1] = W_fmap
            multiscale_fmaps[i]          = fmap

        multiscale_fmaps = torch.concat(multiscale_fmaps, dim=-1).permute(0, 1, 3, 2)

        # apply positional embedding to BEV queries
        bev_pos_embs = self.bev_pos_emb()
        bev_features = lidar_features + bev_pos_embs
        bev_features = bev_features.permute(0, 2, 3, 1)
        bev_features = bev_features.reshape(batch_size, H_bev * W_bev, self.embed_dim)

        bev_spatial_shape = torch.tensor([self.bev_query_hw], device=device, dtype=torch.int64)

        sca_ref_points, sca_attention_mask = SpatialCrossAttention.generate_sca_ref_points_and_attn_mask(
            batch_size=batch_size,
            num_views=self.num_views,
            grid_xy_res=self.grid_xy_res,
            bev_spatial_shape=bev_spatial_shape,
            multiscale_fmap_shapes=multiscale_fmap_shapes,
            z_refs=self.z_refs,
            cam_proj_matrices=cam_proj_matrices,
            device=device
        )

        tsa_ref_points = TemporalSelfAttention.generate_standard_ref_points(
            self.bev_query_hw, 
            batch_size=batch_size, 
            device=device, normalize=True
        )
        tsa_ref_points = tsa_ref_points[..., None, :]

        if bev_histories is not None:
            bev_histories = TemporalSelfAttention.align_bev_histories(
                bev_histories, 
                grid_xy_res=self.grid_xy_res,
                bev_spatial_shape=bev_spatial_shape,
                transition_matrices=transition_matrices,
                device=device
            )

        for encoder_idx in range(0, len(self.encoder_modules)):
            bev_features = self.encoder_modules[encoder_idx](
                bev_queries=bev_features, 
                bev_spatial_shape=bev_spatial_shape,
                multiscale_fmaps=multiscale_fmaps,
                sca_ref_points=sca_ref_points,
                sca_attention_mask=sca_attention_mask,
                multiscale_fmap_shapes=multiscale_fmap_shapes,
                transition_matrices=transition_matrices,
                tsa_ref_points=tsa_ref_points,
                bev_histories=bev_histories
            )
            
        return bev_features