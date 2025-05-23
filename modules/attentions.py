import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()
        
    def forward(
            self, 
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            padding_mask: Optional[torch.BoolTensor]=None,
            attention_mask: Optional[torch.Tensor]=None,
        ) -> torch.Tensor:
        """
        Input
        --------------------------------
        :Q: (N, ..., query_len, embed_dim), attention queries, where N = batch_size
        
        :K: (N, ..., key_len, embed_dim) or (N, num_head, key_len, embed_dim), attention keys

        :V: (N, ..., value_len, embed_dim), attention values

        :padding_mask: (N, ..., query_len), padding mask (0 if padding, else 1)

        :attention_mask: (N, ..., query_len, key_len), attention mask (0 if not attended to, else 1)

        Returns
        --------------------------------
        :output: (N, query_len, embed_dim)
        """
        assert Q.ndim == K.ndim and K.ndim == V.ndim

        K_dims = list(range(0, K.ndim))
        K_T    = K.permute(*(K_dims[:-2] + [K_dims[-1], K_dims[-2]]))
        attn   = torch.matmul(Q, K_T) / math.sqrt(K.shape[-1])

        if torch.is_tensor(padding_mask):
            padding_mask = padding_mask[..., None]
            assert attn.ndim == padding_mask.ndim
            attn = attn.masked_fill(~padding_mask, -torch.inf)
            
        if torch.is_tensor(attention_mask):
            assert attn.ndim == attention_mask.ndim
            if isinstance(attention_mask, torch.BoolTensor):
                attn = attn.masked_fill(~attention_mask, -torch.inf)
            else:
                attn = attn * attention_mask
        
        attn        = F.softmax(attn, dim=-1)
        output      = torch.matmul(attn, V)
        return output
    

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, dropout: float=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by n_head"
        
        self.num_heads   = num_heads
        self.embed_dim = embed_dim
        self.dropout   = dropout
        self.head_dim  = self.embed_dim // self.num_heads
                
        self.Q_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.K_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.V_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
        self.attention     = DotProductAttention()
        self.fc            = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, 
                Q: torch.Tensor,
                K: torch.Tensor, 
                V: torch.Tensor,
                padding_mask: Optional[torch.BoolTensor]=None,
                attention_mask: Optional[torch.Tensor]=None,
        ) -> torch.Tensor:

        """
        Input
        --------------------------------
        :Q: (N, query_len, embed_dim), attention queries, where N = batch_size
        
        :K: (N, key_len, embed_dim), attention keys

        :V: (N, value_len, embed_dim), attention values

        :padding_mask: (N, query_len), padding mask (0 if padding, else 1)

        :attention_mask: (N, query_len, key_len), attention mask (0 if not attended to, else 1)

        Returns
        --------------------------------
        :output: (N, query_len, embed_dim)
        """
        
        assert Q.shape[-1] % self.num_heads == 0
        assert K.shape[-1] % self.num_heads == 0
        assert V.shape[-1] % self.num_heads == 0
        
        N, _, _ = Q.shape
        
        Q = self.Q_fc(Q)
        K = self.K_fc(K)
        V = self.V_fc(V)
        
        Q = Q.reshape(N, Q.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(N, K.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(N, V.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        if padding_mask is not None:
            padding_mask   = padding_mask[:, None]

        if attention_mask is not None:
            attention_mask = attention_mask[:, None]

        output = self.attention(Q, K, V, padding_mask, attention_mask)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(N, -1, self.embed_dim)
        
        output = self.fc(output)
        output = self.dropout_layer(output)
        return output
    

class DeformableAttention(nn.Module):
    def __init__(
            self, 
            num_heads: int, 
            embed_dim: int, 
            num_ref_points: int=4, 
            dropout: float=0.1, 
            offset_scale: float=1.0,
            num_fmap_levels: int=4,
            concat_vq_for_offset: bool=False
        ):
        super(DeformableAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by n_head"

        self.num_heads            = num_heads
        self.embed_dim            = embed_dim
        self.num_ref_points       = num_ref_points
        self.dropout              = dropout
        self.offset_scale         = offset_scale
        self.head_dim             = self.embed_dim // self.num_heads
        self.num_fmap_levels      = num_fmap_levels
        self.concat_vq_for_offset = concat_vq_for_offset
        offset_fc_in_dim          = self.embed_dim * (int(self.concat_vq_for_offset) + 1)
        
        self.V_fc       = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.offsets_fc = nn.Sequential(
            nn.Linear(offset_fc_in_dim, self.num_heads * self.num_fmap_levels * self.num_ref_points * 2),
            nn.Tanh()
        )
        self.attn_fc    = nn.Linear(self.embed_dim, self.num_heads * self.num_fmap_levels * self.num_ref_points)
        self.out_fc     = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(
            self, 
            queries: torch.Tensor, 
            ref_points: torch.Tensor,
            value: torch.Tensor, 
            value_spatial_shapes: torch.LongTensor,
            attention_mask: Optional[torch.Tensor]=None,
            normalize_ref_points: bool=False,
        ) -> torch.Tensor:

        """
        Input
        --------------------------------
        :queries:        (N, query_len, embed_dim) where N = batch_size

        :ref_points:     (N, query_len, L, 2) for reference points for value (multiscale feature maps) 
                         eg: [[x0, y0], ..., [xn, yn]], NOTE: this will further be normalized to be within range [-1, 1]
                         if normalized_ref_points is True, else the function assumes the ref points are already normalized

        :value:          (N, \sum{i=0}^{L-1} H_i \cdot W_i, C), reshaped and concatenated (along dim=1) feature maps
                         from different pyramid levels / scales (L = num_fmap_levels)

        :value_spatial_shapes: (L, 2) shape of each spatial feature across levels [[H0, W0], ...[Hn, Wn]]

        :attention_mask: (N, query_len, L), attention mask (0 if to ignore, else 1)

        :normalize_ref_points: bool, normalize reference points to fall within range [-1, 1], if set to False, ensure that
                               reference points are already normalized

        Returns
        --------------------------------
        :output: (N, query_len, embed_dim)
        """
        assert value_spatial_shapes.shape[0] == self.num_fmap_levels 
        assert ref_points.shape[1] == queries.shape[1] 
        assert ref_points.shape[2] == self.num_fmap_levels 
        assert queries.shape[-1] == self.embed_dim

        batch_size, query_len, _ = queries.shape
        _, value_len, _          = value.shape

        if not self.concat_vq_for_offset:
            offsets = self.offsets_fc(queries)
        else:
            assert value.shape[0] == queries.shape[0] and value.shape[1] == queries.shape[1]
            offsets = self.offsets_fc(torch.concat([queries, value], dim=-1))
            
        value = self.V_fc(value).reshape(batch_size, value_len, self.num_heads, -1)

        # V shape: (batch_size, num_heads, embed_dim // num_head, value_len)
        value = value.permute(0, 2, 3, 1)

        offsets     = offsets.reshape(batch_size, query_len, self.num_heads, self.num_fmap_levels, self.num_ref_points, 2)
        offsets     = self.offset_scale * offsets
        attn        = self.attn_fc(queries)
        attn        = attn.reshape(batch_size, query_len, self.num_heads, self.num_fmap_levels * self.num_ref_points)
        attn        = F.softmax(attn, dim=-1)
        attn        = attn.reshape(batch_size, query_len, self.num_heads, self.num_fmap_levels, self.num_ref_points)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, None, :, None]
            attn = attn.masked_fill(~attention_mask, value=0)
            
        # attn shape: (N, num_heads, num_fmap_levels, query_len, num_ref_points)
        attn        = attn.permute(0, 2, 3, 1, 4)
        
        if normalize_ref_points:
            max_xy      = value_spatial_shapes.flip(dim=-1) - 1
            ref_points  = 2 * (ref_points / max_xy[None, None, ...]) - 1

        sample_locs = (ref_points[:, :, None, :, None, :] + offsets).clamp(min=-1, max=1)

        # sample_locs shape: (N, num_heads, num_fmap_levels, query_len, num_ref_points, 2)
        sample_locs = sample_locs.permute(0, 2, 3, 1, 4, 5)

        value       = value.reshape(-1, *value.shape[2:])
        sample_locs = sample_locs.reshape(-1, *sample_locs.shape[2:])
        attn        = attn.reshape(-1, *attn.shape[2:])
        output      = []

        value_spatial_indexes = torch.tensor(
            [0] + [(shape[0] * shape[1]) for shape in value_spatial_shapes], 
            dtype=torch.int64, 
            device=queries.device
        ).cumsum(dim=-1)
        
        for (lvl, lvl_start) in enumerate(value_spatial_indexes[:-1]):
            level_shape = value_spatial_shapes[lvl]
            sample_loc  = sample_locs[:, lvl, ...]
            lvl_end     = value_spatial_indexes[lvl+1]
            fmap        = value[..., lvl_start:lvl_end]
            
            _, head_dim, lvl_size = fmap.shape

            assert lvl_size == level_shape[0] * level_shape[1]
            fmap             = fmap.reshape(batch_size*self.num_heads, head_dim, *level_shape)
            sampled_features = F.grid_sample(fmap, sample_loc, mode="bilinear", align_corners=True, padding_mode="zeros")
            output.append(sampled_features)

        output = torch.stack(output, dim=1) * attn[:, :, None]
        output = output.sum(dim=1)
        output = output.reshape(batch_size, self.embed_dim, query_len, self.num_ref_points)
        output = output.permute(0, 2, 1, 3).sum(dim=-1)
        output = self.out_fc(output)

        return output
    
    @staticmethod
    def generate_standard_ref_points(
            fmap_hw: Tuple[int, int], 
            batch_size: int=1, 
            device: Union[str, int]="cpu",
            normalize: bool=True,
            n_sample: Optional[int]=None,
        ) -> torch.Tensor:
        
        xindex       = torch.arange(fmap_hw[1], device=device)
        yindex       = torch.arange(fmap_hw[0], device=device)
        ref_points   = torch.stack(torch.meshgrid([yindex, xindex], indexing="ij"), dim=-1)
        ref_points   = ref_points.reshape(-1, 2)

        if n_sample is not None:
            indexes    = torch.linspace(
                0, (fmap_hw[0] * fmap_hw[1]) - 1, steps=n_sample, dtype=torch.int64, device=device
            )
            ref_points = ref_points[indexes]
        if normalize:
            max_xy     = torch.tensor([fmap_hw[1], fmap_hw[0]], device=device)[None, :] - 1
            ref_points = 2 * (ref_points / max_xy) - 1

        ref_points = ref_points[None].tile(batch_size, 1, 1)
        return ref_points
    

class MultiView3DDeformableAttention(DeformableAttention):
    def __init__(
            self, 
            num_heads: int, 
            embed_dim: int, 
            num_ref_points: int=4, 
            num_z_ref_points: int=4,
            dropout: float=0.1, 
            offset_scale: float=1.0,
            num_views: int=6,
            num_fmap_levels: int=4,
            concat_vq_for_offset: bool=False
    ):
        super(MultiView3DDeformableAttention, self).__init__( 
            num_heads=num_heads, 
            embed_dim=embed_dim, 
            num_ref_points=num_ref_points, 
            dropout=dropout, 
            offset_scale=offset_scale,
            num_fmap_levels=num_fmap_levels,
            concat_vq_for_offset=concat_vq_for_offset
        )

        self.num_views        = num_views
        self.num_z_ref_points = num_z_ref_points
        _attn_out             = (
            self.num_heads
            * self.num_views  
            * self.num_fmap_levels 
            * self.num_ref_points 
            * self.num_z_ref_points
        )
        self.offsets_fc       = nn.Sequential(
            nn.Linear(self.embed_dim, _attn_out * 2),
            nn.Tanh()
        )
        self.attn_fc          = nn.Linear(self.embed_dim, _attn_out)

    def forward(
            self, 
            queries: torch.Tensor, 
            ref_points: torch.Tensor,
            value: torch.Tensor, 
            value_spatial_shapes: torch.LongTensor,
            attention_mask: Optional[torch.Tensor]=None,
            normalize_ref_points: bool=False
        ) -> torch.Tensor:
        """
        Input
        --------------------------------
        :queries:        (N, query_len, embed_dim) where N = batch_size

        :ref_points:    (N, query_len, num_views, L, z_refs, 2) for reference points for 
                        value (multiscale feature maps) eg: [[x0, y0], ...[xn, yn]], NOTE: this will further
                        be normalized to be within range [-1, 1] if normalized_ref_points is True, else the 
                        function assumes the ref points are already normalized

        :value:         (N, num_views, \sum{i=0}^{L-1} H_i \cdot W_i, C), reshaped and concatenated (along dim=1)
                        feature maps from different pyramid levels / scales (L = num_fmap_levels)

        :value_spatial_shapes: (L, 2) shape of each spatial feature across levels [[H0, W0], ...[Hn, Wn]]

        :attention_mask: (N, query_len, num_views, L, z_refs), attention mask (0 if to ignore, else 1)

        :normalize_ref_points: bool, normalize reference points to fall within range [-1, 1], if set to False, ensure that
                               reference points are already normalized

        Returns
        --------------------------------
        :output: (N, query_len, embed_dim), output BEV features / queries
        """
        assert value_spatial_shapes.shape[0] == self.num_fmap_levels 
        assert value.shape[1] == self.num_views
        assert ref_points.shape[1] == queries.shape[1] 
        assert ref_points.shape[3] == self.num_fmap_levels 
        assert ref_points.shape[4] == self.num_z_ref_points
        assert queries.shape[2] == self.embed_dim

        batch_size, query_len, _   = queries.shape
        _, num_views, value_len, _ = value.shape
        all_points                 = self.num_ref_points * self.num_z_ref_points

        if not self.concat_vq_for_offset:
            offsets = self.offsets_fc(queries)
        else:
            assert value.shape[0] == queries.shape[0] and value.shape[1] == queries.shape[1]
            offsets = self.offsets_fc(torch.concat([queries, value], dim=-1))

        value = self.V_fc(value).reshape(batch_size, num_views, value_len, self.num_heads, -1)

        # V shape: (batch_size, num_heads, num_views, num_embed // num_head, value_len)
        value = value.permute(0, 3, 1, 4, 2)

        offsets     = offsets.reshape(
            batch_size, 
            query_len, 
            self.num_heads, 
            self.num_views, 
            self.num_fmap_levels, 
            self.num_ref_points, 
            self.num_z_ref_points, 
            2
        )
        offsets     = self.offset_scale * offsets
        attn        = self.attn_fc(queries)
        attn        = attn.reshape(
            batch_size, 
            query_len, 
            self.num_heads, 
            self.num_views,
            self.num_fmap_levels * self.num_ref_points * self.num_z_ref_points
        )
        attn        = F.softmax(attn, dim=-1)
        attn        = attn.reshape(
            batch_size, 
            query_len, 
            self.num_heads, 
            self.num_views, 
            self.num_fmap_levels, 
            self.num_ref_points, 
            self.num_z_ref_points
        )
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, None, :, :, None, :]
            attn = attn.masked_fill(~attention_mask, value=0)

        # attn shape: (batch_size, num_heads, num_views, num_fmap_levels, query_len, num_ref_points, num_z_ref_points)
        attn        = attn.permute(0, 2, 3, 4, 1, 5, 6)
        
        if normalize_ref_points:
            max_xy     = value_spatial_shapes.flip(dim=-1) - 1
            ref_points = 2 * (ref_points / max_xy[None, None, None, :, None, :]) - 1

        # sample_locs shape: 
        #   (batch_size, num_heads, num_views, num_fmap_levels, query_len, num_ref_points, num_z_ref_points, 2)
        sample_locs = ref_points[:, :, None, :, :, None, :, :] + offsets
        sample_locs = sample_locs.clamp(min=-1, max=1)
        sample_locs = sample_locs.permute(0, 2, 3, 4, 1, 5, 6, 7)

        # reshape for calculations
        value       = value.reshape(-1, *value.shape[3:])
        sample_locs = sample_locs.reshape(-1, *sample_locs.shape[3:-3], all_points, 2)
        attn        = attn.reshape(-1, *attn.shape[3:-2], all_points)
        output      = []

        value_spatial_indexes = torch.tensor(
            [0] + [(shape[0] * shape[1]) for shape in value_spatial_shapes], 
            dtype=torch.int64, 
            device=queries.device
        ).cumsum(dim=-1)
        
        for (lvl, lvl_start) in enumerate(value_spatial_indexes[:-1]):
            level_shape = value_spatial_shapes[lvl]
            sample_loc  = sample_locs[:, lvl, ...]
            lvl_end     = value_spatial_indexes[lvl+1]
            fmap        = value[..., lvl_start:lvl_end]
            
            _, head_dim, lvl_size = fmap.shape

            assert lvl_size == level_shape[0] * level_shape[1]
            fmap             = fmap.reshape(batch_size * self.num_heads * num_views, head_dim, *level_shape)
            sampled_features = F.grid_sample(fmap, sample_loc, mode="bilinear", align_corners=True, padding_mode="zeros")
            output.append(sampled_features)

        output = torch.stack(output, dim=1) * attn[:, :, None]
        output = output.sum(dim=1)
        output = output.reshape(
            batch_size, self.num_heads, num_views, head_dim, query_len, all_points
        ).permute(0, 4, 2, 1, 3, 5)
        output = output.reshape(batch_size, query_len, num_views, self.embed_dim, all_points)
        output = output.sum(dim=(2, -1))
        output = self.out_fc(output)

        return output


class TemporalSelfAttention(DeformableAttention):
    def __init__(
            self, 
            num_heads: int, 
            embed_dim: int, 
            num_ref_points: int=4, 
            dropout: float=0.1, 
            offset_scale: float=1.0
        ):
        
        super(TemporalSelfAttention, self).__init__(
            num_heads=num_heads, 
            embed_dim=embed_dim, 
            num_ref_points=num_ref_points, 
            dropout=dropout, 
            offset_scale=offset_scale, 
            num_fmap_levels=1, 
            concat_vq_for_offset=True
        )

        self.offsets_fc = nn.Linear(self.embed_dim * 2, self.num_heads * self.num_ref_points * 2)

    def forward(
            self, 
            bev_queries: torch.Tensor,
            bev_spatial_shape: torch.LongTensor,
            bev_histories: Optional[torch.Tensor]=None,
            transition_matrices: Optional[torch.Tensor]=None,
        ) -> torch.Tensor:

        """
        Input
        --------------------------------
        :bev_queries:   (N, H_bev * W_bev, C_bev) where N = batch_size

        :bev_spatial_shape: (1, 2) shape of each spatial feature map [[H_bev, W_bev]]

        :bev_histories: (N, H_bev * W_bev, C_bev) BEV features from the previous timestep t-1

        :transition_matrices: (N, 3, 3) the matrix that transitions the ego vehicle from t-1 to t 
                              (in homogeneous coordinates)
                              
        Returns
        --------------------------------
        :output: (N, H_bev * W_bev, C_bev), output BEV features / queries that have been attended to temporally
        """

        batch_size, _, C_bev = bev_queries.shape
        H_bev, W_bev   = bev_spatial_shape[0]
        H_bev          = H_bev.item()
        W_bev          = W_bev.item()
        device         = bev_queries.device

        # ref_points for deformable attention: (batch_size, H_bev * W_bev, 1, 2)
        ref_points  = TemporalSelfAttention.generate_standard_ref_points(
            (H_bev, W_bev), batch_size, device=device, normalize=True
        )
        ref_points = ref_points.unsqueeze(dim=-2)

        # for bev_histories to be None, it implies that we are working with the very first timestep of the
        # simulation. As such, the proposed temporal self attention will simply decay to a standard self
        # attention with deformable attention mechanism.
        if bev_histories is None:
            output = super(TemporalSelfAttention, self).forward(
                bev_queries, 
                ref_points, 
                bev_queries,
                value_spatial_shapes=bev_spatial_shape,
                normalize_ref_points=False 
            )
            return output

        assert transition_matrices is not None

        # The goal of this subsection before the deformable attention section is to align the bev_histories (B{t-1}) to
        # the BEV Query. This is to ensure that we account fo the spatial change that has occured in the newly created
        # BEV feature (B{t}). We get the grid space of B{t-1}, we take this space through an affine transformation by
        # multiplying the grid space by the transition matrix (rotation and translation matrices) that transitioned the
        # ego vehicle from t-1 to t on global real world coordinates that have been discretized on the BEV grid. 
        # We then normalise the grid space to be within the range of [-1, 1], then we sample feature maps from B{t-1} 
        # along the newly aligned grid.
        xindex               = torch.arange(W_bev, device=device)
        yindex               = torch.arange(H_bev, device=device)
        ygrid, xgrid         = torch.meshgrid([yindex, xindex], indexing="ij")
        bev_grid_2d          = torch.stack([xgrid, ygrid], dim=-1).to(dtype=torch.float32, device=device)
        ones                 = torch.ones(*bev_grid_2d.shape[:-1], 1, dtype=bev_grid_2d.dtype, device=device)
        bev_grid_3d          = torch.concat([bev_grid_2d, ones], dim=-1)[None].tile(batch_size, 1, 1, 1)
        aligned_grid         = torch.einsum("nttii,nhwik->nhwik", transition_matrices[:, None, None], bev_grid_3d[..., None])
        aligned_grid         = aligned_grid[..., :2, 0]
        aligned_grid[..., 0] = 2 * (aligned_grid[..., 0] / (W_bev - 1)) - 1
        aligned_grid[..., 1] = 2 * (aligned_grid[..., 1] / (H_bev - 1)) - 1
        bev_histories        = bev_histories.permute(0, 2, 1).reshape(batch_size, C_bev, H_bev, W_bev)
        bev_histories        = F.grid_sample(
            bev_histories, aligned_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        
        bev_histories = bev_histories.permute(0, 2, 3, 1).reshape(batch_size, H_bev * W_bev, C_bev)
        output = super(TemporalSelfAttention, self).forward(
            bev_queries,
            ref_points,
            bev_histories,
            value_spatial_shapes=bev_spatial_shape,
            normalize_ref_points=False
        )
        return output


class SpatialCrossAttention(MultiView3DDeformableAttention):
    def __init__(self, 
            num_heads: int, 
            embed_dim: int, 
            num_ref_points: int=4, 
            num_z_ref_points: int=4,
            dropout: float=0.1, 
            offset_scale: float=1.0,
            num_views: int=6,
            num_fmap_levels: int=4, 
            grid_xy_res: Tuple[float, float]=(0.512, 0.512)
        ):
        # num_z_ref_points: number of z-axis references
        # s: size of resolution of BEV grid in meters
        super(SpatialCrossAttention, self).__init__(
            num_heads=num_heads, 
            embed_dim=embed_dim, 
            num_ref_points=num_ref_points, 
            num_z_ref_points=num_z_ref_points,
            dropout=dropout, 
            offset_scale=offset_scale, 
            num_views=num_views,
            num_fmap_levels=num_fmap_levels, 
            concat_vq_for_offset=False
        )
        self.num_views         = num_views
        self.num_z_ref_points  = num_z_ref_points
        self.grid_xy_res       = grid_xy_res

    def forward(
            self, 
            bev_queries: torch.Tensor, 
            bev_spatial_shape: torch.LongTensor,
            multiscale_fmaps: torch.Tensor,
            multiscale_fmap_shapes: torch.LongTensor,
            img_spatial_shape: torch.LongTensor,
            z_refs: torch.Tensor,
            cam_proj_matrices: torch.Tensor,
        ) -> torch.Tensor:
        
        """
        Input
        --------------------------------
        :bev_queries:    (N, H_bev * W_bev, C_bev) where N = batch_size

        :bev_spatial_shape: (1, 2) shape of each spatial feature map [[H_bev, W_bev]]

        :multiscale_fmaps:  (N, num_views, \sum{i=0}^{L-1} H_i \cdot W_i, C), reshaped and concatenated (along dim=1)
            feature maps from different pyramid levels / scales (L = num_fmap_levels)

        :multiscale_fmap_shapes: (L, 2) shape of each spatial feature across levels [[H0, W0], ...[Hn, Wn]]

        :img_spatial_shape: (1, 2) spatial resolution of image [H, W]

        :z_refs: (num_z_ref_points, ) z-axis reference points

        :cam_proj_matrices: (V, 3, 4) projection matrix for each camera, from real 3D coord (ego vehicle frame) to 3D image
            coord. This matrix is the product of the 3 x 3 (homogenized to 3 x 4) camera intrinsic matrix and the 4 x 4 camera
            camera extrinsic matrix. Do note that the 4 x 4 extrinsic matrix could be 3 x 4 and the intrinsic left as a 3 x 3, 
            but this non-homogeneous and not directly invertible.

        Returns
        --------------------------------
        :output: (N, H_bev * W_bev, C_bev), output BEV features / queries that have been attended to spatially
        """
        assert multiscale_fmap_shapes.shape[0] == self.num_fmap_levels
        assert z_refs.shape[0] == self.num_z_ref_points

        batch_size, *_ = bev_queries.shape
        num_views      = multiscale_fmaps.shape[1]
        H_bev, W_bev   = bev_spatial_shape[0]
        H_bev          = H_bev.item()
        W_bev          = W_bev.item()
        device         = bev_queries.device

        # Create BEV 2D grids pace
        xindex          = torch.arange(W_bev, device=device)
        yindex          = torch.arange(H_bev, device=device)
        ygrid, xgrid    = torch.meshgrid([yindex, xindex], indexing="ij")
        grid_2d         = torch.stack([xgrid, ygrid], dim=-1)
        grid_2d         = grid_2d.to(dtype=torch.float32, device=device)

        # map BEV grid space to real world coordinates: If the BEV grid is a (200 x 200) grid
        # the index along both axes will range from 0 to 199, if the real world coordinates range
        # from -51.2m to 51.2m across both x and y axes, with the grid cell space being (0.512m, 0.512m)
        # across both axes, then we need to ensure that index (0, 0) represents a grid cell whose top left
        # corner is at (-51.2m, -51.2m) in real space and index (199, 199) represents a grid whose bottom 
        # right corner is at (51.2m, 51.2m) in real space. This also means that the real x, y center of cell
        # (0, 0) is (-50.944m, -50.944m) and the center for cell (199, 199) is (-50.944m, 50.944m).
        grid_xy_res     = torch.tensor(self.grid_xy_res, device=device)
        grid_wh         = torch.tensor([W_bev, H_bev], device=device)
        max_grid_xy     = grid_wh - 1
        min_real_xy     = ((-grid_wh / 2) * grid_xy_res) + (grid_xy_res - (grid_xy_res / 2))
        max_real_xy     = -min_real_xy
        grid_2d[..., :2] = (((max_real_xy - min_real_xy) * grid_2d[..., :2]) / max_grid_xy) + min_real_xy

        # Create 3D grid space, this phenomenon is called pillaring, this is where we
        # raise the 2D grid space to 3D pillars given some z-axis reference points
        grid_3d          = torch.zeros(H_bev, W_bev, self.num_z_ref_points, 3, dtype=grid_2d.dtype, device=device)
        grid_3d[..., :2] = grid_2d[:, :, None, :]
        grid_3d[..., 2]  = z_refs[None, None, :]

        # map the 3D (pillared 2D real world grid space) to 2D image space and then to
        # feature map space
        ones    = torch.ones(*grid_3d.shape[:-1], 1, device=device)
        grid_3d = torch.concat([grid_3d, ones], dim=-1)

        # project from 3D real world coord to 2D image coord (proj_2d shape: (num_views, H_bev, W_bev, num_z_ref_points, 2))
        proj_2d = torch.einsum("vtttki,txyzij->vxyzkj", cam_proj_matrices[:, None, None, None, ...], grid_3d[None, ..., None])
        proj_2d = proj_2d[..., :2, 0]

        # since we just need reference points normalized to be within range (-1, 1), we just need to divide ref_points indexes
        # by max img indexes (img_w - 1, img_h - 1), then scale accordingly.
        ref_points = proj_2d.reshape(num_views, H_bev * W_bev, self.num_z_ref_points, 2).permute(1, 0, 2, 3)
        ref_points = 2 * (ref_points[:, :, None, :, :] / (img_spatial_shape[None, None, None] - 1)) - 1
        ref_points = ref_points[None].tile(batch_size, 1, 1, multiscale_fmap_shapes.shape[0], 1, 1)

        # calculate the attention mask
        attention_mask    = (ref_points[..., 0] >= -1) & (ref_points[..., 1] <= 1)

        output = super(SpatialCrossAttention, self).forward(
            queries=bev_queries, 
            ref_points=ref_points, 
            value=multiscale_fmaps, 
            value_spatial_shapes=multiscale_fmap_shapes, 
            attention_mask=attention_mask,
            normalize_ref_points=False
        )
        v_hit  = attention_mask.sum(dim=(2, 3, 4))[..., None]
        return (1 / v_hit) * output
        