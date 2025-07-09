import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from utils.img_utils import transform_points
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction \
        as _MMCVDeformableAttentionFunction
    _MMCV_EXTENSION_IS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _MMCV_EXTENSION_IS_AVAILABLE = False

_CUDA_IS_AVAILABLE = torch.cuda.is_available()


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()
        
    def forward(
            self, 
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            padding_mask: Optional[torch.Tensor]=None,
            attention_mask: Optional[torch.Tensor]=None,
            is_query_mask: bool=True
        ) -> torch.Tensor:
        """
        Input
        --------------------------------
        :Q: (N, ..., query_len, embed_dim), attention queries, where N = batch_size
        
        :K: (N, ..., key_len, embed_dim) or (N, num_head, key_len, embed_dim), attention keys

        :V: (N, ..., value_len, embed_dim), attention values

        :padding_mask: (N, ..., query_len) or (N, ..., key_len), padding mask (0 if padding, else 1).
            see the is_query_mask argument

        :attention_mask: (N, ..., query_len, key_len), attention mask 
            for boolean masks, (0 if not to attend to, else 1) if mask is float type with continuous values
            it is directly multiplied with the attention weights

        :is_query_mask: bool, if True, the padding mask will correspond to the queries, else it will correspond
            to the keys

        Returns
        --------------------------------
        :output: (N, query_len, embed_dim)
        """
        assert Q.ndim == K.ndim and K.ndim == V.ndim

        K_dims = list(range(0, K.ndim))
        K_T    = K.permute(*(K_dims[:-2] + [K_dims[-1], K_dims[-2]]))
        attn   = torch.matmul(Q, K_T) / math.sqrt(K.shape[-1])

        if torch.is_tensor(padding_mask):
            assert padding_mask.dtype == torch.bool

            if is_query_mask:
                if (attn.ndim - padding_mask.ndim) == 1:
                    padding_mask = padding_mask[..., None]
            else:
                if (attn.ndim - padding_mask.ndim) == 1:
                    padding_mask = padding_mask[..., None, :]

            assert attn.ndim == padding_mask.ndim
            attn = attn.masked_fill(~padding_mask, -torch.inf)
            
        if torch.is_tensor(attention_mask):
            assert attn.ndim == attention_mask.ndim

            if attention_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attention_mask, -torch.inf)
            else:
                attn = attn * attention_mask
        
        attn   = F.softmax(attn, dim=-1)
        # address NaN values that could be caused by combination of casual mask and padding mask
        attn   = attn.masked_fill(torch.isnan(attn), 0)
        output = torch.matmul(attn, V)
        return output
    

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, dropout: float=0.1, proj_bias: bool=False):
        super(MultiHeadedAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by n_head"
        
        self.num_heads      = num_heads
        self.embed_dim      = embed_dim
        self.dropout        = dropout
        self.head_dim       = self.embed_dim // self.num_heads
        self.proj_bias      = proj_bias
                
        self.Q_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=self.proj_bias)
        self.K_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=self.proj_bias)
        self.V_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=self.proj_bias)
        
        self.attention     = DotProductAttention()
        self.fc            = nn.Linear(self.embed_dim, self.embed_dim, bias=self.proj_bias)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, 
                Q: torch.Tensor,
                K: torch.Tensor, 
                V: torch.Tensor,
                padding_mask: Optional[torch.Tensor]=None,
                attention_mask: Optional[torch.Tensor]=None,
                is_query_mask: bool=True
        ) -> torch.Tensor:

        """
        Input
        --------------------------------
        :Q: (N, ..., query_len, embed_dim), attention queries, where N = batch_size
        
        :K: (N, ..., key_len, embed_dim), attention keys

        :V: (N, ..., value_len, embed_dim), attention values

        :padding_mask: (N, ..., query_len) | (N, ..., query_len, 1) or (N, ..., key_len) | (N, ..., key_len, 1), 
            padding mask (0 if padding, else 1). See is_query_mask

        :attention_mask: (N, ..., query_len, key_len), For boolean masks, (0 if not to attend to, else 1) 
            if mask is float type with continuous values it is directly multiplied with the attention weights

        :is_query_mask: bool, if True, the padding mask will correspond to the queries, else it will correspond
            to the keys
        Returns
        --------------------------------
        :output: (N, query_len, embed_dim)
        """
        
        assert Q.shape[-1] % self.num_heads == 0
        assert K.shape[-1] % self.num_heads == 0
        assert V.shape[-1] % self.num_heads == 0
        assert Q.ndim == K.ndim and K.ndim == V.ndim
        
        orig_shape = Q.shape
        ndim       = Q.ndim
        
        Q = self.Q_fc(Q)
        K = self.K_fc(K)
        V = self.V_fc(V)
        
        unperm_axes = [i for i in range(0, Q.ndim - 2)]
        Q = Q.reshape(orig_shape[0], *Q.shape[1:-1], self.num_heads, self.head_dim).permute(*unperm_axes, -2, -3, -1)
        K = K.reshape(orig_shape[0], *K.shape[1:-1], self.num_heads, self.head_dim).permute(*unperm_axes, -2, -3, -1)
        V = V.reshape(orig_shape[0], *V.shape[1:-1], self.num_heads, self.head_dim).permute(*unperm_axes, -2, -3, -1)
        
        if padding_mask is not None:
            padding_mask = padding_mask[(slice(None), ) * (ndim - 2) + (None, )]

        if attention_mask is not None:
            attention_mask = attention_mask[(slice(None), ) * (ndim - 2) + (None, )]

        output = self.attention(Q, K, V, padding_mask, attention_mask, is_query_mask=is_query_mask)
        output = output.permute(*unperm_axes, -2, -3, -1)
        output = output.reshape(*orig_shape)
        
        output = self.fc(output)

        if self.fc.bias is not None and padding_mask is not None:
            padding_mask = torch.flatten(padding_mask, start_dim=ndim-2, end_dim=ndim-1)
            if padding_mask.ndim < output.ndim:
                padding_mask = padding_mask[..., None]
            output = torch.masked_fill(output, ~padding_mask, value=0.0)

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
            concat_vq_for_offset: bool=False,
            im2col_steps: int=64,
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
        self.im2col_steps         = im2col_steps
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
            value_spatial_shapes: torch.Tensor,
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

        :attention_mask: (N, query_len, L) For boolean masks, (0 if not to attend to, else 1) if mask is float type 
            with continuous values it is directly multiplied with the attention weights

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

        offsets = offsets.reshape(batch_size, query_len, self.num_heads, self.num_fmap_levels, self.num_ref_points, 2)
        offsets = self.offset_scale * offsets
        attn    = self.attn_fc(queries)
        attn    = attn.reshape(batch_size, query_len, self.num_heads, self.num_fmap_levels * self.num_ref_points)
        attn    = F.softmax(attn, dim=-1)
        attn    = attn.reshape(batch_size, query_len, self.num_heads, self.num_fmap_levels, self.num_ref_points)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, None, :, None]
            if attention_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attention_mask, value=0)
            else:
                attn = attn * attention_mask
        
        if normalize_ref_points:
            max_xy     = value_spatial_shapes.flip(dim=-1) - 1
            ref_points = 2 * (ref_points / max_xy[None, None, ...]) - 1

        sample_locs = torch.clamp(ref_points[:, :, None, :, None, :] + offsets, min=-1, max=1)

        level_start_indexes = value_spatial_shapes.prod(dim=-1)
        level_start_indexes = level_start_indexes.cumsum(dim=-1) - level_start_indexes

        if _MMCV_EXTENSION_IS_AVAILABLE and _CUDA_IS_AVAILABLE:
            sample_locs = (sample_locs + 1) / 2
            output = _MMCVDeformableAttentionFunction.apply(
                value, value_spatial_shapes, level_start_indexes, sample_locs, attn, self.im2col_steps
            )
            output = self.out_fc(output)
            return output

        # value shape: 
        #   (N, value_len, num_heads, embed_dim // num_head) ->
        #   (N, num_heads, embed_dim // num_head, value_len)

        # attn shape: 
        #   (N, query_len, num_heads, num_fmap_levels, num_points) ->
        #   (N, num_heads, num_fmap_levels, query_len, num_points)

        # sample_locs shape: 
        #   (N, query_len, num_heads, num_fmap_levels, num_points, 2) ->
        #   (N, num_heads, num_fmap_levels, query_len, num_points, 2)
        value       = value.permute(0, 2, 3, 1)
        attn        = attn.permute(0, 2, 3, 1, 4)
        sample_locs = sample_locs.permute(0, 2, 3, 1, 4, 5)

        value       = value.reshape(-1, *value.shape[2:])
        attn        = attn.reshape(-1, *attn.shape[2:])
        sample_locs = sample_locs.reshape(-1, *sample_locs.shape[2:])
        output      = None
        
        for (lvl_idx, lvl_start) in enumerate(level_start_indexes):
            level_shape = value_spatial_shapes[lvl_idx]
            sample_loc  = sample_locs[:, lvl_idx, ...]
            lvl_end     = lvl_start + (level_shape[0] * level_shape[1])
            fmap        = value[..., lvl_start:lvl_end]
            
            _, head_dim, lvl_size = fmap.shape

            assert lvl_size == level_shape[0] * level_shape[1]
            fmap             = fmap.reshape(batch_size*self.num_heads, head_dim, *level_shape)
            sampled_features = F.grid_sample(fmap, sample_loc, mode="bilinear", align_corners=True, padding_mode="zeros")
            sampled_features = sampled_features * attn[:, lvl_idx, None]

            if output is None:
                output = sampled_features
            else:
                output = output + sampled_features
        
        output = output.reshape(batch_size, self.embed_dim, query_len, self.num_ref_points)
        output = output.sum(dim=-1).permute(0, 2, 1)
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
        
        xindex     = torch.arange(fmap_hw[1], device=device)
        yindex     = torch.arange(fmap_hw[0], device=device)
        ref_points = torch.stack(torch.meshgrid([yindex, xindex], indexing="ij"), dim=-1)
        ref_points = ref_points.reshape(-1, 2)

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
    

class MultiViewDeformableAttention(DeformableAttention):
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
        super(MultiViewDeformableAttention, self).__init__( 
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
            self.num_views
            * self.num_heads
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
            value_spatial_shapes: torch.Tensor,
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

        :attention_mask: For boolean masks, (0 if not to attend to, else 1) if mask is float type 
            with continuous values it is directly multiplied with the attention weights

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

        offsets = offsets.reshape(
            batch_size, 
            query_len, 
            self.num_views,
            self.num_heads,
            self.num_fmap_levels, 
            self.num_ref_points, 
            self.num_z_ref_points, 
            2
        )
        offsets = self.offset_scale * offsets
        attn    = self.attn_fc(queries)
        attn    = attn.reshape(
            batch_size, 
            query_len, 
            self.num_views,
            self.num_heads, 
            self.num_fmap_levels * self.num_ref_points * self.num_z_ref_points
        )
        attn    = F.softmax(attn, dim=-1)
        attn    = attn.reshape(
            batch_size, 
            query_len, 
            self.num_views, 
            self.num_heads, 
            self.num_fmap_levels, 
            self.num_ref_points, 
            self.num_z_ref_points
        )
        v_hit = None
        if attention_mask is not None:
            v_hit          = attention_mask.sum(dim=(2, 3, 4))[..., None]
            attention_mask = attention_mask[:, :, :, None, :, None, :]
            if attention_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attention_mask, value=0)
            else:
                attn = attn * attention_mask
        
        if normalize_ref_points:
            max_xy     = value_spatial_shapes.flip(dim=-1) - 1
            ref_points = 2 * (ref_points / max_xy[None, None, None, :, None, :]) - 1

        sample_locs = ref_points[:, :, :, None, :, None, :, :] + offsets
        sample_locs = torch.clamp(sample_locs, min=-1, max=1)

        level_start_indexes = value_spatial_shapes.prod(dim=-1)
        level_start_indexes = level_start_indexes.cumsum(dim=-1) - level_start_indexes

        if _MMCV_EXTENSION_IS_AVAILABLE and _CUDA_IS_AVAILABLE:
            value       = value.reshape(-1, *value.shape[2:])
            attn        = attn.permute(0, 2, 1, 3, 4, 5, 6)
            attn        = attn.reshape(-1, *attn.shape[2:-2], all_points).contiguous()
            sample_locs = sample_locs.permute(0, 2, 1, 3, 4, 5, 6, 7)
            sample_locs = sample_locs.reshape(-1, *sample_locs.shape[2:-3], all_points, 2).contiguous()
            sample_locs = (sample_locs + 1) / 2
            output      = _MMCVDeformableAttentionFunction.apply(
                value, value_spatial_shapes, level_start_indexes, sample_locs, attn, self.im2col_steps
            )
            output      = torch.unflatten(output, dim=0, sizes=(batch_size, num_views)).sum(dim=1)
            output      = self.out_fc(output)
            return output

        # value shape: 
        #   (N, num_views, value_len, num_heads, num_embed // num_head) ->
        #   (N, num_views, num_heads, num_embed // num_head, value_len)

        # attn shape: 
        #   (N, query_len, num_views, num_heads, num_fmap_levels, num_ref_points, num_z_ref_points) ->
        #   (N, num_views, num_heads, num_fmap_levels, query_len, num_ref_points, num_z_ref_points)

        # samplwe_locs shape: 
        #   (N, query_len, num_views, num_heads, num_fmap_levels, num_ref_points, num_z_ref_points, 2) ->
        #   (N, num_views, num_heads, num_fmap_levels, query_len, num_ref_points, num_z_ref_points, 2)
        value       = value.permute(0, 1, 3, 4, 2)
        attn        = attn.permute(0, 2, 3, 4, 1, 5, 6)
        sample_locs = sample_locs.permute(0, 2, 3, 4, 1, 5, 6, 7)

        # reshape for calculations
        value       = value.reshape(-1, *value.shape[3:])
        attn        = attn.reshape(-1, *attn.shape[3:-2], all_points)
        sample_locs = sample_locs.reshape(-1, *sample_locs.shape[3:-3], all_points, 2)
        output      = None
        
        for (lvl_idx, lvl_start) in enumerate(level_start_indexes):
            level_shape = value_spatial_shapes[lvl_idx]
            sample_loc  = sample_locs[:, lvl_idx, ...]
            lvl_end     = lvl_start + (level_shape[0] * level_shape[1])
            fmap        = value[..., lvl_start:lvl_end]
            
            _, head_dim, lvl_size = fmap.shape

            assert lvl_size == level_shape[0] * level_shape[1]
            fmap             = fmap.reshape(batch_size * self.num_heads * num_views, head_dim, *level_shape)
            sampled_features = F.grid_sample(fmap, sample_loc, mode="bilinear", align_corners=True, padding_mode="zeros")
            sampled_features = sampled_features * attn[:, lvl_idx, None]
            if output is None:
                output = sampled_features
            else:
                output = output + sampled_features

        output = output.reshape(batch_size, num_views, self.num_heads, head_dim, query_len, all_points)
        output = output.sum(dim=(1, -1)).reshape(batch_size, self.embed_dim, query_len)
        output = output.permute(0, 2, 1)
        output = self.out_fc(output)

        if v_hit is not None:
            output = (1 / v_hit) * output

        return output

class TemporalSelfAttention(DeformableAttention):
    def __init__(
            self, 
            num_heads: int, 
            embed_dim: int, 
            num_ref_points: int=4, 
            dropout: float=0.1, 
            offset_scale: float=1.0,
            grid_xy_res: Tuple[float, float]=(0.512, 0.512)
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
        self.grid_xy_res = grid_xy_res
        self.offsets_fc  = nn.Linear(self.embed_dim * 2, self.num_heads * self.num_ref_points * 2)

    def forward(
            self, 
            bev_queries: torch.Tensor,
            bev_spatial_shape: torch.Tensor,
            bev_histories: Optional[torch.Tensor]=None,
            transition: Optional[torch.Tensor]=None,
        ) -> torch.Tensor:

        """
        Input
        --------------------------------
        :bev_queries:   (N, H_bev * W_bev, C_bev) where N = batch_size

        :bev_spatial_shape: (1, 2) shape of each spatial feature map [[H_bev, W_bev]]

        :bev_histories: (N, H_bev * W_bev, C_bev) BEV features from the previous timestep t-1

        :transition: (N, 4, 4) the matrix that transitions the ego vehicle from t-1 to t 
                              (in homogeneous coordinates)
                              
        Returns
        --------------------------------
        :output: (N, H_bev * W_bev, C_bev), output BEV features / queries that have been attended to temporally
        """

        batch_size   = bev_queries.shape[0]
        H_bev, W_bev = bev_spatial_shape[0]
        H_bev        = H_bev.item()
        W_bev        = W_bev.item()
        device       = bev_queries.device        

        ref_points = TemporalSelfAttention.generate_standard_ref_points(
            (H_bev, W_bev), batch_size, device=device, normalize=True
        )
        ref_points = ref_points[..., None, :]
        
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

        assert transition is not None
        # The goal of this subsection before the deformable attention section is to align the BEV feature 
        # history (B{t-1}) to the current BEV feature (B{t}). This is to ensure that we account fo the spatial 
        # change that has occured in the newly created BEV feature (B{t}).

        # The transition matrix in question does the following:

        # 1. Transform the generated grid / ego vehicle coordinates to global coordinates
        #    with the pose of the ego vehicle in the previous frame

        # 2. Transform the global coordinates back to ego vehicle coordinates with the 
        #   pose-inverse of the ego vehicle in current frame

        # Both steps can be combined into one step with a single transformation matrix which will serve as our
        # transition matrix, pretty neat.

        bev_histories = TemporalSelfAttention.align_bev_histories(
            bev_histories=bev_histories,
            grid_xy_res=self.grid_xy_res,
            bev_spatial_shape=bev_spatial_shape,
            transition=transition,
            device=device
        )

        output = super(TemporalSelfAttention, self).forward(
            bev_queries,
            ref_points,
            bev_histories,
            value_spatial_shapes=bev_spatial_shape,
            normalize_ref_points=False
        )
        return output
    
    @staticmethod
    def align_bev_histories(
        bev_histories: torch.Tensor,
        grid_xy_res: Tuple[float, float],
        bev_spatial_shape: torch.Tensor,
        transition: torch.Tensor,
        device: Union[str, int, torch.device]="cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        H_bev, W_bev = bev_spatial_shape[0]
        H_bev        = H_bev.item()
        W_bev        = W_bev.item()

        batch_size, _, C_bev = bev_histories.shape

        x_range              = (grid_xy_res[0] * W_bev / 2) - (grid_xy_res[0] / 2)
        y_range              = (grid_xy_res[1] * H_bev / 2) - (grid_xy_res[1] / 2)
        xindex               = torch.linspace(-x_range, x_range, steps=W_bev, device=device)
        yindex               = torch.linspace(-y_range, y_range, steps=H_bev, device=device)
        ygrid, xgrid         = torch.meshgrid([yindex, xindex], indexing="ij")
        bev_grid_2d          = torch.stack([xgrid, ygrid], dim=-1).to(dtype=torch.float32, device=device)

        # bev_grid_2d: (H_bev, W_bev, 2) -> (1, H_bev, W_bev, 1, 2)
        # transition: (N, 4, 4) -> (N, 1, 1, 4, 4)
        # aligned_grid: (N, H_bev, W_bev, 2)
        bev_grid_2d  = bev_grid_2d[None, ..., None, :]
        transition   = transition[:, None, None, :, :]
        aligned_grid = transform_points(bev_grid_2d, transform_matrix=transition)
        aligned_grid = aligned_grid[..., 0, :]

        aligned_grid[..., 0] /= x_range
        aligned_grid[..., 1] /= y_range
        bev_histories        = bev_histories.permute(0, 2, 1).reshape(batch_size, C_bev, H_bev, W_bev)
        bev_histories        = F.grid_sample(
            bev_histories, aligned_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        bev_histories = bev_histories.permute(0, 2, 3, 1).reshape(batch_size, H_bev * W_bev, C_bev)
        return bev_histories



class SpatialCrossAttention(MultiViewDeformableAttention):
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
            multiscale_fmaps: torch.Tensor,
            bev_spatial_shape: Tuple[int, int],
            multiscale_fmap_shapes: torch.Tensor,
            z_refs: torch.Tensor,
            projection: torch.Tensor,
        ) -> torch.Tensor:
        
        """
        Input
        --------------------------------
        :bev_queries:    (N, H_bev * W_bev, C_bev) where N = batch_size
        
        :multiscale_fmaps:  (N, num_views, \sum{i=0}^{L-1} H_i \cdot W_i, C), reshaped and concatenated (along dim=1)
            feature maps from different pyramid levels / scales (L = num_fmap_levels)

        :bev_spatial_shape: (1, 2) shape of each spatial feature map [[H_bev, W_bev]]

        :multiscale_fmap_shapes: (L, 2) shape of each spatial feature across levels [[H0, W0], ...[Hn, Wn]]

        :z_refs: (num_z_ref_points, ) z-axis reference points

        :projection: (V, 3, 4) projection matrix for each camera, from real 3D coord (ego vehicle frame) to 3D image
            coord. This matrix is the product of the 3 x 3 (homogenized to 3 x 4) camera intrinsic matrix and the 4 x 4 camera
            camera extrinsic-inverse matrix with homogenized coordinates.

            NOTE: By convention, the extrinsic matrix maps from vehicle coordinate frame to sensor
                coordinate frame, but somehow it is the reverse in the case of waymo data, hence the reason the projection matrix
                is a product of intrinsic and extrinsic-inverse instead of intrinsic and extrinsic (check out section 3.2 of the
                paper: https://arxiv.org/pdf/1912.04838). If by any chance you use this on a different dataset, like NuScenes 
                for example, do ensure to stick to their own convention.

        Returns
        --------------------------------
        :output: (N, H_bev * W_bev, C_bev), output BEV features / queries that have been attended to spatially
        """
        assert multiscale_fmap_shapes.shape[0] == self.num_fmap_levels
        assert z_refs.shape[0] == self.num_z_ref_points
        assert multiscale_fmaps.shape[1] == self.num_views

        ref_points, attention_mask = SpatialCrossAttention.generate_sca_ref_points_and_attn_mask(
            batch_size=bev_queries.shape[0],
            num_views=self.num_views,
            grid_xy_res=self.grid_xy_res,
            bev_spatial_shape=bev_spatial_shape,
            multiscale_fmap_shapes=multiscale_fmap_shapes,
            z_refs=z_refs,
            projection=projection,
            device=bev_queries.device
        )
        return super(SpatialCrossAttention, self).forward(
            queries=bev_queries, 
            ref_points=ref_points, 
            value=multiscale_fmaps, 
            value_spatial_shapes=multiscale_fmap_shapes,
            attention_mask=attention_mask,
            normalize_ref_points=False
        )
    
    @staticmethod
    def generate_sca_ref_points_and_attn_mask(
            batch_size: int,
            num_views: int,
            grid_xy_res: Tuple[float, float],
            bev_spatial_shape: torch.Tensor,
            multiscale_fmap_shapes: torch.Tensor,
            z_refs: torch.Tensor,
            projection: torch.Tensor,
            device: Union[str, int, torch.device]="cpu"
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        H_bev, W_bev   = bev_spatial_shape[0]
        H_bev          = H_bev.item()
        W_bev          = W_bev.item()

        # Create BEV 2D grids pace
        # map BEV grid space to real world coordinates: If the BEV grid is a (200 x 200) grid
        # the index along both axes will range from 0 to 199, if the real world coordinates range
        # from -51.2m to 51.2m across both x and y axes, with the grid cell space being (0.512m, 0.512m)
        # across both axes, then we need to ensure that index (0, 0) represents  the center of a grid cell
        # whose top left corner is at (-51.2m, -51.2m) in real ego vehicle space and index (199, 199)
        # represents the center of a grid cell whose bottom  right corner is at (51.2m, 51.2m) in real ego
        # vehicle space. This also means that the x, y center of cell (0, 0) is (-50.944m, -50.944m) and
        # the center for cell (199, 199) is (-50.944m, 50.944m).
        x_range         = (grid_xy_res[0] * W_bev / 2) - (grid_xy_res[0] / 2)
        y_range         = (grid_xy_res[1] * H_bev / 2) - (grid_xy_res[1] / 2)
        xindex          = torch.linspace(-x_range, x_range, steps=W_bev, device=device)
        yindex          = torch.linspace(-y_range, y_range, steps=H_bev, device=device)
        ygrid, xgrid    = torch.meshgrid([yindex, xindex], indexing="ij")
        grid_2d         = torch.stack([xgrid, ygrid], dim=-1)
        grid_2d         = grid_2d.to(dtype=torch.float32, device=device)

        # Create 3D grid space, this is called pillaring, it's where we
        # raise the 2D grid space by reference points along the z-axis to make 3D pillars
        grid_3d          = torch.zeros(H_bev, W_bev, z_refs.shape[0], 3, dtype=grid_2d.dtype, device=device)
        grid_3d[..., :2] = grid_2d[:, :, None, :]
        grid_3d[..., 2]  = z_refs[None, None, :]

        # map the 3D (pillared 2D real world grid space) to 2D image space and then to
        # feature map space
        ones    = torch.ones(*grid_3d.shape[:-1], 1, device=device)
        grid_3d = torch.concat([grid_3d, ones], dim=-1)

        # projection: (V, 3, 4) -> (V, 1, 1, 1, 3, 4)
        # grid_3d:  (H_bev, W_bev, 4, 3)   -> (1, H_bev, W_bev, nz, 1, 3)
        # proj_2d: (V, H_bev, W_bev, nz, 2)
        projection = projection[:, None, None, None, :, :]
        grid_3d    = grid_3d[None, :, :, :, None, :]
        proj_2d    = transform_points(grid_3d, transform_matrix=projection)
        proj_2d    = proj_2d[..., 0, :2]

        # We need reference points normalized to be within range (-1, 1), we just need to divide ref_points
        # by (x_range, y_range).
        ref_points = proj_2d.reshape(num_views, H_bev * W_bev, z_refs.shape[0], 2).permute(1, 0, 2, 3)
        ref_points[..., 0] /= x_range
        ref_points[..., 1] /= y_range
        ref_points = ref_points[None, :, :, None, :, :].tile(batch_size, 1, 1, multiscale_fmap_shapes.shape[0], 1, 1)

        attention_mask = (ref_points[..., 0] >= -1) & (ref_points[..., 1] <= 1)

        return ref_points, attention_mask