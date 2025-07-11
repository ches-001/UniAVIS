import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class ConvBNorm(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: Union[int, Tuple[int, int]], 
            stride: Union[int, Tuple[int, int]]=1, 
            padding: Optional[Union[int, Tuple[int, int]]]=None,
            activation: Optional[nn.Module]=None,
            bias: bool=True,
            no_batchnorm: bool=False,
            batchnorm_first: bool=True,
            dim_mode: str="2d"
        ):
        super(ConvBNorm, self).__init__() 

        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = [i//2 for i in kernel_size]

        dim_mode              = dim_mode.lower()
        self.in_channels      = in_channels
        self.out_channels     = out_channels
        self._batchnorm_first = batchnorm_first

        self.conv             = getattr(nn, f"Conv{dim_mode}")(
            in_channels,
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=bias
        )
        self.norm             = getattr(nn, f"BatchNorm{dim_mode}")(out_channels) if (not no_batchnorm) else nn.Identity()
        self.activation       = activation or nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self._batchnorm_first:
            x = self.norm(x)
            x = self.activation(x)
        else:
            x = self.activation(x)
            x = self.norm(x)
        return x
    
class ConvTransposeBNorm(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: Union[int, Tuple[int, int]], 
            stride: Union[int, Tuple[int, int]]=1, 
            padding: Optional[Union[int, Tuple[int, int]]]=None,
            activation: Optional[nn.Module]=None,
            bias: bool=True,
            no_batchnorm: bool=False,
            batchnorm_first: bool=True,
            dim_mode: str="2d"
        ):
        super(ConvTransposeBNorm, self).__init__()

        dim_mode              = dim_mode.lower()
        self.in_channels      = in_channels
        self.out_channels     = out_channels
        padding               = padding or 0
        self._batchnorm_first = batchnorm_first

        self.conv_transpose   = getattr(nn, f"ConvTranspose{dim_mode}")(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )
        self.norm             = getattr(nn, f"BatchNorm{dim_mode}")(out_channels) if (not no_batchnorm) else nn.Identity()
        self.activation       = activation or nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        if self._batchnorm_first:
            x = self.norm(x)
            x = self.activation(x)
        else:
            x = self.activation(x)
            x = self.norm(x)
        return x

class AddNorm(nn.Module):
    def __init__(self, input_dim: int, **kwargs):
        super(AddNorm, self).__init__()

        self.input_dim  = input_dim
        self.layer_norm = nn.LayerNorm(self.input_dim, **kwargs)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(a + b)


class PosEmbedding1D(nn.Module):
    def __init__(
            self, 
            seq_len: int, 
            embed_dim: int, 
            learnable: bool=False, 
            padding_idx: Optional[int]=None, 
            _n: int=10000
        ):
        super(PosEmbedding1D, self).__init__()

        assert padding_idx is None or padding_idx < seq_len

        self.seq_len     = seq_len
        self.embed_dim   = embed_dim
        self.learnable   = learnable
        self.padding_idx = padding_idx
        self._n          = _n

        if self.learnable:
            self.pos_emb_module = nn.Embedding(
                seq_len, 
                embedding_dim=self.embed_dim, 
                padding_idx=self.padding_idx
            )

        else:
            pos                    = torch.arange(0, self.seq_len, dtype=torch.float32)
            vec_pos                = torch.arange(0, self.embed_dim, dtype=torch.float32)
            even_mask              = (vec_pos % 2) == 0
            odd_mask               = ~even_mask

            pos_embs               = vec_pos[None].tile(seq_len, 1)
            pos_embs[:, even_mask] = torch.sin(
                pos[:, None] / (self._n ** (2 * (pos_embs[:, even_mask] / 2) / self.embed_dim))
            )
            pos_embs[:, odd_mask]  = torch.cos(
                pos[:, None] / (self._n ** (2 * ((pos_embs[:, odd_mask] - 1) / 2) / self.embed_dim))
            )
            if padding_idx is not None:
                pos_embs[padding_idx] = pos_embs[padding_idx].fill_(0)

            self.register_buffer("pos_embs", pos_embs)

    def forward(self, indexes: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Inputs
        --------------------------------
        :indexes: (N, seq_len). The sequence of indexes to embed (if provided)

        Returns
        --------------------------------
        :output: (1 | N, max_seq_len | seq_len, embed_dim)
        """
        if self.learnable:
            device   = self.pos_emb_module.weight.device
            if indexes is None:
                indexes = torch.arange(0, self.seq_len, device=device)[None]
            pos_embs = self.pos_emb_module(indexes)

        else:
            if indexes is None:
                pos_embs = self.pos_embs[None]
            else:
                pos_embs = self.pos_embs[indexes]
        return pos_embs


class PosEmbedding2D(nn.Module):
    def __init__(
            self, 
            y_dim: int, 
            x_dim: int, 
            embed_dim: int, 
            learnable: bool=False, 
            padding_idx: Optional[int]=None, 
            _n: int=10000
        ):
        super(PosEmbedding2D, self).__init__()

        assert padding_idx is None or padding_idx < min(x_dim, y_dim)

        self.y_dim       = y_dim
        self.x_dim       = x_dim
        self.embed_dim   = embed_dim
        self.learnable   = learnable
        self.padding_idx = padding_idx
        self._n          = _n

        if self.learnable:
            # instead of having a single positional embedding module to embed BEV_H x BEV_W positional tokens
            # it seems better to have two positional embedding modules, one for each axis. This way, the number 
            # of parameters for positional embedding decreases drastically (from more than 10million to 100k+).
            # while also achieving the desired goal.
            self.x_pos_emb_module = nn.Embedding(x_dim, embedding_dim=self.embed_dim, padding_idx=self.padding_idx)
            self.y_pos_emb_module = nn.Embedding(y_dim, embedding_dim=self.embed_dim, padding_idx=self.padding_idx)

        else:
            x_pos                    = torch.arange(0, self.x_dim, dtype=torch.float32)
            y_pos                    = torch.arange(0, self.y_dim, dtype=torch.float32)
            vec_pos                  = torch.arange(0, self.embed_dim, dtype=torch.float32)
            even_mask                = (vec_pos % 2) == 0
            odd_mask                 = ~even_mask

            x_pos_embs               = vec_pos[None].tile(x_dim, 1)
            x_pos_embs[:, even_mask] = torch.sin(
                x_pos[:, None] / (self._n ** (2 * (x_pos_embs[:, even_mask] / 2) / self.embed_dim))
            )
            x_pos_embs[:, odd_mask]  = torch.cos(
                x_pos[:, None] / (self._n ** (2 * ((x_pos_embs[:, odd_mask] - 1) / 2) / self.embed_dim))
            )

            y_pos_embs               = vec_pos[None].tile(y_dim, 1)
            y_pos_embs[:, even_mask] = torch.sin(
                y_pos[:, None] / (self._n ** (2 * (y_pos_embs[:, even_mask] / 2) / self.embed_dim))
            )
            y_pos_embs[:, odd_mask]  = torch.cos(
                y_pos[:, None] / (self._n ** (2 * ((y_pos_embs[:, odd_mask] - 1) / 2) / self.embed_dim))
            )

            if padding_idx is not None:
                x_pos_embs[padding_idx] = x_pos_embs[padding_idx].fill_(0)
                y_pos_embs[padding_idx] = y_pos_embs[padding_idx].fill_(0)
                
            self.register_buffer("x_pos_embs", x_pos_embs)
            self.register_buffer("y_pos_embs", y_pos_embs)

    def forward(
            self,  
            y_indexes: Optional[torch.Tensor]=None,
            x_indexes: Optional[torch.Tensor]=None,
            flatten: bool=False
        ) -> torch.Tensor:
        """
        Input
        --------------------------------
        :x_indexes: (N, seq_len_x), a sequence of indexes along the x / horizontal / column dimension

        :y_indexes: (N, seq_len_y), a sequence of indexes along the y / vertical / row dimension
        
        :flatten: if True, reshape embeddings from shape (1, num_embed, H, W) to (1, H * W, num_embed)

        Returns
        --------------------------------
        :output: (1, num_embed, H, W) or (1, H * W, num_embed), (where: H = y_dim, X = x_dim)
        """
        if y_indexes is not None and x_indexes is not None:
            assert y_indexes.ndim == x_indexes.ndim and y_indexes.ndim >= 2
            assert (
                y_indexes.shape[0] == 1 
                or x_indexes.shape[0] == 1
                or y_indexes.shape[0] == x_indexes.shape[0]
            )

        if self.learnable:
            device = self.x_pos_emb_module.weight.device

            if x_indexes is None:
                x_indexes = torch.arange(0, self.x_dim, device=device)[None]

            if y_indexes is None:
                y_indexes = torch.arange(0, self.y_dim, device=device)[None]

            x_pos_embs = self.x_pos_emb_module(x_indexes)
            y_pos_embs = self.y_pos_emb_module(y_indexes)

        else:
            if x_indexes is None:
                x_pos_embs = self.x_pos_embs[None]
            else:
                x_pos_embs = self.x_pos_embs[x_indexes]

            if y_indexes is None:
                y_pos_embs = self.y_pos_embs[None]
            else:
                y_pos_embs = self.y_pos_embs[y_indexes]


        x_unperm_axes = [i for i in range(0, x_pos_embs.ndim - 2)]
        y_unperm_axes = [i for i in range(0, y_pos_embs.ndim - 2)]
        x_pos_embs = x_pos_embs.permute(*x_unperm_axes, -1, -2)[..., None, :]
        y_pos_embs = y_pos_embs.permute(*y_unperm_axes, -1, -2)[..., None]
        embs       = x_pos_embs + y_pos_embs    

        if flatten:
            embs = embs.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        return embs
    

class SpatialSinusoidalPosEmbedding(nn.Module):
    def __init__(self, embed_dim: int, _n: int=10000):
        super(SpatialSinusoidalPosEmbedding, self).__init__()
        self._n        = _n
        self.embed_dim = embed_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Input
        --------------------------------
        :x: (N, ..., 2) batch of raw x and y or y and x spatial coordinates

        Returns
        --------------------------------
        :output: (N, ..., embed_dim)
        """
        orig_shape               = input.shape
        input                    = input.reshape(orig_shape[0], -1, orig_shape[-1])
        axis_embed_dim           = self.embed_dim // input.shape[-1]
        vec_pos                  = torch.arange(0, axis_embed_dim, device=input.device, dtype=torch.float32)
        even_mask                = (vec_pos % 2) == 0
        odd_mask                 = ~even_mask
        vec_pos                  = vec_pos[((None,) * (input.ndim - 1)) + (slice(None),)].tile(*input.shape[:-1], 1)
        x_embs                   = torch.zeros(*input.shape[:-1], axis_embed_dim, device=input.device)
        y_embs                   = torch.zeros(*input.shape[:-1], axis_embed_dim, device=input.device)
        x_embs[..., even_mask]   = torch.sin(
            input[..., 0, None] / (self._n ** (2 * (vec_pos[..., even_mask] / 2) / self.embed_dim))
        )
        x_embs[..., odd_mask]      = torch.cos(
            input[..., 0, None] / (self._n ** (2 * ((vec_pos[..., odd_mask] - 1) / 2) / self.embed_dim))
        )
        y_embs[..., even_mask]   = torch.sin(
            input[..., 1, None] / (self._n ** (2 * (vec_pos[..., even_mask] / 2) / self.embed_dim))
        )
        y_embs[..., odd_mask]      = torch.cos(
            input[..., 1, None] / (self._n ** (2 * ((vec_pos[..., odd_mask] - 1) / 2) / self.embed_dim))
        )
        embs                     = torch.concat([x_embs, y_embs], dim=-1).reshape(*orig_shape[:-1], self.embed_dim)
        return embs
    

class DetectionHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super(DetectionHead, self).__init__()
        self.embed_dim       = embed_dim
        self.num_classes     = num_classes

        self.inception_module = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.loc_module   = nn.Sequential(
            nn.Linear(self.embed_dim, 6),
            nn.Sigmoid()
        )
        self.angle_module = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Tanh()
        )
        self.cls_module   = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
        --------------------------------
        :x: (N, max_objs, embed_dim) output of a TrackFormerDecoderLayer in the TrackFormer net

        Returns
        --------------------------------
        :output: (N, max_objs, det_params) (det_params = num_classes + 7), 
            we have: [center_x, center_y, center_z, length, width, height, heading_angle, [classes]]
        """
        out    = self.inception_module(x)
        loc    = self.loc_module(out)
        angle  = self.angle_module(out)
        cls    = self.cls_module(out)
        output = torch.concat([loc, angle, cls], dim=-1)
        return output
    

class RasterMapDetectionHead(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            num_classes: int, 
            num_coefs: int=None
        ):
        super(RasterMapDetectionHead, self).__init__()
        self.embed_dim        = embed_dim
        self.num_classes      = num_classes
        self.num_coefs        = num_coefs

        self.inception_module = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.loc_module       = nn.Sequential(
            nn.Linear(self.embed_dim, 4),
            nn.Sigmoid()
        )
        self.seg_coef_module  = nn.Sequential(
            nn.Linear(self.embed_dim, num_coefs),
            nn.Tanh()
        )
        self.cls_module       = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
        --------------------------------
        :x: (N, max_objs, embed_dim) output of the RasterMapFormerDecoderLayer

        Returns
        --------------------------------
        :output: (N, max_objs, 4, seg_coefs + num_classes)

        """
        out    = self.inception_module(x)
        loc    = self.loc_module(out)
        coefs  = self.seg_coef_module(out)
        cls    = self.cls_module(out)
        output = torch.concat([loc, coefs, cls], dim=-1)
        return output
    

class ProtoSegModule(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int=32, 
            c_h: int=256, 
        ):
        super(ProtoSegModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvBNorm(self.in_channels, c_h, kernel_size=3)
        self.conv2 = ConvBNorm(c_h, c_h, kernel_size=3)
        self.conv3 = ConvBNorm(c_h, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
    

class SimpleMLP(nn.Module):
    def __init__(
            self, 
            in_dim: int, 
            out_dim: int, 
            hidden_dim: int, 
            mid_activation: Optional[nn.Module]=None, 
            final_activation: Optional[nn.Module]=None
        ):
        super(SimpleMLP, self).__init__()

        self.in_dim           = in_dim
        self.out_dim          = out_dim
        self.hidden_dim       = hidden_dim
        self.mid_activation   = mid_activation or nn.ReLU()
        self.final_activation = final_activation or nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            self.mid_activation,
            nn.Linear(hidden_dim, self.out_dim),
            self.final_activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TemporalSpecificMLP(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            num_timesteps: int, 
            hidden_dim: int=512,
            num_layers: int=2,
        ):
        super(TemporalSpecificMLP, self).__init__()

        assert num_layers > 0
        assert in_features > 0
        assert out_features > 0

        self.in_features   = in_features
        self.out_features  = out_features
        self.num_timesteps = num_timesteps
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self._num_hidden   = max(0, self.num_layers - 2)

        if num_layers == 1:
            self.in_weight        = nn.Parameter(
                torch.empty((self.num_timesteps, self.out_features, self.in_features))
            )
            self.in_bias          = nn.Parameter(torch.empty(self.num_timesteps, self.out_features))
            self.n_hidden_weights = None
            self.n_hidden_biases  = None
            self.out_weight       = None
            self.out_bias         = None

        else:
            self.in_weight = nn.Parameter(
                torch.empty((self.num_timesteps, self.hidden_dim, self.in_features))
            )
            self.in_bias   = nn.Parameter(torch.empty(self.num_timesteps, self.hidden_dim))

            if self._num_hidden > 0:
                self.n_hidden_weights = nn.Parameter(
                    torch.empty((self._num_hidden, self.num_timesteps, self.hidden_dim, self.hidden_dim))
                )
                self.n_hidden_biases  = nn.Parameter(
                    torch.empty(self._num_hidden, self.num_timesteps, self.hidden_dim)
                )
            else:
                self.n_hidden_weights = None
                self.n_hidden_biases  = None
            
            self.out_weight = nn.Parameter(
                torch.empty((self.num_timesteps, self.out_features, self.hidden_dim))
            )
            self.out_bias   = nn.Parameter(
                torch.empty(self.num_timesteps, self.out_features)
            )

        self.reset_parameters()

    def _reset_bias(self, weight: nn.Parameter, bias: nn.Parameter):
        if weight is None or bias is None:
            return
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    def reset_parameters(self):
        params = [
            (self.in_weight, self.in_bias), 
            (self.n_hidden_weights, self.n_hidden_biases), 
            (self.out_weight, self.out_bias)
        ]
        for (weight, bias) in params:
            if weight is None or bias is None:
                continue
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self._reset_bias(weight, bias)

    def forward(self, x: torch.Tensor, tidx: int) -> torch.Tensor:
        out = F.linear(x, weight=self.in_weight[tidx], bias=self.in_bias[tidx])
        if self._num_hidden > 0:
            for lidx in range(0, self._num_hidden):
                out = F.relu(out)
                out = F.linear(
                    out, weight=self.n_hidden_weights[lidx, tidx], bias=self.n_hidden_biases[lidx, tidx]
                )
        if self.num_layers > 1:
            out = F.relu(out)
            out = F.linear(out, weight=self.out_weight[tidx], bias=self.out_bias[tidx])
        return out
