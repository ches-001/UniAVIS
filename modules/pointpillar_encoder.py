import torch
import torch.nn as nn
from .common import SimpleConvMLP, SimpleMLP
from typing import Tuple, List, Optional, Type, Union

class PillarFeatureGenerator(nn.Module):
    def __init__(
            self, 
            pillar_wh: Tuple[float, float]=(0.16, 0.16),
            max_points: int=100,
            max_pillars: int=12_000,
            xyz_range: Optional[List[Tuple[float, float]]]=None,
        ):
        super(PillarFeatureGenerator, self).__init__()
        self.pillar_wh   = pillar_wh
        self.max_points  = max_points
        self.max_pillars = max_pillars
        self.xyz_range   = xyz_range or [(0.0, 70.4), (-40.0, 40.0), (-3.0, 1.0)]

        for r in self.xyz_range: assert r[1] > r[0]

    def forward(self, point_clouds: torch.Tensor, pad_value: float=0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input
        --------------------------------
        :point_clouds: (N, num_points, d) Point cloud data (num_points = number of points per sample), 
            with d=4 dimensions (x, y, z, r) where r is reflectance and x, y, z are in meters

        :pad_value: Value to pad empty points in pillars

        Returns
        --------------------------------
        :output: (N, max_pillars, max_points, 9)

        :output_pillars: (N, max_pillars), output pillar indexes
        """
        batch_size   = point_clouds.shape[0]
        device       = point_clouds.device
        x_range      = self.xyz_range[0]
        y_range      = self.xyz_range[1]
        min_xy       = torch.tensor([x_range[0], y_range[0]], device=device)
        max_xy       = torch.tensor([x_range[1], y_range[1]], device=device)
        pillar_wh    = torch.tensor(self.pillar_wh, device=device)
        num_xy_grids = torch.ceil((max_xy - min_xy) / pillar_wh)
        num_xy_grids = num_xy_grids.to(device=device, dtype=torch.int64)

        # pillar i (x-axis / col idx), j (y-axis / row idx) indexes (The are indexes of all the pillars each
        # point belongs to, so these are indexes of non-empty pillars)
        pillar_ij      = torch.min(torch.floor((point_clouds[:, :, :2] - min_xy) / pillar_wh), num_xy_grids-1)
        pillar_ij      = pillar_ij.to(device=device, dtype=torch.int64)
        pillar_indexes = (pillar_ij[:, :, 1] * num_xy_grids[0]) + pillar_ij[:, :, 0]
        pillar_indexes = pillar_indexes.to(device=device, dtype=torch.int64)
        output         = torch.full((batch_size, self.max_pillars, self.max_points, 9), torch.nan, device=device)
        output_pillars = torch.full((batch_size, self.max_pillars), -1, device=device, dtype=torch.int64)

        for i in range(0, batch_size):
            unique_pillar_indexes = pillar_indexes[i].unique(sorted=False)
            pillar_perm           = torch.randperm(unique_pillar_indexes.shape[0], device=device)
            pillar_perm           = pillar_perm[:self.max_pillars]
            unique_pillar_indexes = unique_pillar_indexes[pillar_perm]

            selected_pillar_mask  = torch.isin(pillar_indexes[i], unique_pillar_indexes)
            unique_out_return     = pillar_indexes[i][selected_pillar_mask].unique(return_inverse=True)

            num_pillars           = unique_out_return[0].shape[0]
            output_pillars[i, :num_pillars] = unique_out_return[0]

            # compute x, y center coordinate of each pillar
            sample_pillars_ij     = torch.stack([
                unique_out_return[0] % num_xy_grids[0],
                 unique_out_return[0] // num_xy_grids[0]
            ], dim=-1)
            pillar_xy_centers     = min_xy + (sample_pillars_ij * pillar_wh) + (pillar_wh / 2)

            # sort inversed pillar indexes and sample point clouds accordingly
            sample_pillar_indexes = unique_out_return[1]
            sample_point_clouds   = point_clouds[i][selected_pillar_mask]
            sample_pillar_indexes, sort_indexes = torch.sort(sample_pillar_indexes)
            sample_point_clouds   = sample_point_clouds[sort_indexes]

            # compute x, y arithmetic mean of points for each pillar
            point_xyz_sums  = torch.zeros(sample_pillar_indexes.max()+1, 3, device=device)
            point_xyz_count = torch.zeros(sample_pillar_indexes.max()+1, 1, dtype=torch.int64, device=device)
            ones            = torch.ones(sample_pillar_indexes.shape[0], 1, dtype=torch.int64, device=device)
            point_xyz_sums  = point_xyz_sums.index_add(0, sample_pillar_indexes, sample_point_clouds[..., :3])
            point_xyz_count = point_xyz_count.index_add(0, sample_pillar_indexes, ones)
            point_xyz_mean  = point_xyz_sums / point_xyz_count

            # calculate index of each point in the pillar (with the max points per pillar limit in mind)
            point_xyz_count = point_xyz_count[:, 0]
            repeated        = torch.repeat_interleave(point_xyz_count)
            offsets         = torch.cumsum(
                torch.concat([torch.zeros(1, device=device, dtype=torch.int64), point_xyz_count], dim=0), dim=0
            )
            sample_point_indexes  = torch.arange(repeated.shape[0], device=device) - offsets[repeated]
            selected_mask         = sample_point_indexes < self.max_points
            sample_pillar_indexes = sample_pillar_indexes[selected_mask]
            sample_point_indexes  = sample_point_indexes[selected_mask]
            sample_point_clouds   = sample_point_clouds[selected_mask]
            flattened_indexes     = (sample_pillar_indexes * self.max_points) + sample_point_indexes

            # caculate the x, y, z distance from each point to the arithmetic mean for each pillar,
            # and the x, y offset of each point in a pillar to the center of that pillar
            sample_output           = output[i, :num_pillars].reshape(-1, output.shape[-1])
            sample_output[flattened_indexes, :4] = sample_point_clouds
            sample_output           = sample_output.reshape(num_pillars, *output.shape[2:])
            sample_output[..., 4:7] = (sample_output[..., :3] - point_xyz_mean[:, None, :]).abs()
            sample_output[..., 7:]  = (pillar_xy_centers[:, None, :] - sample_output[..., :2])
            output[i, :num_pillars] = sample_output

        output[output != output] = pad_value
        return output, output_pillars


class GMaxPoolMixin:
    shared_mlp1: Union[SimpleMLP, SimpleMLP]
    dim_mode: str

    def apply_max_pooling(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.shared_mlp1, SimpleConvMLP):
            if self.dim_mode == "1d":
                return x.max(dim=2)[0]
            return x.max(dim=3)[0].permute(0, 2, 1)
        return x.max(dim=(1 if self.dim_mode == "1d" else 2))[0]


class TNet(nn.Module, GMaxPoolMixin):
    def __init__(self, in_dim: int, mlp_type: Type=SimpleConvMLP, net_scale: float=1.0, dim_mode: str="2d"):
        super(TNet, self).__init__()

        assert mlp_type in (SimpleMLP, SimpleConvMLP)

        out_dims         = list(map(lambda d : max(int(d * net_scale), 16), [16, 32, 256, 128, 64]))
        self.dim_mode    = dim_mode
        kwargs           = {"dim_mode": self.dim_mode} if mlp_type == SimpleConvMLP else {}
        self.shared_mlp1 = mlp_type(in_dim=in_dim, out_dim=out_dims[0], hidden_dim=out_dims[0]//2, **kwargs)
        self.shared_mlp2 = mlp_type(in_dim=out_dims[0], out_dim=out_dims[1], hidden_dim=out_dims[1]//2, **kwargs)
        self.shared_mlp3 = mlp_type(in_dim=out_dims[1], out_dim=out_dims[2], hidden_dim=out_dims[2]//2, **kwargs)
        self.fc1         = nn.Sequential(
            nn.Linear(out_dims[2], out_dims[3]),
            nn.LeakyReLU(0.2)
        )
        self.fc2         = nn.Sequential(
            nn.Linear(out_dims[3], out_dims[4]),
            nn.LeakyReLU(0.2)
        )
        self.fc3         = nn.Linear(out_dims[4], in_dim**2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input
        --------------------------------
        :x: (N, n, d) for 1D or (N, p, n, d) for 2D, input batch of points grouped by pillars where:
            (N = batch size, p = number of pillars, n = number of points per pillar, d = number of dimensions)

        Returns
        --------------------------------
        :out: (N, n, d) or (N, p, n, d), transformed input
        :tm: (N, d, d) or (N, p, d, d), Transformation matrix used to transform input
        """
        if isinstance(self.shared_mlp1, SimpleConvMLP):
            if self.dim_mode == "1d":
                input = x.permute(0, 2, 1)
            else:
                input = x.permute(0, 3, 1, 2)
            tm = self.shared_mlp1(input, permute_dim=False)
            tm = self.shared_mlp2(tm, permute_dim=False)
            tm = self.shared_mlp3(tm, permute_dim=False)
        else:
            tm = self.shared_mlp1(x)
            tm = self.shared_mlp2(tm)
            tm = self.shared_mlp3(tm)
        
        tm = self.apply_max_pooling(tm)
        tm = self.fc1(tm)
        tm = self.fc2(tm)
        tm = self.fc3(tm)
        x, tm = self._apply_transform(x, tm)
        return x, tm
    
    def _apply_transform(self, x: torch.Tensor, tm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dim_mode == "1d":
            tm = tm.reshape(tm.shape[0], self.shared_mlp1.in_dim, self.shared_mlp1.in_dim)
        else:
            tm = tm.reshape(tm.shape[0], tm.shape[1], self.shared_mlp1.in_dim, self.shared_mlp1.in_dim)
        x  = torch.matmul(x, tm)
        return x, tm


class PointNet(nn.Module, GMaxPoolMixin):
    def __init__(
            self, 
            in_dim: int, 
            out_dim: int=256,
            mlp_type: Type=SimpleConvMLP, 
            net_scale: float=1.0, 
            dim_mode: str="2d"
        ):
        super(PointNet, self).__init__()

        
        self.dim_mode         = dim_mode
        self.net_scale        = net_scale
        out_dims              = list(map(lambda d : max(int(d * self.net_scale), 16), [16, 16, 32]))
        kwargs                = {"dim_mode": self.dim_mode} if mlp_type == SimpleConvMLP else {}

        self.tnet1            = TNet(in_dim, mlp_type=mlp_type, net_scale=net_scale, dim_mode=self.dim_mode)
        self.shared_mlp1      = mlp_type(in_dim=in_dim, out_dim=out_dims[0], hidden_dim=out_dims[0]//2, **kwargs)
        self.shared_mlp2      = mlp_type(in_dim=out_dims[0], out_dim=out_dims[1], hidden_dim=out_dims[1]//2, **kwargs)

        self.tnet2            = TNet(out_dims[1], mlp_type=mlp_type, net_scale=net_scale, dim_mode=self.dim_mode)
        self.shared_mlp3      = mlp_type(in_dim=out_dims[1], out_dim=out_dims[2], hidden_dim=out_dims[2]//2, **kwargs)
        self.shared_mlp4      = mlp_type(in_dim=out_dims[2], out_dim=out_dim, hidden_dim=out_dim//2, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
        --------------------------------
        :x: (N, n, d) for 1D or (N, p, n, d) for 2D, input batch of points grouped by pillars where:
            (N = batch size, p = number of pillars, n = number of points per pillar, d = number of dimensions)

        Returns
        --------------------------------
        :gfeatures: (N, c) or (N, p, c), Global feature representation of points per sample, where:
            (c = new dim size)

        :l_reg: 0D tensor, Loss regularisation term. This term is computed from the L2-norm between
             an identity matrix I and the transformation matrix of the second TNet in the network 
             multiplied by its transpose, This term is to be added to the loss function (accompanied 
             with a scale factor). mathematically: **L_reg = ||I - M.M^T||^2**
            This is necessary because the M matrix is large since its meant to affine transform points
            in higher dimensional embedding space, as such we need to maintain orthogonality to avoid
            things like abrupt scaling and shearing of points / objects
        """
        out, _    = self.tnet1(x)
        out       = self._apply_mlp(out, self.shared_mlp1, permute_in=True, permute_out=False)
        out       = self._apply_mlp(out, self.shared_mlp2, permute_in=False, permute_out=True)
        out, hdtm = self.tnet2(out) # hdtm: High Dimensional Transform Matrix
        out       = self._apply_mlp(out, self.shared_mlp3, permute_in=True, permute_out=False)
        out       = self._apply_mlp(out, self.shared_mlp4, permute_in=False, permute_out=False)
        gfeatures = self.apply_max_pooling(out)
        l_reg     = self._compute_l_reg(hdtm)
        return gfeatures, l_reg
    
    def _compute_l_reg(self, hdtm: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(hdtm.shape[2])
        if self.dim_mode == "1d":
            permute_dim = (0, 2, 1)
            identity    = identity[None, :, :]
        else:
            permute_dim = (0, 1, 3, 2)
            identity    = identity[None, None, :, :]
        
        hdmt_hdmt_t = torch.matmul(hdtm, hdtm.permute(*permute_dim))
        l_reg       = (identity - hdmt_hdmt_t).pow(2).sum(dim=-1).sum(dim=-1)
        if l_reg.ndim == 1:
            return l_reg.mean()
        return l_reg.sum().mean()

    def _apply_mlp(
            self, 
            x: torch.Tensor, 
            mlp: Union[SimpleMLP, SimpleConvMLP], 
            permute_in: bool=False, 
            permute_out: bool=False,
        ) ->torch.Tensor:
        if isinstance(mlp, SimpleConvMLP):
            if permute_in: 
                dims = (0, 2, 1) if self.dim_mode == "1d" else (0, 3, 1, 2)
                x = x.permute(*dims)
            out = mlp(x, permute_dim=False)
            if permute_out:
                dims = (0, 2, 1) if self.dim_mode == "1d" else (0, 2, 3, 1)
                return out.permute(*dims)
            return out
        return mlp(x)
    

class PillarFeatureNet(nn.Module):
    def __init__(
            self,
            out_dim: int=256,
            pillar_wh: Tuple[float, float]=(0.16, 0.16),
            max_points: int=100,
            max_pillars: int=12_000,
            out_grid_hw: int=(200, 200),
            xyz_range: Optional[List[Tuple[float, float]]]=None,
            mlp_type: Type=SimpleConvMLP,
            net_scale: float=1.0
    ):
        super(PillarFeatureNet, self).__init__()

        self.out_grid_hw    = out_grid_hw
        self.pillar_gen     = PillarFeatureGenerator(
            pillar_wh, max_points=max_points, max_pillars=max_pillars, xyz_range=xyz_range
        )
        self.point_net      = PointNet(in_dim=9, out_dim=out_dim, mlp_type=mlp_type, net_scale=net_scale, dim_mode="2d")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input
        --------------------------------
        :x: (N, n, d), input batch of points with d dimensions:
            (N = batch size, n = number of points per pillar, d = number of dimensions)

        Returns
        --------------------------------
        :out: (N, c, H, W) Global feature from the point net scattered across a 2D grid of pillars

        :l_reg: 0D tensor, Loss regularisation term from the point net
        """
        xp, pillar_indexes = self.pillar_gen(x)
        gfeatures, l_reg   = self.point_net(xp)

        batch_size    = x.shape[0]
        num_pillars   = gfeatures.shape[1]
        num_channels  = gfeatures.shape[2]
        device        = x.device
        min_yx        = torch.tensor([self.pillar_gen.xyz_range[1][0], self.pillar_gen.xyz_range[0][0]])
        max_yx        = torch.tensor([self.pillar_gen.xyz_range[1][1], self.pillar_gen.xyz_range[0][1]])
        pillar_hw     = torch.tensor([self.pillar_gen.pillar_wh[1], self.pillar_gen.pillar_wh[1]])
        grid_hw       = torch.ceil((max_yx - min_yx) / pillar_hw)
        grid_hw       = grid_hw.to(dtype=torch.int64)
        
        # pillar i (x-axis / col idx), j (y-axis / row idx) in this case
        pillars_ij    = torch.stack([pillar_indexes % grid_hw[1], pillar_indexes // grid_hw[1]], dim=-1)

        # I was stuck in a delimma, where I had to make one of two choices concerning the design of this forward
        # method. This code scatters the computed global features on a grid canvas in their respective 2D pillar
        # positions as defined in the pillars_ij index tensor, I believe there are one of two ways of doing this:
        #
        # 1. Create the grid_canvas with its original (calculated) size, scatter the features on the canvas and
        #   reshape the canvas to its desired output size via bilinear interpolation.
        #
        # 2. Create the grid_canvas with the desire output size, recalculate the ij indexes for the corresponding
        #   pillars (scale to the new size) and then scatter the features on the canvas.
        #
        # Both methods ultimately result in the same grid size and I was about to go with the second method because
        # the lack of interpolation made it look efficient and hence a better option, and also, since the grid is
        # initialized to its desired size rather than its actual size, if the desired size is smaller (which it is 
        # in my case) then it would have a lesser memory footprint. However, I decided to go with the first option
        # because I figured that if I scaled the ij indexes down, it may cause features to be placed upon features, or
        # in better words, features will be completely replaced by neighbouring features, simply because the ij indexes
        # were calculated to be the same. Take for instance the (i, j) indexes of (300, 320) and (302, 322) on a (H x W)
        # grid of size (500 x 441), suppose our desired grid shape is (200 x 200), these indexes will be updated with
        # the formula: ij := ⌊ij * (new_WH / old_WH)⌋ where: ⌊.⌋ = floor(.), which will yield (136, 128) and (136, 128)
        # respectively. These are two different global pillar features but they have the same spatial position on the 
        # new grid despite having different positions on the old one, this means that one would simply overwrite the 
        # other completely. Of course one can argue that despite being on different spatial positions on the old grid, 
        # after bilinear interpolation (with the first method), the feature values will not remain the same. While that 
        # is true, the resulting feature values from bilinear interpolation is a linear combination of neighbouring
        # values, so the information of both features still make contributions, unlike in the second method.

        # Perhaps I am just being paranoid and overreacting to the complete overwriting of global pillar features by close 
        # neighbours, because afterall, the output is probably too big to tell the difference, well I might change the
        # implementation later, time will tell.
        batch_indexes = torch.arange(batch_size, device=device)[:, None].tile(1, num_pillars)
        grid_canvas   = torch.zeros(batch_size, grid_hw[0].item(), grid_hw[1].item(), num_channels, device=device)
        grid_canvas[batch_indexes, pillars_ij[..., 1], pillars_ij[..., 0], :] = gfeatures
        grid_canvas   = grid_canvas.permute(0, 3, 1, 2)
        
        if grid_hw[0] != self.out_grid_hw[0] and grid_hw[1] != self.out_grid_hw[1]:
            grid_canvas   = nn.functional.interpolate(grid_canvas, size=self.out_grid_hw, mode="bilinear")
        return grid_canvas, l_reg