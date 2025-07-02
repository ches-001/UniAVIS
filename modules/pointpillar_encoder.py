import torch
import torch.nn as nn
from typing import Tuple, List, Optional

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
        self.xyz_range   = xyz_range or [(-51.2, 51.2), (-51.2, 51.2), (-5.0, 3.0)]

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
        :output: (N, max_pillars, max_points, 9), with the last dimension being:
            (x, y, z, r, x_c, y_c, z_c, x_p, y_p), where subscript c and p denote distance to arithmetic mean
            of all points in a pillar and the offset from the pillar x, y center

        :output_pillars: (N, max_pillars), output pillar indexes
        """
        batch_size   = point_clouds.shape[0]
        device       = point_clouds.device
        x_range      = self.xyz_range[0]
        y_range      = self.xyz_range[1]
        z_range      = self.xyz_range[2]
        min_xyz      = torch.tensor([x_range[0], y_range[0], z_range[0]], device=device)
        max_xyz      = torch.tensor([x_range[1], y_range[1], z_range[1]], device=device)
        min_xy       = min_xyz[:2]
        max_xy       = max_xyz[:2]
        pillar_wh    = torch.tensor(self.pillar_wh, device=device)
        num_xy_grids = torch.ceil((max_xy - min_xy) / pillar_wh)
        num_xy_grids = num_xy_grids.to(device=device, dtype=torch.int64)

        # scale from min_xy, max_xy to (-1, -1, -1), (1, 1, 1)
        pillar_wh             = (2 * pillar_wh) / (max_xy - min_xy)
        point_clouds[..., :3] = (2 * (point_clouds[..., :3] - min_xyz) / (max_xyz - min_xyz)) - 1
        min_xy                = min_xy / max_xy
        max_xy                = max_xy / max_xy

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


class SimplifiedPointNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int=256,):
        super(SimplifiedPointNet, self).__init__()

        self._layer = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
        --------------------------------
        :x: (N, p, n, d) input batch of points grouped by pillars where:
            (N = batch size, p = number of pillars, n = number of points per pillar, d = number of dimensions)

        Returns
        --------------------------------
        :gfeatures: (N, p, c), Global feature representation of points per sample, where:
            (c = new dim size)
        """
        x = self._layer(x)
        x = torch.max(x, dim=2).values
        return x
    

class PillarFeatureNet(nn.Module):
    def __init__(
            self,
            out_dim: int=256,
            pillar_wh: Tuple[float, float]=(0.16, 0.16),
            max_points: int=100,
            max_pillars: int=12_000,
            out_grid_hw: int=(200, 200),
            xyz_range: Optional[List[Tuple[float, float]]]=None,
    ):
        super(PillarFeatureNet, self).__init__()

        self.out_grid_hw    = out_grid_hw
        self.pillar_gen     = PillarFeatureGenerator(
            pillar_wh, max_points=max_points, max_pillars=max_pillars, xyz_range=xyz_range
        )
        self.point_net      = SimplifiedPointNet(in_dim=9, out_dim=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
        --------------------------------
        :x: (N, n, d), input batch of points with d dimensions:
            (N = batch size, n = number of points per pillar, d = number of dimensions)

        Returns
        --------------------------------
        :out: (N, c, H, W) Global feature from the point net scattered across a 2D grid of pillars

        """
        xp, pillar_indexes = self.pillar_gen(x)
        gfeatures          = self.point_net(xp)

        batch_size    = x.shape[0]
        num_pillars   = gfeatures.shape[1]
        num_channels  = gfeatures.shape[2]
        device        = x.device

        grid_h = (self.pillar_gen.xyz_range[1][1] - self.pillar_gen.xyz_range[1][0]) / self.pillar_gen.pillar_wh[1]
        grid_w = (self.pillar_gen.xyz_range[0][1] - self.pillar_gen.xyz_range[0][0]) / self.pillar_gen.pillar_wh[0]

        # pillar i (x-axis / col idx), j (y-axis / row idx) in this case
        pillars_ij = torch.stack([pillar_indexes % grid_w, pillar_indexes // grid_h], dim=-1)
        pillars_ij[..., 0] = pillars_ij[..., 0] * (self.out_grid_hw[1] - 1) / (grid_w - 1)
        pillars_ij[..., 1] = pillars_ij[..., 1] * (self.out_grid_hw[0] - 1) / (grid_h - 1)
        pillars_ij = pillars_ij.long()
        
        batch_indexes = torch.arange(batch_size, device=device)[:, None].tile(1, num_pillars)
        grid_canvas   = torch.zeros(batch_size, num_channels, self.out_grid_hw[0], self.out_grid_hw[1], device=device)
        grid_canvas[batch_indexes, :, pillars_ij[..., 1], pillars_ij[..., 0]] = gfeatures
        return grid_canvas