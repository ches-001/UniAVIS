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
        self.pillar_wh              = pillar_wh
        self.max_points  = max_points
        self.max_pillars = max_pillars
        self.xyz_range = xyz_range or [(0.0, 70.4), (-40.0, 40.0), (-3.0, 1.0)]

        for r in self.xyz_range: assert r[1] > r[0]

    def forward(self, point_clouds: torch.Tensor, pad_value: float=0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input
        --------------------------------
        :point_clouds: (N, n, 4) Point cloud data (n = number of points per sample), with 4
          dimensions (x, y, z, r) where r is reflectance and x, y, z are in meters

        :pad_value: Value to pad empty points in pillars

        Returns
        --------------------------------
        :output: (N, max_pillars, max_points, 9)

        :output_pillars: (N, max_pillars), output pillar indexes
        """
        batch_size      = point_clouds.shape[0]
        device          = point_clouds.device
        x_range         = self.xyz_range[0]
        y_range         = self.xyz_range[1]
        min_xy          = torch.tensor([x_range[0], y_range[0]], device=device)
        max_xy          = torch.tensor([x_range[1], y_range[1]], device=device)
        pillar_wh       = torch.tensor(self.pillar_wh, device=device)
        num_xy_grids    = torch.ceil((max_xy - min_xy) / pillar_wh)
        num_xy_grids    = num_xy_grids.to(device=device, dtype=torch.int64)

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