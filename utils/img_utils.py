import cv2
import torch
import numpy as np
from typing import Tuple

def point_clouds_to_binary_bev_maps(
        point_clouds: torch.Tensor, 
        map_size: Tuple[int, int], 
        x_min: float=-51.2,
        x_max: float=51.2,
        y_min: float=-51.2,
        y_max: float=51.2,
    ) -> torch.Tensor:

    """
    point_clouds shape: (batch_size, num_points, D)
    """

    ndim = point_clouds.ndim
    assert ndim == 2 or ndim == 3
    if ndim == 2:
        point_clouds = point_clouds[None]

    binary_map = torch.zeros(point_clouds.shape[0], *map_size, dtype=torch.uint8, device=point_clouds.device)
    points = point_clouds[..., :2].clone()
    points[..., 0] = (points[..., 0] - x_min) / (x_max - x_min)
    points[..., 1] = (points[..., 1] - y_min) / (y_max - y_min)

    # since the ego vehicle is always at the center, the center pixels are always blank. To completely vectorize this
    # operation, we set the scaled invalid points that fall out of range to 0.5, instead of having to iterate through
    # each sample in the batch one by one, since number of invalid points will vary across batches. After converting
    # these points to image indexes, we zero out the center pixels to remove those invalid points from the final
    # image.
    invalid_points_mask = (points[..., 0] < 0) | (points[..., 0] > 1) | (points[..., 1] < 0) | (points[..., 1] > 1)
    points[invalid_points_mask] = 0.5
    points[..., 0] *= (map_size[0] - 1)
    points[..., 1] *= (map_size[1] - 1)
    points = points.to(device=point_clouds.device, dtype=torch.int64)
    batch_indexes = torch.arange(0, point_clouds.shape[0], dtype=torch.int64, device=point_clouds.device)
    batch_indexes = batch_indexes[:, None].tile(1, points.shape[1])
    binary_map[batch_indexes, points[..., 0], points[..., 1]] = 1
    binary_map[:, map_size[0]//2, map_size[1]//2] = 0

    if ndim == 2:
        binary_map = binary_map[0]
    return binary_map


def generate_occupancy_map(
        motion_tracks: torch.Tensor,
        point_clouds: torch.Tensor,
        map_size: Tuple[int, int], 
        x_min: float=-51.2,
        x_max: float=51.2,
        y_min: float=-51.2,
        y_max: float=51.2,
    ) -> torch.Tensor:
    
    """
    motion_tracks shape: (num_detections, timesteps, 8)

    point_clouds shape: (timesteps, num_points, D)
    """

    assert motion_tracks.device == point_clouds.device
    xy_pos = motion_tracks[..., :2]

    num_dets, num_timesteps = motion_tracks.shape[:2]

    relative_xy_corners = torch.stack([
        torch.stack([-motion_tracks[..., 3], -motion_tracks[..., 4]], dim=2),
        torch.stack([+motion_tracks[..., 3], -motion_tracks[..., 4]], dim=2),
        torch.stack([+motion_tracks[..., 3], +motion_tracks[..., 4]], dim=2),
        torch.stack([-motion_tracks[..., 3], +motion_tracks[..., 4]], dim=2)
    ], dim=2)

    heading_angles = motion_tracks[..., -1]
    heading_cos = torch.cos(heading_angles)
    heading_sin = torch.sin(heading_angles)
    rotations = torch.stack(
        [heading_cos, -heading_sin, heading_sin, heading_cos],
    axis=2).reshape(*motion_tracks.shape[:2], 2, 2)

    vertices = torch.matmul(rotations, torch.permute(relative_xy_corners, (0, 1, 3, 2)))
    vertices = torch.permute(vertices, (0, 1, 3, 2)) + xy_pos[:, :, None, :]
    vertices = vertices.cpu()

    vertices[..., 0] = ((vertices[..., 0] - x_min) / (x_max - x_min)) * (map_size[0] - 1)
    vertices[..., 1] = ((vertices[..., 1] - y_min) / (y_max - y_min)) * (map_size[1] - 1)
    vertices = vertices.cpu().to(dtype=torch.int64)

    occ_maps = torch.zeros((*motion_tracks.shape[:2], *map_size), dtype=torch.uint8)

    binary_occ_maps = point_clouds_to_binary_bev_maps(
        point_clouds, 
        map_size, 
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    binary_occ_maps = binary_occ_maps.cpu()

    # TODO: This block of code needs optimization
    for tidx in range(0, num_timesteps):
        binary_occ_map = binary_occ_maps[tidx]
        for didx in range(0, num_dets):
            v = vertices[didx, tidx].cpu().numpy()
            if np.any(v < 0) or np.any((v[..., 0] >= map_size[0]) | (v[..., 1] >= map_size[1])):
                continue
            occ_map = cv2.fillPoly(occ_maps[didx, tidx].cpu().numpy(), pts=[v], color=1)
            occ_map = torch.from_numpy(occ_map)
            occ_maps[didx, tidx] = occ_map * binary_occ_map
    occ_maps = occ_maps.cumsum(dim=1).clamp(min=0, max=1)
    return occ_maps
    