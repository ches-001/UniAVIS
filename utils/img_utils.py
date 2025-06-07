import cv2
import torch
import numpy as np
from typing import Tuple, Optional, List

def point_clouds_to_binary_bev_maps(
        point_clouds: torch.Tensor, 
        map_hw: Tuple[int, int], 
        x_min: float=-51.2,
        x_max: float=51.2,
        y_min: float=-51.2,
        y_max: float=51.2,
    ) -> torch.Tensor:
    """
    Inputs
    --------------------------------
    :point_clouds: 
        shape: (batch_size, num_points, D) | (num_points, D)

    :map_hw:
        BEV (Bird Eye View) 2D height and width

    :x_min:
        minimum value along the x-axis of Ego vehicle frame

    :x_max:
        maximum value along the x-axis of Ego vehicle frame

    :y_min:
        minimum value along the y-axis of Ego vehicle frame

    :y_max:
        maximum value along the y-axis of Ego vehicle frame

    Returns
    --------------------------------
    :detection agonistic binary BEV map:
        shape: (batch_size, bev_H, bev_W) | (bev_H, bev_W)
    """

    ndim = point_clouds.ndim
    assert ndim == 2 or ndim == 3
    if ndim == 2:
        point_clouds = point_clouds[None]

    binary_map = torch.zeros(point_clouds.shape[0], *map_hw, dtype=torch.uint8, device=point_clouds.device)
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
    points[..., 0] *= (map_hw[1] - 1)
    points[..., 1] *= (map_hw[0] - 1)
    points = points.to(device=point_clouds.device, dtype=torch.int64)
    batch_indexes = torch.arange(0, point_clouds.shape[0], dtype=torch.int64, device=point_clouds.device)
    batch_indexes = batch_indexes[:, None].tile(1, points.shape[1])
    binary_map[batch_indexes, points[..., 1], points[..., 0]] = 1
    binary_map[:, map_hw[0]//2, map_hw[1]//2] = 0

    if ndim == 2:
        binary_map = binary_map[0]
    return binary_map


def generate_occupancy_map(
        motion_tracks: torch.Tensor,
        map_hw: Tuple[int, int], 
        x_min: float=-51.2,
        x_max: float=51.2,
        y_min: float=-51.2,
        y_max: float=51.2,
        point_clouds: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
    
    """
    Inputs
    --------------------------------
    :motion_tracks: 
        shape: (num_detections, num_timesteps, D)

    :map_hw:
        BEV (Bird Eye View) 2D size

    :x_min:
        minimum value along the x-axis of Ego vehicle frame

    :x_max:
        maximum value along the x-axis of Ego vehicle frame

    :y_min:
        minimum value along the y-axis of Ego vehicle frame

    :y_max:
        maximum value along the y-axis of Ego vehicle frame

    :point_clouds (optional): 
        shape: (num_timesteps, num_points, D). If provided, it will be used to compute a detection 
        instance agonistic binary occupancy map which will serve as a mask to be applied to the detection
        level occupancy map made from the bounding box data of each agent across various timesteps from the
        motion tracks.

    Returns
    --------------------------------
    :detection instance level binary BEV map:
        shape: (num_detections, num_timesteps, bev_H, bev_W)
    """
    xy_pos = motion_tracks[..., :2]

    num_dets, num_timesteps = motion_tracks.shape[:2]

    relative_xy_corners = torch.stack([
        torch.stack([-motion_tracks[..., 3], -motion_tracks[..., 4]], dim=2),
        torch.stack([+motion_tracks[..., 3], -motion_tracks[..., 4]], dim=2),
        torch.stack([+motion_tracks[..., 3], +motion_tracks[..., 4]], dim=2),
        torch.stack([-motion_tracks[..., 3], +motion_tracks[..., 4]], dim=2)
    ], dim=2)
    relative_xy_corners /= 2

    heading_angles = motion_tracks[..., -1]
    heading_cos = torch.cos(heading_angles)
    heading_sin = torch.sin(heading_angles)
    rotations = torch.stack(
        [heading_cos, heading_sin, -heading_sin, heading_cos],
    axis=2).reshape(*motion_tracks.shape[:2], 2, 2)

    vertices = torch.matmul(relative_xy_corners, rotations) + xy_pos[:, :, None, :]
    vertices[..., 0] = ((vertices[..., 0] - x_min) / (x_max - x_min)) * (map_hw[1] - 1)
    vertices[..., 1] = ((vertices[..., 1] - y_min) / (y_max - y_min)) * (map_hw[0] - 1)

    vertices = vertices.cpu().to(dtype=torch.int64).numpy()
    occ_maps = np.zeros((*motion_tracks.shape[:2], *map_hw), dtype=np.uint8)

    # TODO: This block of code might needs optimization
    for tidx in range(0, num_timesteps):
        for didx in range(0, num_dets):
            v = vertices[didx, tidx]
            if np.any(v < 0) or np.any((v[..., 0] >= map_hw[0]) | (v[..., 1] >= map_hw[1])):
                continue
            occ_map = cv2.fillPoly(occ_maps[didx, tidx], pts=[v], color=1)
            occ_maps[didx, tidx] = occ_map

    occ_maps = occ_maps
    occ_maps = torch.from_numpy(occ_maps)
    
    if point_clouds is not None:
        binary_occ_maps = point_clouds_to_binary_bev_maps(
            point_clouds, 
            map_hw, 
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        ).cpu()
        occ_maps *= binary_occ_maps[None]
    return occ_maps
    

def polylines_to_xywh(polylines: List[np.ndarray]) -> List[np.ndarray]:
    bboxes = []
    for polyline in polylines:
        assert polyline.ndim == 2
        x1, y1, x2, y2 = polyline[:, 0].min(), polyline[:, 1].min(), polyline[:, 0].max(), polyline[:, 1].max()
        w, h = x2 - x1, y2 - y1
        x, y = x1 + (w/2), y1 + (h/2)
        bboxes.append(np.asarray([x, y, w, h]))
    return bboxes


def polylines_to_masks(
        polylines: List[np.ndarray],
        img_hw: Tuple[int, int],
        scale_factor: float=1.0,
        fill_mask: Optional[List[bool]]=None
    ) -> np.ndarray:
    masks = []
    for i, polyline in enumerate(polylines):
        assert polyline.ndim == 2
        mask = np.zeros((round(img_hw[0] * scale_factor), round(img_hw[1] * scale_factor)), dtype=np.uint8)
        if fill_mask is not None:
            if not fill_mask[i]:
                cv2.polylines(mask, pts=polyline[None], isClosed=False, color=1)
            else:
                cv2.fillPoly(mask, pts=polyline[None], color=1)
        else:
            cv2.polylines(mask, pts=polyline[None], isClosed=False, color=1)
        masks.append(mask)
    masks = np.stack(masks, axis=0)
    return masks


def overlap_masks(masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert masks.ndim == 3
    areas = masks.sum((1, 2))
    sorted_indices = np.argsort(-areas)
    final_mask = np.zeros(masks.shape[1:], dtype=(np.uint8 if masks.shape[0] <= 255 else np.uint32))
    for i, sorted_idx in enumerate(sorted_indices):
        final_mask += (masks[sorted_idx] * (i + 1)).astype(final_mask.dtype)
        final_mask = np.clip(final_mask, a_min=0, a_max=i+1)
    final_mask = final_mask
    return final_mask[None], sorted_indices


def polygons_to_overlapped_mask(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    masks = polylines_to_masks(*args, **kwargs)
    return overlap_masks(masks)
