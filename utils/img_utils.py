import cv2
import torch
import numpy as np
from typing import Tuple, Optional, List, Union


def _safe_nanmin(data: torch.Tensor, dim: int) -> torch.Tensor:
    data = data.clone()
    data[torch.isnan(data)] = torch.inf
    return data.min(dim).values


def _safe_nanmax(data: torch.Tensor, dim: int) -> torch.Tensor:
    data = data.clone()
    data[torch.isnan(data)] = -torch.inf
    return data.max(dim).values


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

    binary_map = torch.zeros(point_clouds.shape[0], *map_hw, dtype=torch.float32, device=point_clouds.device)
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

    occ_maps = occ_maps.astype(np.float32)
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


def xywh_to_xyxy(bboxes: torch.Tensor) -> torch.Tensor:
    """convert xywh -> xyxy"""
    x1y1 = bboxes[..., :2] - (bboxes[..., 2:] / 2)
    x2y2 = x1y1 + bboxes[..., 2:]
    bboxes = torch.concat([x1y1, x2y2], dim=-1)
    return bboxes


def xyxy_to_xywh(bboxes: torch.Tensor) -> torch.Tensor:
    """convert xyxy -> xywh"""
    wh = bboxes[..., 2:] - bboxes[..., :2]
    xy = bboxes[..., :2] + (wh / 2)
    bboxes = torch.concat([xy, wh], axis=-1)
    return bboxes


def polyline_2_xywh(
        polygons: torch.Tensor, 
        pad_mask_vals: Optional[torch.Tensor]=None
    ) -> torch.Tensor:

    if pad_mask_vals is not None:
        assert pad_mask_vals.ndim == 1

    assert polygons.ndim == 3

    if pad_mask_vals is not None:
        polygons = polygons.clone().float()
        polygons[torch.isin(polygons, pad_mask_vals)] = torch.nan
        x1, y1, x2, y2 = (
            _safe_nanmin(polygons[..., 0], dim=1), 
            _safe_nanmin(polygons[..., 1], dim=1),
            _safe_nanmax(polygons[..., 0], dim=1),
            _safe_nanmax(polygons[..., 1], dim=1)
        )
    else:
        x1, y1, x2, y2 = (
            polygons[..., 0].min(dim=1).values,
            polygons[..., 1].min(dim=1).values,
            polygons[..., 0].max(dim=1).values,
            polygons[..., 1].max(dim=1).values
        )
    w, h = x2 - x1, y2 - y1
    x, y = x1 + (w/2), y1 + (h/2)
    bboxes = torch.stack([x, y, w, h], dim=-1)
    return bboxes


def rotate_points(points: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rotate a 2D point or rotate a 3D point about the z-axis.

    points: (N, num_objects, 2 | 3)  expects either 2D or 3D points
    angle: (N, num_objects | 1) heading / rotation angle in radiance
    """
    assert points.shape[-1] == 2 or points.shape[-1] == 3
    
    angle_cos = torch.cos(angle)
    angle_sin = torch.sin(angle)

    if points.shape[-1] == 2:
        rotation_matrix = torch.stack([angle_cos, angle_sin, -angle_sin, angle_cos], dim=-1)
        rotation_matrix = torch.unflatten(rotation_matrix, dim=-1, sizes=(2, 2))
    
    else:
        zeros = torch.zeros_like(angle_cos)
        ones = torch.ones_like(angle_cos)
        rotation_matrix = torch.stack([angle_cos, angle_sin, zeros, -angle_sin, angle_cos, zeros, zeros, zeros, ones], dim=-1)
        rotation_matrix = torch.unflatten(rotation_matrix, dim=-1, sizes=(3, 3))

    return torch.matmul(points[..., None, :], rotation_matrix.permute(0, 1, 3, 2))[..., 0, :]


def translate_points(points: torch.Tensor, ref_point: torch.Tensor) -> torch.Tensor:
    """
    Translate a point to the frame of a reference point.
    
    points: (N, num_objects, 2 | 3)  expects either 2D or 3D points
    ref_point: (N, 1, 2 | 3)  expects either 2D or 3D points
    """
    assert points.shape[-1] == ref_point.shape[-1]
    if ref_point.ndim == 2:
        ref_point = ref_point[:, None, :]
    return points - ref_point


def transform_points(
        points: torch.Tensor, 
        *, 
        ref_points: Optional[torch.Tensor]=None, 
        angles: Optional[torch.Tensor]=None,
        transform_matrix: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
    """
    Apply rotation and then translation.

    points: (N, 1 | num_objects, 2 | 3)  expects either 2D or 3D points
    angle: (N, num_objects | 1) heading / rotation angle in radiance
    ref_point: (N, 1 | num_objects, 2 | 3)  expects either 2D or 3D points
    transform_matrix: (N, 1 | num_objects, 2, 2) | (N, 1 | num_objects, 3, 3)
    """
    if ref_points is not None and angles is not None:
        return translate_points(rotate_points(points, angles), ref_points)
    
    elif transform_matrix is not None:
        return torch.matmul(points[..., None, :], transform_matrix.permute(0, 1, 3, 2))[..., 0, :]

    else:
        raise ValueError(
            "expects angle and ref_points for rotation and translation respectively, or transform_matrix for transformation"
        )
    

def overlap_img_masks(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert masks.ndim == 3
    areas = masks.sum((1, 2))
    sorted_indexes = torch.argsort(-areas)

    final_mask = masks[sorted_indexes]
    labels = torch.arange(1, sorted_indexes.shape[0] + 1, step=1, device=masks.device)[:, None, None]
    final_mask = masks * labels
    final_mask = final_mask.max(dim=0).values[None]
    return final_mask, sorted_indexes


def polylines_to_img_mask(
        polylines: torch.Tensor, 
        img_hw: Tuple[int, int], 
        pad_vals: Optional[torch.Tensor]=None,
        overlap: bool=True
    ) -> Union[torch.Tensor, Optional[torch.Tensor]]:
    """
    polylines: (N, num_vertices, 2) (dtype must be int64), The last axis corresponds to x, y

    img_hw: [H, W] of generated mask image

    pad_vals: values used to pad the polylines (including EOS and PAD tokens)

    overlap: whether to overlap generated masks of each element (creating a (1, H, W) tensor) 
        or not (N,c H, W) tensor
    """
    assert polylines.dtype == torch.int64 and polylines.ndim == 3 and polylines.shape[2] == 2

    if pad_vals is not None:
        polylines = polylines.clone()
        invalid_mask = torch.isin(polylines, pad_vals)
        b_i, *_ = torch.where(invalid_mask)
        polylines[invalid_mask] = polylines[b_i[0::2], 0, :].flatten(start_dim=0, end_dim=1)
        
    device = polylines.device
    batch_indexes = torch.arange(polylines.shape[0], device=device)[:, None].tile(1, polylines.shape[1])
    img_masks = torch.zeros(polylines.shape[0], *img_hw, device=device, dtype=torch.bool)
    img_masks[batch_indexes, polylines[:, :, 1], polylines[:, :, 0]] = 1

    sorted_indexes = None
    if overlap:
        img_masks, sorted_indexes = overlap_img_masks(img_masks)
    return img_masks, sorted_indexes