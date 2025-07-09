import cv2
import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Optional, Union


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
    points  = point_clouds[..., :2].clone()
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
        agent_motions: torch.Tensor,
        map_hw: Tuple[int, int], 
        x_min: float=-51.2,
        x_max: float=51.2,
        y_min: float=-51.2,
        y_max: float=51.2,
    ) -> torch.Tensor:
    
    """
    Inputs
    --------------------------------
    :agent_motions: 
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

    Returns
    --------------------------------
    :detection instance level binary BEV map:
        shape: (num_detections, num_timesteps, bev_H, bev_W)
    """
    xy_pos = agent_motions[..., :2]

    num_dets, num_timesteps = agent_motions.shape[:2]

    relative_xy_corners = torch.stack([
        torch.stack([-agent_motions[..., 3], -agent_motions[..., 4]], dim=2),
        torch.stack([+agent_motions[..., 3], -agent_motions[..., 4]], dim=2),
        torch.stack([+agent_motions[..., 3], +agent_motions[..., 4]], dim=2),
        torch.stack([-agent_motions[..., 3], +agent_motions[..., 4]], dim=2)
    ], dim=2)
    relative_xy_corners /= 2

    heading_angles = agent_motions[..., -1]
    heading_cos = torch.cos(heading_angles)
    heading_sin = torch.sin(heading_angles)
    rotations = torch.stack(
        [heading_cos, heading_sin, -heading_sin, heading_cos],
    axis=2).reshape(*agent_motions.shape[:2], 2, 2)

    vertices = torch.matmul(relative_xy_corners, rotations) + xy_pos[:, :, None, :]
    vertices[..., 0] = ((vertices[..., 0] - x_min) / (x_max - x_min)) * (map_hw[1] - 1)
    vertices[..., 1] = ((vertices[..., 1] - y_min) / (y_max - y_min)) * (map_hw[0] - 1)

    vertices = vertices.cpu().to(dtype=torch.int64).numpy()
    occ_maps = np.zeros((*agent_motions.shape[:2], *map_hw), dtype=np.uint8)

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

    points: (N, ..., 2 | 3)  expects either 2D or 3D points
    angle: (N, ...) heading / rotation angle in radiance
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

    return torch.matmul(points, rotation_matrix.transpose(-1, -2))[..., :]


def translate_points(points: torch.Tensor, locs: torch.Tensor) -> torch.Tensor:
    """
    Translate a point to the frame of a reference point.
    
    points: (N, ..., d)  expects either 2D or 3D points
    locs: (N, ..., d)  reference points to translate to, expects either 2D or 3D points
    """
    assert points.shape[-1] == locs.shape[-1]
    return points + locs


def transform_points(
        points: torch.Tensor, 
        *, 
        locs: Optional[torch.Tensor]=None, 
        angles: Optional[torch.Tensor]=None,
        transform_matrix: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
    """
    Apply rotation and then translation.

    points: (N, ..., d)  expects either 2D or 3D points
    angle: (N, ...) heading / rotation angle in radiance
    ref_point: (N, ..., d)  reference points to translate to, expects either 2D or 3D points
    transform_matrix: (N, ..., d, d)
    """
    if angles is not None:
        points = rotate_points(points, angles)

    if locs is not None:
        points = translate_points(points, locs)
    
    if transform_matrix is not None:
        assert transform_matrix.shape[-1] in [2, 3, 4]

        d = points.shape[-1]
        if d == transform_matrix.shape[-1]:
            points = torch.matmul(points, transform_matrix.transpose(-1, -2))[..., :]
        
        elif transform_matrix.shape[-1] - d == 1:
            points = torch.concat([points, torch.ones_like(points[..., [0]])], dim=-1)
            points = torch.matmul(points, transform_matrix.transpose(-1, -2))[..., :d]
        
        elif transform_matrix.shape[-1] - d == 2:
            matmul_pad = torch.ones_like(points[..., :2])
            matmul_pad[..., 0] = 0
            points = torch.concat([points, matmul_pad], dim=-1)
            points = torch.matmul(points, transform_matrix.transpose(-1, -2))[..., :d]
        
        else:
            raise ValueError(f"points ({points.shape}) and transform_matrix ({transform_matrix.shape}) have incompatible shapes")

    return points
    

def overlap_img_masks(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert masks.ndim == 3
    areas = masks.sum((1, 2))
    sorted_indexes = torch.argsort(-areas)

    final_mask = masks[sorted_indexes]
    labels = torch.arange(1, sorted_indexes.shape[0] + 1, step=1, device=masks.device)[:, None, None]
    final_mask = masks * labels
    final_mask = final_mask.max(dim=0).values[None]
    return final_mask, sorted_indexes


def savgol_kernel(num_coefs: int=5, num_power: int=4, device: Union[str, int, torch.device]="cpu") -> torch.Tensor:
    coefs = torch.arange(-(num_coefs//2), num_coefs//2+1, device=device)
    powers = torch.arange(num_power, device=device)
    J = coefs[:, None].pow(powers[None, :]).float()
    JTJ_inv = torch.linalg.inv(J.T @ J)
    M = JTJ_inv @ J.T
    return M


def polylines_to_img_mask(
        polylines: torch.Tensor, 
        img_hw: Tuple[int, int], 
        img_scale: Optional[float]=None,
        pad_vals: Optional[torch.Tensor]=None,
        overlap: bool=True,
        smoothen: bool=False,
        **kwargs
    ) -> Union[torch.Tensor, Optional[torch.Tensor]]:
    """
    polylines: (N, num_vertices, 2) (dtype must be int64), The last axis corresponds to x, y

    img_hw: [H, W] of generated mask image

    new_img_hw: [new_H, new_W], applicable if you wish to increase resolution

    pad_vals: values used to pad the polylines (including EOS and PAD tokens)

    overlap: whether to overlap generated masks of each element (creating a (1, H, W) tensor) 
        or not (N,c H, W) tensor

    smoothen: If True, apply smoothening kernel to the polyline vertices
    """
    assert polylines.dtype == torch.int64 and polylines.ndim == 3 and polylines.shape[2] == 2

    if pad_vals is not None:
        polylines = polylines.clone()
        invalid_mask = torch.isin(polylines, pad_vals)
        b_i, *_ = torch.where(invalid_mask)
        polylines[invalid_mask] = polylines[b_i[0::2], 0, :].flatten(start_dim=0, end_dim=1)

    if img_scale is not None:
        max_xy = torch.tensor([img_hw[1], img_hw[0]], device=polylines.device) - 1
        new_wh = (max_xy + 1) * img_scale
        img_hw = [new_wh[1].int().item(), new_wh[0].int().item()]
        polylines = (new_wh - 1) * (polylines / max_xy)
        polylines = F.interpolate(polylines.transpose(2, 1), scale_factor=img_scale, mode="linear")
        polylines = polylines.transpose(2, 1).long()

    if smoothen:
        kernel = savgol_kernel(**kwargs, device=polylines.device)
        kernel  = kernel[None, None, 0, :].tile(2, 1, 1).flip(-1)
        pad_size = kernel.shape[-1] // 2
        polylines = polylines.transpose(2, 1).float()
        polylines = F.pad(polylines, (pad_size, pad_size), mode="replicate")
        polylines = F.conv1d(polylines, weight=kernel, groups=2)
        polylines = polylines.transpose(2, 1).long()

    polylines[..., 0] = polylines[..., 0].clip(min=0, max=img_hw[1]-1)
    polylines[..., 1] = polylines[..., 1].clip(min=0, max=img_hw[0]-1)
        
    device = polylines.device
    batch_indexes = torch.arange(polylines.shape[0], device=device)[:, None].tile(1, polylines.shape[1])
    img_masks = torch.zeros(polylines.shape[0], *img_hw, device=device, dtype=torch.bool)
    img_masks[batch_indexes, polylines[:, :, 1], polylines[:, :, 0]] = 1

    sorted_indexes = None
    if overlap:
        img_masks, sorted_indexes = overlap_img_masks(img_masks)
    return img_masks, sorted_indexes