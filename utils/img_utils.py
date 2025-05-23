import cv2
import torch
import numpy as np
from typing import Union, Tuple, List, Optional

def generate_occupancy_map(
        motion_tracks: Union[np.ndarray, torch.Tensor], 
        map_size: Tuple[int, int], 
        track_colors: Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]=None
    ) -> Union[np.ndarray, torch.Tensor]:

    # motion_tracks shape: (num_objects, tracks, 8)
    assert isinstance(motion_tracks, (np.ndarray, torch.Tensor))
    assert track_colors is None or isinstance(track_colors, (tuple, list))

    track_colors = track_colors or (255, 255, 255)

    if isinstance(track_colors, tuple):
        track_colors = [track_colors]
    
    og_type = type(motion_tracks)
    if isinstance(motion_tracks, torch.Tensor):
        motion_tracks = motion_tracks.cpu().numpy()

    imgs = np.zeros((motion_tracks.shape[0], *map_size), dtype=np.uint8)
    
    xy_pos = motion_tracks[..., :2]

    relative_xy_corners = np.stack([
        np.stack([+motion_tracks[..., 3], +motion_tracks[..., 4]], axis=2),
        np.stack([+motion_tracks[..., 3], -motion_tracks[..., 4]], axis=2),
        np.stack([-motion_tracks[..., 3], +motion_tracks[..., 4]], axis=2),
        np.stack([-motion_tracks[..., 3], -motion_tracks[..., 4]], axis=2)
    ], axis=2)

    heading_angles = motion_tracks[..., -1]
    heading_cos = np.cos(heading_angles)
    heading_sin = np.sin(heading_angles)
    rotations = np.stack(
        [heading_cos, -heading_sin, heading_sin, heading_cos],
    axis=2).reshape(*motion_tracks.shape[:2], 2, 2)

    polygon_tracks = np.matmul(rotations, np.transpose(relative_xy_corners, (0, 1, 3, 2)))
    polygon_tracks = np.transpose(polygon_tracks, (0, 1, 3, 2)) + xy_pos[:, :, None, :]

    for obj_idx in range(0, motion_tracks.shape[0]):
        color = track_colors[0] if len(track_colors) == 1 else track_colors[obj_idx]
        cv2.fillPoly(imgs[obj_idx], pts=polygon_tracks[obj_idx], color=color)
    
    if og_type == torch.Tensor:
        imgs = torch.from_numpy(imgs)
    return imgs
    