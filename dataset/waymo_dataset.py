import math
import torch
import time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.io_utils import load_pickle_file
from utils.img_utils import generate_occupancy_map
from ._container import FrameData, MultiFrameData, BatchMultiFrameData
from typing import Union, Dict, Any, Tuple, Optional, List


def check_perf(func):
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        out = func(self, *args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} time: {(end - start) * 1000} ms")
        return out
    return wrapper

class WaymoDataset(Dataset):
    def __init__(
            self, 
            data_or_path: Union[str, Dict[str, Any]], 
            motion_horizon: int=12,
            occupancy_horizon: int=5,
            planning_horizon: int=6,
            bev_map_hw: Tuple[int, int]=(200, 200),
            cam_img_hw: Tuple[int, int]=(480, 640),
            num_cloud_points: int=100_000,
            xyz_range: Optional[List[Tuple[int, int]]] = None,
            frames_per_sample: int=3
        ):
        self.data = data_or_path if isinstance(data_or_path, dict) else load_pickle_file(data_or_path)
        self.motion_horizon = motion_horizon
        self.occupancy_horizon = occupancy_horizon
        self.planning_horizon = planning_horizon
        self.bev_map_hw = bev_map_hw
        self.cam_img_hw = cam_img_hw
        self.num_cloud_points = num_cloud_points
        self.xyz_range = xyz_range or [(-51.2, 51.2), (-51.2, 51.2), (-5.0, 3.0)]
        self.frames_per_sample = frames_per_sample
        
        self._set_data_len_and_indexes()


    def _set_data_len_and_indexes(self):
        self._samples_names = list(self.data.keys())
        sample_sizes = [len(self.data[sample]) for sample in self._samples_names]
        self._start_indexes = np.cumsum([0] + sample_sizes[:-1])
        self._end_indexes = np.concatenate(
            [(self._start_indexes[1:] - 1), [self._start_indexes[-1] + (sample_sizes[-1] - 1)]
        ], axis=0)
        self._data_len = sum(sample_sizes)

    @check_perf
    def _search_for_idx(self, idx: int) -> Tuple[str, int]:
        op_len = len(self._start_indexes)
        center_idx = op_len // 2
        if idx < self._start_indexes[0] or idx > self._end_indexes[-1]:
            raise IndexError((
                f"index {idx} is out of range for dataset with index"
                f" range {self._start_indexes[0]} to {self._end_indexes[-1]}"
            ))

        while True:
            start_idx = self._start_indexes[center_idx]
            end_idx   = self._end_indexes[center_idx]

            if start_idx <= idx and idx <= end_idx:
                sample_name = self._samples_names[center_idx]
                frame_idx = idx - start_idx
                return sample_name, frame_idx
            
            elif start_idx > idx:
                op_len //= 2
                center_idx -= math.ceil(op_len / 2)
                continue

            elif end_idx < idx:
                op_len //= 2
                center_idx += math.ceil(op_len / 2)
                continue


    def __len__(self) -> int:
        return self._data_len


    def __getitem__(self, idx: int) -> MultiFrameData:
        sample_name, frame_idx = self._search_for_idx(idx)
        sample_dict = self.data[sample_name]

        start = max(0, (idx - self.frames_per_sample) + 1)
        end = idx + 1

        frames = []
        for i in range(start, end):
            frames.append(self._load_frame_data(sample_dict, frame_idx, is_partial_data=(i != idx)))
        return MultiFrameData.from_framedata_list(frames)


    def _load_frame_data(self, sample_dict: Dict[str, Any], frame_idx: int, is_partial_data: bool) -> FrameData:
        frame_dict = sample_dict[frame_idx]

        cam_views = self._load_cam_views(frame_dict)
        point_cloud = self._load_point_cloud(frame_dict)
        laser_detections = self._load_laser_labels(frame_dict)
        
        ego_pose = None
        cam_intrinsic = None
        cam_extrinsics = None
        motion_tracks = None
        occupancy_map = None
        bev_road_map = None
        ego_trajectory = None
        motion_tracks = None

        if not is_partial_data:
            ego_pose, cam_intrinsic, cam_extrinsics = self._load_ego_pose_and_transforms(frame_dict)
            motion_tracks = self._generate_motion_trajectory(sample_dict, frame_idx)
            occupancy_map = self._generate_occupancy_map(motion_tracks)
            bev_road_map = self._load_bev_road_map(frame_dict)
            ego_trajectory = self._get_planning_trajectory(sample_dict, frame_idx, ego_pose)
            motion_tracks = motion_tracks[:, 1:, :2]

        return FrameData(
            cam_views=cam_views,
            point_cloud=point_cloud,
            ego_pose=ego_pose,
            laser_detections=laser_detections,
            cam_intrinsic=cam_intrinsic,
            cam_extrinsics=cam_extrinsics,
            motion_tracks=motion_tracks,
            occupancy_map=occupancy_map,
            bev_road_map=bev_road_map,
            ego_trajectory=ego_trajectory
        )


    @check_perf
    def _generate_motion_trajectory(self, sample_dict: Dict[str, Any], frame_idx: int) -> torch.Tensor:
        num_frames = len(sample_dict)
        max_horizon = max(self.motion_horizon, self.occupancy_horizon)
        
        track_maps = {}
        for idx in range(frame_idx, min(num_frames, frame_idx + max_horizon + 1)):
            laser_labels_path = sample_dict[idx]["laser_labels_path"]
            laser_labels_list = load_pickle_file(laser_labels_path)

            for obj_idx in range(0, len(laser_labels_list)):
                obj_id = laser_labels_list[obj_idx]["id"]
                obj_type = laser_labels_list[obj_idx]["type"]
                obj_3d_bbox = laser_labels_list[obj_idx]["box"]
                if (
                    (obj_3d_bbox["center_x"] < self.xyz_range[0][0] or obj_3d_bbox["center_x"] > self.xyz_range[0][1])
                    or (obj_3d_bbox["center_y"] < self.xyz_range[1][0] or obj_3d_bbox["center_y"] > self.xyz_range[1][1])
                    or (obj_3d_bbox["center_z"] < self.xyz_range[2][0] or obj_3d_bbox["center_z"] > self.xyz_range[2][1])
                ): continue
                
                if obj_id not in track_maps:
                    if idx == frame_idx:
                        track_maps[obj_id] = []
                    else: continue
                
                obj_det = [
                    obj_3d_bbox["center_x"],
                    obj_3d_bbox["center_y"],
                    obj_3d_bbox["center_z"],
                    obj_3d_bbox["length"],
                    obj_3d_bbox["width"],
                    obj_3d_bbox["height"],
                    obj_3d_bbox["heading"],
                ]
                track_maps[obj_id].append(obj_det)
        # NOTE: Nested tensors is experimental feature and may change behaviour in the future
        # shape: (num_detections, motion_timesteps, 8)
        tracks = torch.nested.nested_tensor(list(track_maps.values())).to_padded_tensor(0)
        return tracks

    
    @check_perf
    def _generate_occupancy_map(self, motion_tracks: torch.Tensor) -> torch.Tensor:
        # shape: (num_detections, timesteps, H_bev, W_bev)
        occ_map = generate_occupancy_map(
            motion_tracks[:, :self.occupancy_horizon, :], 
            map_hw=self.bev_map_hw,
            x_min=self.xyz_range[0][0],
            x_max=self.xyz_range[0][1],
            y_min=self.xyz_range[1][0],
            y_max=self.xyz_range[1][1],
            point_clouds=None
        )
        return occ_map


    @check_perf
    def _load_cam_views(self, frame_dict: Dict[str, Any]) -> torch.Tensor:
        cam_view_path = frame_dict["camera_view_path"]
        cam_views = np.load(cam_view_path)
        cam_views = torch.from_numpy(cam_views) / 255
        if cam_views.shape[2] != self.cam_img_hw[0] or cam_views.shape[3] != self.cam_img_hw[1]:
            cam_views = F.interpolate(cam_views, size=self.cam_img_hw, mode="bilinear")
        return cam_views

    
    @check_perf
    def _load_point_cloud(self, frame_dict: Dict[str, Any]) -> torch.Tensor:
        point_cloud_path = frame_dict["laser"]["point_cloud_path"]
        point_cloud = np.load(point_cloud_path)
        point_cloud = torch.from_numpy(point_cloud)
        if point_cloud.shape[0] != self.num_cloud_points:
            point_cloud = F.interpolate(point_cloud.permute(1, 0)[None], size=self.num_cloud_points, mode="linear")
            point_cloud = point_cloud[0].permute(1, 0)
        # shape: (num_points, 4)
        return point_cloud[..., [3, 4, 5, 0]]


    @check_perf
    def _load_laser_labels(self, frame_dict: Dict[str, Any]) -> torch.Tensor:
        laser_labels_path = frame_dict["laser_labels_path"]
        laser_labels_list = load_pickle_file(laser_labels_path)
        obj_3d_dets = []

        for obj_idx in range(0, len(laser_labels_list)):
            obj_3d_bbox = laser_labels_list[obj_idx]["box"]
            if (
                (obj_3d_bbox["center_x"] < self.xyz_range[0][0] or obj_3d_bbox["center_x"] > self.xyz_range[0][1])
                or (obj_3d_bbox["center_y"] < self.xyz_range[1][0] or obj_3d_bbox["center_y"] > self.xyz_range[1][1])
                or (obj_3d_bbox["center_z"] < self.xyz_range[2][0] or obj_3d_bbox["center_z"] > self.xyz_range[2][1])
            ): continue
            obj_3d_dets.append([
                0,
                0,
                obj_3d_bbox["center_x"],
                obj_3d_bbox["center_y"],
                obj_3d_bbox["center_z"],
                obj_3d_bbox["length"],
                obj_3d_bbox["width"],
                obj_3d_bbox["height"],
                obj_3d_bbox["heading"],
                laser_labels_list[obj_idx]["type"]
            ])
        obj_3d_dets = torch.tensor(obj_3d_dets, dtype=torch.float32)
        # shape: (num_objs, 8)
        return obj_3d_dets

    
    @check_perf
    def _load_ego_pose_and_transforms(self, frame_dict: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        ego_pose = torch.from_numpy(frame_dict["ego_pose"])
        cam_intrinsic_dict = frame_dict["camera_proj_matrix"]["intrinsic"]
        cam_extrinsics_dict = frame_dict["camera_proj_matrix"]["extrinsic"]
        cam_keys = sorted(list(cam_intrinsic_dict.keys()))
        cam_intrinsic = []
        cam_extrinsics = []

        for cam in cam_keys:
            cam_intrinsic.append(cam_intrinsic_dict[cam])
            cam_extrinsics.append(cam_extrinsics_dict[cam])

        cam_intrinsic = torch.from_numpy(np.stack(cam_intrinsic, axis=0))
        cam_extrinsics = torch.from_numpy(np.stack(cam_extrinsics, axis=0))
        # shape: (4, 4), (3, 3), (4, 4), (4, 4) respectively
        return ego_pose, cam_intrinsic, cam_extrinsics


    @check_perf
    def _load_bev_road_map(self, frame_dict: Dict[str, Any]) -> torch.Tensor:
        map_img_path = frame_dict["map_img_path"]
        map_img = np.load(map_img_path)
        map_img = torch.from_numpy(map_img) / 255
        if map_img.shape[1] != self.bev_map_hw[0] or map_img.shape[2] != self.bev_map_hw[1]:
            map_img = F.interpolate(map_img[None], size=self.bev_map_hw, mode="bilinear")[0]
        return map_img
    

    @check_perf
    def _get_planning_trajectory(
        self, 
        sample_dict: Dict[str, Any],
        frame_idx: int,
        ego_pose: torch.Tensor
    ) -> torch.Tensor:
        num_samples = len(sample_dict)
        if frame_idx + 1 == num_samples:
            return torch.tensor([], dtype=torch.float32)
        global_positions = [
            sample_dict[idx]["ego_pose"][:, -1] 
            for idx in range(frame_idx + 1, min(num_samples, frame_idx + self.planning_horizon + 1))
        ]
        global_positions = np.stack(global_positions, axis=0)
        global_positions = torch.from_numpy(global_positions)
        ego_positions = torch.matmul(global_positions, torch.linalg.inv(ego_pose).permute(1, 0))
        # shape: (timesteps, 2)
        return ego_positions[:, :2]
        
    
    @check_perf
    def _get_timestamp(self, frame_dict: Dict[str, Any]) -> torch.Tensor:
        return torch.tensor(frame_dict["timestamp_seconds"], dtype=torch.float32)
    
    
    @staticmethod
    def collate_fn(batch: List[MultiFrameData]) -> BatchMultiFrameData:
        return BatchMultiFrameData.from_multiframedata_list(batch)