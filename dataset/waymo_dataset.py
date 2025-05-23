import math
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.io_utils import load_yaml_file, load_pickle_file, load_json_file, load_and_process_img
from utils.img_utils import generate_occupancy_map
from typing import Union, Dict, Any, Tuple


class WaymoDataset(Dataset):
    def __init__(self, data_or_path: Union[str, Dict[str, Any]], config_or_path: Union[str, Dict[str, Any]]):
        self.config = config_or_path if isinstance(config_or_path, dict) else load_yaml_file(config_or_path)
        self.data = data_or_path if isinstance(data_or_path, dict) else load_pickle_file(data_or_path)
        
        self._set_data_len_and_indexes()


    def _set_data_len_and_indexes(self):
        self._samples_names = list(self.data.keys())
        sample_sizes = [len(self.data[sample]) for sample in self._samples_lists]
        self._start_indexes = np.cumsum([0] + sample_sizes[:-1])
        self._end_indexes = np.concatenate(
            [(self._start_indexes[1:] - 1), [self._start_indexes[-1] + (sample_sizes[-1] - 1)]
        ], axis=0)
        self._data_len = sum(sample_sizes)

    
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


    def __getitem__(self, idx: int):
        sample_name, frame_idx = self._search_for_idx(idx)
        self._load_frame_data(sample_name, frame_idx)


    def _load_frame_data(self, sample_name: str, frame_idx: int, **kwargs):
        sample_dict = self.data[sample_name]
        frame_dict = sample_dict[frame_idx]
        
        # required model data
        cam_views = self._load_cam_views(frame_dict, **kwargs)
        point_cloud = self._load_point_cloud(frame_dict)
        cam_labels = self._load_cam_labels(frame_dict)
        laser_labels = self._load_laser_labels(frame_dict)
        ego_pose, cam_intrinsics, cam_extrinsics, laser_extrinsics = self._load_ego_pose_and_transforms(frame_dict)
        cam_detections = self._load_cam_labels(frame_dict)
        laser_detections = self._load_laser_labels(frame_dict)
        motion_tracks = self._generate_motion_trajectory(sample_dict, frame_idx)
        occupancy_map = self._generate_occupancy_map(motion_tracks)


    def _generate_motion_trajectory(self, sample_dict: Dict[str, Any], frame_idx: int) -> torch.Tensor:
        num_frames = len(sample_dict)
        motion_horizon = self.config["motion_horizon"]
        occ_horizon = self.config["occ_horizon"]
        max_horizon = max(motion_horizon, occ_horizon)
        
        track_maps = {}
        for step in range(0, max_horizon + 1):
            idx = frame_idx + step
            if idx >= num_frames - 1:
                break
            laser_labels_path = sample_dict[idx]["laser_labels_path"]
            laser_labels_list = load_json_file(laser_labels_path)

            for obj_idx in range(0, len(laser_labels_list)):
                obj_id = laser_labels_list[obj_idx]["id"]
                if obj_id not in track_maps:
                    if step == 0:
                        track_maps[obj_id] = []
                    else: continue
                obj_type = laser_labels_list[obj_idx]["type"]
                obj_3d_bbox = laser_labels_list[obj_idx]["box"]
                obj_det = [
                    obj_3d_bbox["center_x"],
                    obj_3d_bbox["center_y"],
                    obj_3d_bbox["center_z"],
                    obj_3d_bbox["length"],
                    obj_3d_bbox["width"],
                    obj_3d_bbox["height"],
                    obj_3d_bbox["heading"],
                    obj_type
                ]
                track_maps[obj_id].append(obj_det)
        # NOTE: Nested tensors is experimental feature and may change behaviour in the future
        # shape: (num_objects, tracks, 8)
        tracks = torch.nested.nested_tensor(list(track_maps.values())).to_padded_tensor(0)
        return tracks

    
    def _generate_occupancy_map(self, motion_tracks: torch.Tensor) -> torch.Tensor:
        occ_horizon = self.config["occ_horizon"]
        occ_map_size = self.config["occ_map_size"]
        return generate_occupancy_map(motion_tracks[:, :occ_horizon, :], occ_map_size)


    def _load_cam_views(self, frame_dict: Dict[str, Any], **kwargs) -> torch.Tensor:
        cam_view_paths = frame_dict["camera_view_paths"]
        cam_view_paths = sorted(cam_view_paths)
        cam_views = []
        for view_path in cam_view_paths:
            view = load_and_process_img(view_path, **kwargs)
            cam_views.append(view)
        cam_views = torch.stack(cam_views, dim=0)
        return cam_views

    
    def _load_point_cloud(self, frame_dict: Dict[str, Any]) -> torch.Tensor:
        point_cloud_path = frame_dict["laser"]["point_cloud_path"]
        point_cloud = np.load(point_cloud_path)
        point_cloud = torch.from_numpy(point_cloud)
        num_cloud_points = self.config["num_cloud_points"]
        point_cloud = F.interpolate(
            point_cloud.permute(1, 0)[None], 
            size=num_cloud_points, 
            mode="linear"
        )[0].permute(1, 0)
        # shape: (num_points, 6)
        return point_cloud
    

    def _load_cam_labels(self, frame_dict: Dict[str, Any]) -> torch.Tensor:
        cam_labels_path = frame_dict["camera_labels_path"]
        cam_labels_dict = load_json_file(cam_labels_path)
        view_keys = sorted(list(cam_labels_dict.keys()))
        obj_dets = []

        for view_key in view_keys:
            labels_list = cam_labels_dict[view_key]
            for obj_idx in range(0, len(labels_list)):
                obj_bbox = labels_list[obj_idx]["box"]
                obj_det = [
                    view_key,
                    obj_bbox["center_x"], 
                    obj_bbox["center_y"], 
                    obj_bbox["length"], 
                    obj_bbox["width"],
                    labels_list[obj_idx]["type"]
                ]
                obj_dets.append(obj_det)
        obj_dets = torch.tensor(obj_dets, dtype=torch.float32)
        # shape: (num_objs, 6)
        return obj_dets

    
    def _load_laser_labels(self, frame_dict: Dict[str, Any]) -> torch.Tensor:
        laser_labels_path = frame_dict["laser_labels_path"]
        laser_labels_list = load_json_file(laser_labels_path)
        obj_3d_dets = []

        for obj_idx in range(0, len(laser_labels_list)):
            obj_3d_bbox = laser_labels_list[obj_idx]["box"]
            obj_3d_dets.append([
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

    
    def _load_ego_pose_and_transforms(self, frame_dict: Dict[str, Any]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        ego_pose = torch.from_numpy(frame_dict["ego_pose"])
        cam_intrinsics_dict = frame_dict["camera_proj_matrix"]["intrinsic"]
        cam_extrinsics_dict = frame_dict["camera_proj_matrix"]["extrinsic"]
        laser_extrinsics_dict = frame_dict["laser_proj_matrix"]["extrinsic"]
        cam_keys = sorted(list(cam_intrinsics_dict.keys()))
        laser_keys = sorted(list(laser_extrinsics_dict.keys()))
        cam_intrinsics = []
        cam_extrinsics = []
        laser_extrinsics = []

        for cam in cam_keys:
            cam_intrinsics.append(cam_intrinsics_dict[cam])
            cam_extrinsics.append(cam_extrinsics_dict[cam])

        for laser in laser_keys:
            laser_extrinsics.append(laser_extrinsics_dict[laser])

        cam_intrinsics = torch.from_numpy(np.stack(cam_intrinsics, axis=0))
        cam_extrinsics = torch.from_numpy(np.stack(cam_extrinsics, axis=0))
        laser_extrinsics = torch.from_numpy(np.stack(laser_extrinsics, axis=0))
        # shape: (4, 4), (3, 3), (4, 4), (4, 4) respectively
        return ego_pose, cam_intrinsics, cam_extrinsics, laser_extrinsics


    def _load_map(self, frame_dict: Dict[str, Any], **kwargs) -> torch.Tensor:
        map_img_path = frame_dict["map_img_path"]
        map_img = load_and_process_img(map_img_path, **kwargs)
        return map_img
    
    
    def _get_timestamp(self, frame_dict: Dict[str, Any]) -> torch.Tensor:
        return torch.tensor(frame_dict["timestamp_seconds"], dtype=torch.float32)
        



