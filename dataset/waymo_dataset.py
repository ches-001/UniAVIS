import math
import torch
import time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.io_utils import load_pickle_file, load_json_file
from utils.img_utils import generate_occupancy_map
from utils.img_utils import polyline_2_xywh
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
            metadata_or_path: Union[str, Dict[str, Any]],
            motion_horizon: int=12,
            occupancy_horizon: int=5,
            planning_horizon: int=6,
            bev_map_hw: Tuple[int, int]=(200, 200),
            cam_img_hw: Tuple[int, int]=(480, 640),
            num_cloud_points: int=100_000,
            xyz_range: Optional[List[Tuple[float, float]]]=None,
            frames_per_sample: int=4,
            sample_period_secs: Optional[float]=2.0,
            max_num_agents: int=100,
            max_num_map_elements: int=100,
            max_num_polyline_points: int=1000,
            motion_sample_freq: float=2.0,
            pad_polylines_out_of_index: bool=False,
            motion_command_opts: Optional[Dict[str, Any]]=None,
            stop_command_disp_thresh: float=0.5
        ):
        self.data = (
            data_or_path
            if isinstance(data_or_path, dict)
            else load_pickle_file(data_or_path)
        )
        self.metadata = (
            metadata_or_path
            if isinstance(metadata_or_path, dict)
            else load_json_file(metadata_or_path)
        )
        self.motion_horizon = motion_horizon
        self.occupancy_horizon = occupancy_horizon
        self.planning_horizon = planning_horizon
        self.bev_map_hw = bev_map_hw
        self.cam_img_hw = cam_img_hw
        self.num_cloud_points = num_cloud_points
        self.xyz_range = xyz_range or [(-51.2, 51.2), (-51.2, 51.2), (-5.0, 3.0)]
        self.frames_per_sample = frames_per_sample
        self.sample_period_secs = sample_period_secs
        self.max_num_agents = max_num_agents
        self.max_num_map_elements = max_num_map_elements
        self.max_num_polyline_points = max_num_polyline_points
        self.motion_sample_freq = motion_sample_freq
        self.pad_polylines_out_of_index = pad_polylines_out_of_index
        self.stop_command_disp_thresh = stop_command_disp_thresh
        self.motion_command_opts = motion_command_opts or {
            "straight": (0, -10, 10),
            "left": (1, 10, float("inf")),
            "right": (2, -float("inf"), -10),
        }
        
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

    
    @check_perf
    def _sample_frame_indexes_from_horizon(self, sample_dict: Dict[str, Any], frame_idx: int) -> np.ndarray:
        if not self.sample_period_secs:
            start = max(0, frame_idx - self.frames_per_sample + 1)
            end = frame_idx + 1
            return np.arange(start, end, step=1)

        start_idx = frame_idx
        current_timestep = sample_dict[frame_idx]["timestamp_seconds"]

        while start_idx > max(0, frame_idx - self.frames_per_sample + 1):
            prev_timestep = sample_dict[start_idx]["timestamp_seconds"]
            if current_timestep - prev_timestep >= self.sample_period_secs:
                break
            start_idx -= 1
        
        if start_idx == frame_idx:
            return np.asarray([frame_idx])
        
        size = min(self.frames_per_sample, frame_idx - start_idx)
        index_pool = np.random.choice(np.arange(start_idx, frame_idx, step=1), size=size, replace=False)
        index_pool.sort()
        index_pool = np.concatenate([index_pool, [frame_idx]], axis=0)
        return index_pool


    def __len__(self) -> int:
        return self._data_len


    def __getitem__(self, idx: int) -> MultiFrameData:
        sample_name, frame_idx = self._search_for_idx(idx)
        sample_dict = self.data[sample_name]

        frames = []
        obj_id_map = {}
        frame_indexes = self._sample_frame_indexes_from_horizon(sample_dict, frame_idx)
        for i in frame_indexes:
            frames.append(self._load_frame_data(
                sample_dict, i, is_partial_data=(i != frame_idx), obj_id_map=obj_id_map
            ))
        return MultiFrameData.from_framedata_list(frames)


    def _load_frame_data(
            self, 
            sample_dict: Dict[str, Any], 
            frame_idx: int, 
            is_partial_data: bool, 
            obj_id_map: Dict[str, int]

        ) -> FrameData:
        frame_dict = sample_dict[frame_idx]

        cam_views = self._load_cam_views(frame_dict)
        point_cloud = self._load_point_cloud(frame_dict)
        laser_detections = self._load_laser_labels(frame_dict, obj_id_map=obj_id_map)
        ego_pose, cam_intrinsic, cam_extrinsic = self._load_ego_pose_and_transforms(
            frame_dict, only_pose=is_partial_data
        )            

        motion_tracks = None
        occupancy_map = None
        map_elements_polylines = None
        map_elements_boxes = None
        ego_trajectory = None
        motion_tracks = None
        command = None

        if not is_partial_data:
            motion_tracks = self._generate_motion_trajectory(sample_dict, frame_idx)
            occupancy_map = self._generate_occupancy_map(motion_tracks)
            ego_trajectory = self._get_planning_trajectory(sample_dict, frame_idx, ego_pose)
            command = self._get_high_level_command(ego_trajectory)
            map_elements_polylines, map_elements_boxes = self._load_bev_map_elements(frame_dict)
            motion_tracks = motion_tracks[:, 1:, :2]

            # pad targets
            map_element_pad = max(0, self.max_num_map_elements - map_elements_polylines.shape[0])
            agents_pad = max(0, self.max_num_agents - motion_tracks.shape[0])
            ego_plan_pad = max(0, self.planning_horizon - ego_trajectory.shape[0])
            map_label_pad_val = len(self.metadata["LABELS"]["MAP_ELEMENT_LABEL_INDEXES"])
            map_point_pad_val = -999
            
            map_elements_polylines = F.pad(
                map_elements_polylines, pad=(0, 0, 0, 0, 0, map_element_pad), mode="constant", value=map_point_pad_val
            )
            map_elements_boxes = F.pad(
                map_elements_boxes, pad=(0, 0, 0, map_element_pad), mode="constant", value=map_label_pad_val
            )
            motion_tracks = F.pad(
                motion_tracks, pad=(0, 0, 0, 0, 0, agents_pad), mode="constant", value=-999
            )
            occupancy_map = F.pad(
                occupancy_map, pad=(0, 0, 0, 0, 0, 0, 0, agents_pad), mode="constant", value=-999
            )

            ego_trajectory = F.pad(
                ego_trajectory, pad=(0, 0, 0, ego_plan_pad), mode="constant", value=-999
            )
        
        det_pad = self.max_num_agents - laser_detections.shape[0]
        det_label_pad_val = len(self.metadata["LABELS"]["DETECTION_LABEL_INDEXES"])
        laser_detections = F.pad(laser_detections, pad=(0, 0, 0, det_pad), mode="constant", value=-999)
        laser_detections[:, -1][laser_detections[:, -1] == -999] = det_label_pad_val

        frame_data = FrameData(
            cam_views=cam_views,
            point_cloud=point_cloud,
            ego_pose=ego_pose,
            laser_detections=laser_detections,
            cam_intrinsic=cam_intrinsic,
            cam_extrinsic=cam_extrinsic,
            motion_tracks=motion_tracks,
            occupancy_map=occupancy_map,
            map_elements_polylines=map_elements_polylines,
            map_elements_boxes=map_elements_boxes,
            ego_trajectory=ego_trajectory,
            command=command
        )
        return frame_data


    @check_perf
    def _generate_motion_trajectory(self, sample_dict: Dict[str, Any], frame_idx: int) -> torch.Tensor:
        num_frames = len(sample_dict)
        max_horizon = max(self.motion_horizon, self.occupancy_horizon)
        
        current_timestamp = sample_dict[frame_idx]["timestamp_seconds"]
        next_timestamp = sample_dict[frame_idx + 1]["timestamp_seconds"]

        dt = next_timestamp - current_timestamp
        iter_start = frame_idx
        iter_step = max(1, round(1 / (dt * self.motion_sample_freq)))
        iter_end = min(num_frames, frame_idx + (iter_step * max_horizon) + 1)

        track_maps = {}
        for idx in range(iter_start, iter_end, iter_step):
            laser_labels_path = sample_dict[idx]["laser_labels_path"]
            laser_labels_list = load_pickle_file(laser_labels_path)

            for obj_idx in range(0, len(laser_labels_list)):
                obj_id = laser_labels_list[obj_idx]["id"]
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
        tracks = torch.nested.nested_tensor(list(track_maps.values())).to_padded_tensor(-999)
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
        
        # shape: (num_points, 4), ), (x, y, z, intensity)
        point_cloud = point_cloud[..., [3, 4, 5, 1]]
        points_mask = (
            (point_cloud[:, 0] >= self.xyz_range[0][0]) & (point_cloud[:, 0] <= self.xyz_range[0][1]) &
            (point_cloud[:, 1] >= self.xyz_range[1][0]) & (point_cloud[:, 1] <= self.xyz_range[1][1]) &
            (point_cloud[:, 2] >= self.xyz_range[2][0]) & (point_cloud[:, 2] <= self.xyz_range[2][1])
        )
        point_cloud = point_cloud[points_mask]

        if point_cloud.shape[0] != self.num_cloud_points:
            point_cloud = F.interpolate(point_cloud.permute(1, 0)[None], size=self.num_cloud_points, mode="linear")
            point_cloud = point_cloud[0].permute(1, 0)

        point_cloud = point_cloud.to(dtype=torch.float32)
        point_cloud[..., -1] = torch.tanh(point_cloud[..., -1])
        return point_cloud


    @check_perf
    def _load_laser_labels(self, frame_dict: Dict[str, Any], obj_id_map: Dict[str, int]) -> torch.Tensor:
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

            obj_id = laser_labels_list[obj_idx]["id"]
            if obj_id not in obj_id_map:
                obj_id_map[obj_id] = len(obj_id_map)

            obj_3d_dets.append([
                obj_id_map[obj_id],
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
        # shape: (num_detections, 9)
        return obj_3d_dets

    
    @check_perf
    def _load_ego_pose_and_transforms(self, frame_dict: Dict[str, Any], only_pose: bool=False) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        ego_pose = torch.from_numpy(frame_dict["ego_pose"])
        if only_pose:
            return ego_pose, None, None
        cam_intrinsic_dict = frame_dict["camera_proj_matrix"]["intrinsic"]
        cam_extrinsic_dict = frame_dict["camera_proj_matrix"]["extrinsic"]
        cam_keys = sorted(list(cam_intrinsic_dict.keys()))
        cam_intrinsic = []
        cam_extrinsic = []

        for cam in cam_keys:
            cam_intrinsic.append(cam_intrinsic_dict[cam])
            cam_extrinsic.append(cam_extrinsic_dict[cam])

        cam_intrinsic = torch.from_numpy(np.stack(cam_intrinsic, axis=0))
        cam_extrinsic = torch.from_numpy(np.stack(cam_extrinsic, axis=0))
        # shape: (4, 4), (3, 3), (4, 4), (4, 4) respectively
        return ego_pose, cam_intrinsic, cam_extrinsic


    @check_perf
    def _load_bev_map_elements(
        self, 
        frame_dict: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        map_element_dict = frame_dict["map_elements"]
        polylines_path = map_element_dict["polylines_path"]
        eos_token = map_element_dict["eos_token"]
        pad_token = map_element_dict["pad_token"]

        polylines = np.load(polylines_path)

        cls = polylines[:, 0:1]
        polylines = polylines[:, 1:]
        polylines = polylines.reshape(polylines.shape[0], polylines.shape[1] // 2, 2)

        points_pad_size = max(0, self.max_num_polyline_points - polylines.shape[1])
        polylines = np.pad(polylines, ((0, 0), (0, points_pad_size), (0, 0)), constant_values=pad_token)

        eos_mask = (polylines == eos_token)
        pad_mask = (polylines == pad_token)

        xy_min = np.asarray([self.xyz_range[0][0], self.xyz_range[1][0]])
        xy_max = np.asarray([self.xyz_range[0][1], self.xyz_range[1][1]])
        polylines = (polylines - xy_min) / (xy_max - xy_min)
        polylines *= np.asarray([self.bev_map_hw[1] - 1, self.bev_map_hw[0] - 1])
        polylines = np.floor(polylines).astype(int)

        eos_token = max(self.bev_map_hw)
        pad_token = eos_token + 1

        polylines[eos_mask] = eos_token
        polylines[pad_mask] = pad_token
        pad_vals = None

        if not self.pad_polylines_out_of_index:
            # pad within index, specifically replace EOS and PAD elements with the first valid vertex for each
            # polyline
            polylines = np.copy(polylines)
            invalid_mask = np.isin(polylines, [eos_token, pad_token])
            b_i, *_ = np.where(invalid_mask)
            polylines[invalid_mask] = polylines[b_i[0::2], 0, :].reshape(-1)

        else:
            pad_vals = torch.tensor([eos_token, pad_token])

        polylines = torch.from_numpy(polylines)
        polyline_boxes = polyline_2_xywh(polylines, pad_vals)

        # scale boxes
        polyline_boxes[..., [0, 2]] /= self.bev_map_hw[1]
        polyline_boxes[..., [1, 3]] /= self.bev_map_hw[0]
        cls = torch.from_numpy(cls)
        boxes = torch.concat([polyline_boxes, cls], dim=-1)
        
        # polylines: (num_elements, num_points, 2)
        # boxes: (num_elements, 4 + 1)
        return polylines, boxes
    

    @check_perf
    def _get_planning_trajectory(
        self, 
        sample_dict: Dict[str, Any],
        frame_idx: int,
        ego_pose: torch.Tensor
    ) -> torch.Tensor:
        num_frames = len(sample_dict)

        if frame_idx + 1 == num_frames:
            return torch.tensor([], dtype=torch.float32)
            
        current_timestamp = sample_dict[frame_idx]["timestamp_seconds"]
        next_timestamp = sample_dict[frame_idx + 1]["timestamp_seconds"]

        horizon = max(self.planning_horizon, self.motion_horizon)
        
        dt = next_timestamp - current_timestamp
        iter_start = frame_idx + 1
        iter_step = max(1, round(1 / (dt * self.motion_sample_freq)))
        iter_end = min(num_frames, frame_idx + (iter_step * horizon) + 1)

        global_positions = [sample_dict[idx]["ego_pose"][:, -1] for idx in range(iter_start, iter_end, iter_step)]
        global_positions = np.stack(global_positions, axis=0)
        global_positions = torch.from_numpy(global_positions)
        ego_trajectory = torch.matmul(global_positions, torch.linalg.inv(ego_pose).permute(1, 0))
        # shape: (timesteps, 2)
        return ego_trajectory[:, :2].to(dtype=torch.float32)
    

    @check_perf
    def _get_high_level_command(self, ego_trajectory: torch.Tensor) -> torch.Tensor:
        # to get high level command of ego planning motion, compute the cross and dot roducts of the last
        # direction vector, relative to a reference vector. The reference vector in this case is [1.0, 0.0]
        # which corresponds to straight motion along the x axis, and the last direction vector is [xt - x0, yt - y0]
        # which is essentially equal to [xt, yt], because the ego vehicle is always situated at the reference point 
        # [0, 0]. Next we compute the arctan2 between the cross and dot product, this is equivalent to computing the
        # arctan2 between the sin and cos of the angles between the two vectors.
        x0, y0 = 1.0, 0.0
        xt, yt = ego_trajectory[min(self.planning_horizon, ego_trajectory.shape[0]) - 1]

        disp = torch.sqrt(xt.pow(2) + yt.pow(2))
        if disp < self.stop_command_disp_thresh:
            return torch.tensor([len(self.motion_command_opts)], dtype=torch.int64)

        cross_prod = (x0 * yt) - (y0 * xt)
        dot_prod = (x0 * xt) + (y0 * yt)
        angle = torch.arctan2(cross_prod, dot_prod)
        angle = torch.rad2deg(angle)
        for k in self.motion_command_opts:
            opt = self.motion_command_opts[k]
            if angle >= opt[1] and angle < opt[2]:
                return torch.tensor([opt[0]], dtype=torch.int64)
        raise Exception(f"angle: {angle} degs not covered in motion_command_opts {self.motion_command_opts}")
        
    
    @check_perf
    def _get_timestamp(self, frame_dict: Dict[str, Any]) -> torch.Tensor:
        return torch.tensor(frame_dict["timestamp_seconds"], dtype=torch.float32)
    
    
    @staticmethod
    def collate_fn(batch: List[MultiFrameData], frames_per_sample: int) -> BatchMultiFrameData:
        return BatchMultiFrameData.from_multiframedata_list(batch, frames_per_sample)