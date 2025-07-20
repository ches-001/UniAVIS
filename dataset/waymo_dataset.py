import math
import torch
import time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.io_utils import load_pickle_file, load_json_file
from utils.img_utils import generate_occupancy_map
from utils.img_utils import polyline_2_xywh
from utils.metric_utils import intra2inter_cluster_var_ratio
from utils.img_utils import transform_points
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
            ego_vehicle_dims: Tuple[float, float, float]=(5.182, 2.032, 1.778),
            num_cloud_points: int=100_000,
            xyz_range: Optional[List[Tuple[float, float]]]=None,
            frames_per_sample: int=3,
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
        self.ego_vehicle_dims = ego_vehicle_dims
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

        # Considering that the original data has a sample frequency of 10Hz (0.1sec between 2 consecutive frames)
        # It means that there are some points in time where we will be unable to get motion data of agents and ego
        # vehicle. For example, if the number of frames per sample is 198 and frame_idx = 190, if we sample at 2Hz, 
        # the next motion point will be at frame_idx 195 and the next at frame_idx 200, which is an invalid index 
        # for a size of 198. Even if we decide to only use positional motion data at frame_idx 190 and 195, we will
        # only have a single velocity value, but and no acceleration value. We need atleast 2 velocity values, 
        # this is necessary for the multi-shooting solver used in the motion_lossfn, hence we cap frame_idx like so:

        # frame_idx = min(frame_idx, num_frames - (3 * offset) - 1)
        # where: offset = max(1, round(freq / new_freq))
        # where: freq = 10 Hz and new_freq = 2Hz

        num_frames = len(sample_dict)
        t1 = sample_dict[1]["timestamp_seconds"]
        t0 = sample_dict[0]["timestamp_seconds"]
        offset = max(1, round(1 / ((t1 - t0) * self.motion_sample_freq)))
        last_frame_idx = num_frames - (3 * offset) - 1
        frame_idx = min(frame_idx, last_frame_idx)

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
        tracks = self._load_track_labels(frame_dict, obj_id_map=obj_id_map)
        ego_pose, cam_intrinsic, cam_extrinsic = self._load_ego_pose_and_transforms(frame_dict)            

        agent_motions = None
        occupancy_map = None
        map_elements_polylines = None
        map_elements_boxes = None
        ego_motions = None
        agent_motions = None
        command = None

        if not is_partial_data:
            agent_motions = self._get_agent_motions(sample_dict, frame_idx)
            occupancy_map = self._generate_occupancy_map(agent_motions[..., [0, 1, 6, 7, 8]])
            ego_motions = self._get_ego_motions(sample_dict, frame_idx, ego_pose)
            command = self._get_high_level_command(ego_motions[1:, :2])
            map_elements_polylines, map_elements_boxes = self._load_bev_map_elements(frame_dict)

            # pad targets
            map_element_pad = max(0, self.max_num_map_elements - map_elements_polylines.shape[0])
            agents_pad = max(0, self.max_num_agents - agent_motions.shape[0])
            ego_plan_pad = max(0, self.planning_horizon - ego_motions.shape[0])
            
            map_elements_polylines = F.pad(
                map_elements_polylines, pad=(0, 0, 0, 0, 0, map_element_pad), mode="constant", value=-999
            )
            map_elements_boxes = F.pad(
                map_elements_boxes, pad=(0, 0, 0, map_element_pad), mode="constant", value=-999
            )
            agent_motions = F.pad(
                agent_motions, pad=(0, 0, 0, 0, 0, agents_pad), mode="constant", value=-999
            )
            occupancy_map = F.pad(
                occupancy_map, pad=(0, 0, 0, 0, 0, 0, 0, agents_pad), mode="constant", value=-999
            )

            ego_motions = F.pad(
                ego_motions, pad=(0, 0, 0, ego_plan_pad), mode="constant", value=-999
            )
        
            map_cls_pad_val = len(self.metadata["LABELS"]["MAP_ELEMENT_LABEL_INDEXES"])
            map_elements_boxes[:, -1][map_elements_boxes[:, -1] == -999] = map_cls_pad_val

        # we add +1 below because rthe tracks data also includes that of the ego vehicle, as the
        # first track
        det_pad = (self.max_num_agents + 1) - tracks.shape[0]
        det_cls_pad_val = len(self.metadata["LABELS"]["DETECTION_LABEL_INDEXES"])
        tracks = F.pad(tracks, pad=(0, 0, 0, det_pad), mode="constant", value=-999)
        tracks[:, -1][tracks[:, -1] == -999] = det_cls_pad_val

        frame_data = FrameData(
            cam_views=cam_views,
            point_cloud=point_cloud,
            ego_pose=ego_pose,
            tracks=tracks,
            cam_intrinsic=cam_intrinsic,
            cam_extrinsic=cam_extrinsic,
            agent_motions=agent_motions,
            occupancy_map=occupancy_map,
            map_elements_polylines=map_elements_polylines,
            map_elements_boxes=map_elements_boxes,
            ego_motions=ego_motions,
            command=command
        )
        return frame_data


    @check_perf
    def _get_agent_motions(self, sample_dict: Dict[str, Any], frame_idx: int) -> torch.Tensor:
        max_horizon = max(self.motion_horizon, self.occupancy_horizon)
        motions = WaymoDataset.get_frame_agent_motions(
            sample_dict=sample_dict, 
            frame_idx=frame_idx, 
            max_horizon=max_horizon, 
            sample_freq=self.motion_sample_freq, 
            xyz_range=self.xyz_range,
        )
        # [x, y, l, w, angle]
        motions = motions[..., [0, 1, 3, 4, 6]]

        # impute past velocity with average of the first two velocities
        vel = self.motion_sample_freq * (motions[:, 1:, :2] - motions[:, :-1, :2])
        vel = torch.concat([(vel[:, [0], :] + vel[:, [1], :]) / 2, vel], dim=1)
        
        # impute past accelerations with average of the last two accelerations
        accel = self.motion_sample_freq * (vel[:, 1:, :2] - vel[:, :-1, :2])
        accel = torch.concat([(accel[:, [0], :] + accel[:, [1], :]) / 2, accel], dim=1)

        # [x, y, vx, vy, ax, ay, l, w, angle]
        motions = torch.concat([motions[..., :2], vel, accel, motions[..., 2:]], dim=-1)
        
        pad_size = (max_horizon + 1) - motions.shape[1]
        if pad_size != 0:
            motions = F.pad(motions, (0, 0, 0, pad_size, 0, 0), mode="constant", value=-999)
        # shape: (num_detections, motion_timesteps+1, 8)
        return motions

    
    @check_perf
    def _generate_occupancy_map(self, agent_motions: torch.Tensor) -> torch.Tensor:
        occ_map = generate_occupancy_map(
            agent_motions[:, :self.occupancy_horizon, :], 
            map_hw=self.bev_map_hw,
            x_min=self.xyz_range[0][0],
            x_max=self.xyz_range[0][1],
            y_min=self.xyz_range[1][0],
            y_max=self.xyz_range[1][1],
        )
        # shape: (num_detections, timesteps, H_bev, W_bev)
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
    def _load_track_labels(self, frame_dict: Dict[str, Any], obj_id_map: Dict[str, int]) -> torch.Tensor:
        laser_labels_path = frame_dict["laser_labels_path"]
        laser_labels_list = load_pickle_file(laser_labels_path)
        detections = []

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

            detections.append([
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
        # shape: (1 + num_detections, 9)
        detections = torch.tensor(detections, dtype=torch.float32)
        vehicle_label = self.metadata["LABELS"]["DETECTION_LABEL_INDEXES"]["VEHICLE"]

        # ego vehicle z-center is not at 0, since 0 corresponds to the ground level, instead 
        # it is at a location equal to half of its height.
        ego_detections = torch.tensor(
            [-1, *([0.0]*2), self.ego_vehicle_dims[2] / 2, *self.ego_vehicle_dims, 0.0, vehicle_label], dtype=torch.float32
        )
        detections = torch.concat([ego_detections[None, :], detections], dim=0)
        return detections

    
    @check_perf
    def _load_ego_pose_and_transforms(
        self, 
        frame_dict: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        ego_pose = torch.from_numpy(frame_dict["ego_pose"])
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
        ego_pose = ego_pose.to(torch.float32)
        cam_intrinsic = cam_intrinsic.to(torch.float32)
        cam_extrinsic = cam_extrinsic.to(torch.float32)
        
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
        
        cls = torch.from_numpy(cls)
        boxes = torch.concat([polyline_boxes, cls], dim=-1)
        
        # polylines: (num_elements, num_points, 2)
        # boxes: (num_elements, 4 + 1)
        return polylines.long(), boxes.long()
    

    @check_perf
    def _get_ego_motions(
        self, 
        sample_dict: Dict[str, Any],
        frame_idx: int,
        ego_pose: torch.Tensor
    ) -> torch.Tensor:
        
        num_frames = len(sample_dict)
        horizon = max(self.planning_horizon, self.motion_horizon)
        
        t0 = sample_dict[frame_idx]["timestamp_seconds"]
        t1 = sample_dict[frame_idx + 1]["timestamp_seconds"]
        dt = t1 - t0
        iter_step = max(1, round(1 / (dt * self.motion_sample_freq)))
        iter_start = frame_idx
        iter_end = min(num_frames, frame_idx + (iter_step * horizon) + 1)

        global_positions = [sample_dict[idx]["ego_pose"][:, -1] for idx in range(iter_start, iter_end, iter_step)]
        global_positions = np.stack(global_positions, axis=0)
        global_positions = torch.from_numpy(global_positions).to(ego_pose.dtype)

        # shape: (timesteps, 2)
        ego_motions = torch.matmul(global_positions, torch.linalg.inv(ego_pose).permute(1, 0))[:, :2]

        # impute past velocity with average of the last two velocities
        vel = self.motion_sample_freq * (ego_motions[1:] - ego_motions[:-1])
        vel = torch.concat([(vel[[0]] + vel[[1]]) / 2, vel], dim=0)

        # impute past accelerations with average of the last two accelerations
        accel = self.motion_sample_freq * (vel[1:] - vel[:-1])
        accel = torch.concat([(accel[[0]] + accel[[1]]) / 2, accel], dim=0)

        ego_motions = torch.concat([ego_motions, vel, accel], dim=-1)
        return ego_motions
        
    

    @check_perf
    def _get_high_level_command(self, ego_motions: torch.Tensor) -> torch.Tensor:
        # to get high level command of ego planning motion, compute the cross and dot products of the last
        # direction vector, relative to a reference vector. The reference vector in this case is [1.0, 0.0]
        # which corresponds to straight motion along the x axis, and the last direction vector is [xt - x0, yt - y0]
        # which is essentially equal to [xt, yt], because the ego vehicle is always situated at the reference point 
        # [0, 0]. Next we compute the arctan2 between the cross and dot product, this is equivalent to computing the
        # arctan2 between the sin and cos of the angles between the two vectors.
        x0, y0 = 1.0, 0.0
        xt, yt = ego_motions[min(self.planning_horizon, ego_motions.shape[0]) - 1]

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
    def get_frame_agent_motions(
        sample_dict: Dict[str, Any], 
        frame_idx: int, 
        max_horizon: int, 
        sample_freq: float, 
        xyz_range: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ) -> torch.Tensor:
        num_frames = len(sample_dict)
        
        t0 = sample_dict[frame_idx]["timestamp_seconds"]
        t1 = sample_dict[frame_idx + 1]["timestamp_seconds"]
        dt = t1 - t0
        iter_step = max(1, round(1 / (dt * sample_freq)))
        iter_start = frame_idx
        iter_end = min(num_frames, frame_idx + (iter_step * max_horizon) + 1)
        
        motions_maps = {}
        for idx in range(iter_start, iter_end, iter_step):
            laser_labels_path = sample_dict[idx]["laser_labels_path"]
            laser_labels_list = load_pickle_file(laser_labels_path)

            for obj_idx in range(0, len(laser_labels_list)):
                obj_id = laser_labels_list[obj_idx]["id"]
                obj_3d_bbox = laser_labels_list[obj_idx]["box"]
                if (
                    (obj_3d_bbox["center_x"] < xyz_range[0][0] or obj_3d_bbox["center_x"] > xyz_range[0][1])
                    or (obj_3d_bbox["center_y"] < xyz_range[1][0] or obj_3d_bbox["center_y"] > xyz_range[1][1])
                    or (obj_3d_bbox["center_z"] < xyz_range[2][0] or obj_3d_bbox["center_z"] > xyz_range[2][1])
                ): continue
                
                if obj_id not in motions_maps:
                    if idx == frame_idx:
                        motions_maps[obj_id] = []
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
                motions_maps[obj_id].append(obj_det)

        # NOTE: Nested tensors is experimental feature and may change behaviour in the future
        motions = torch.nested.nested_tensor(list(motions_maps.values()))
        motions = motions.to_padded_tensor(-999)
        return motions
    

    @staticmethod
    def collate_fn(batch: List[MultiFrameData], frames_per_sample: int) -> BatchMultiFrameData:
        return BatchMultiFrameData.from_multiframedata_list(batch, frames_per_sample)
    

    @staticmethod
    def cluster_agent_motion_endpoints(
        data_or_path: Union[str, Dict[str, Any]],
        *,
        num_clusters: int=6,
        num_iters: int=100,
        num_mut_gens: int=100,
        mut_proba: float=0.9,
        mut_sigma: float=0.1,
        max_horizon: int=12,
        sample_freq: float=2,
        device: Union[int, str, torch.device]="cpu",
        **kwargs
    ) -> Tuple[torch.Tensor, float, float, float, int]:
        
        import tqdm
        from modules.motionformer import MotionFormer

        data = (
            data_or_path
            if isinstance(data_or_path, dict)
            else load_pickle_file(data_or_path)
        )
        sample_freq = sample_freq / max_horizon
        max_horizon = 1
        dt = None
        frame_steps = None

        pbar = tqdm.tqdm(total=len(data), unit_scale=True)

        endpoints_data = []
        for sample_name in data:
            sample_dict = data[sample_name]
            num_frames = len(sample_dict)
            frame_idx = 0
            if dt is None:
                dt = sample_dict[frame_idx + 1]["timestamp_seconds"] - sample_dict[frame_idx]["timestamp_seconds"]
                frame_steps = max(1, round(1 / (sample_freq * dt)))

            while frame_idx < num_frames:
                pbar.update(frame_steps / (math.ceil(num_frames / frame_steps) * frame_steps))

                if frame_idx + 1 == num_frames:
                    continue
                motion_data = WaymoDataset.get_frame_agent_motions(
                    sample_dict, frame_idx, max_horizon=max_horizon, sample_freq=sample_freq, **kwargs
                )
                
                first_points = motion_data[:, 0, [0, 1, 2, 6]]
                last_points = motion_data[:, -1, [0, 1, 2, 6]]
                
                valid_points_mask = last_points[:, 0] != -999
                if valid_points_mask.sum() == 0:
                    continue

                last_points = last_points[valid_points_mask]
                first_points = first_points[valid_points_mask]
                agent2scene_transform = MotionFormer.create_agent2scene_transforms(first_points[:, 3], first_points[:, :3])
                scene2agent_transform = torch.linalg.inv(agent2scene_transform)
                last_points = transform_points(last_points[:, None, :2], transform_matrix=scene2agent_transform)
                endpoints_data.append(last_points[:, 0, :])
                
                frame_idx += frame_steps

        pbar.close()
        
        endpoints_data = torch.concat(endpoints_data, dim=0).to(device)
        
        # initialize ccentroids with kmeans++ initialization technique
        centroids = []
        centroids.append(endpoints_data[torch.randint(0, endpoints_data.shape[0], size=(), device=device)])
        for cidx in range(1, num_clusters):
            stacked_centroids = torch.stack(centroids, dim=0)
            dists_squared = (endpoints_data[:, None, :] - stacked_centroids[None, :, :]).pow(2).sum(dim=-1)
            min_dists_squared = torch.min(dists_squared, dim=1).values
            proba = min_dists_squared / min_dists_squared.sum()
            centroids.append(endpoints_data[torch.multinomial(proba, num_samples=1)][0])

        centroids = torch.stack(centroids, dim=0)
        prev_centroids = None
        converged = False

        for _ in range(0, num_iters):
            prev_centroids = centroids.clone()
            dists = (endpoints_data[:, None, :] - centroids[None, :, :]).pow(2).sum(dim=-1)
            cluster_ids = torch.argmin(dists, dim=1)

            for cidx in range(0, num_clusters):
                mask = (cluster_ids == cidx)
                if mask.any():
                    centroids[cidx] = endpoints_data[mask].mean(dim=0)

            if (prev_centroids == centroids).all():
                converged = True
                break
        
        dists = (endpoints_data[:, None, :] - centroids[None, :, :]).pow(2).sum(dim=-1)
        cluster_ids = torch.argmin(dists, dim=1)
        
        best_intra_var, best_inter_var, best_score = intra2inter_cluster_var_ratio(
            endpoints_data, centroids, cluster_ids
        )
        best_gen = None

        if converged:
            return centroids, best_score, best_gen
        
        for gen in range(0, num_mut_gens):
            mut_factor = torch.ones_like(centroids)

            while (mut_factor == 1).all():
                scale = torch.randn(1, device=device)
                rand = torch.rand_like(centroids)
                randn = torch.randn_like(centroids)
                mut_factor = ((rand > mut_proba) * scale * randn * mut_sigma) + 1

            new_centroids = centroids + mut_factor
            new_intra_var, new_inter_var, new_score = intra2inter_cluster_var_ratio(
                endpoints_data, new_centroids, cluster_ids
            )
            if new_score < best_score:
                centroids = new_centroids
                best_score = new_score
                best_intra_var = new_intra_var
                best_inter_var = new_inter_var
                best_gen = gen

        return centroids, best_score.item(), best_intra_var.item(), best_inter_var.item(), best_gen