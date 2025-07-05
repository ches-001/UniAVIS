import torch
from dataclasses import dataclass
from typing import Optional, Union, List

class DeviceChangeMixin:
    def to(self, *, attr: Optional[str]=None, **kwargs):
        if attr:
            getattr(self, attr).to(**kwargs)
            return self
        for attr in self.__dict__:
            getattr(self, attr).to(**kwargs)
        return self

@dataclass
class FrameData(DeviceChangeMixin):
    """
    This class stores data of a single frame at a given timestep

    Attributes:
    -----------------------------------------
    cam_views: multi-view images from different cameras in a single instance in time with shape 
        [V_cam, C_img, H_img, W_img], where V_cam, C_img, H_img and W_img are the number of cameras 
        on the ego vehicle, number of color channels, image height and image width respectively

    point_cloud: Unstructured array of 4D points with shape [Np, D], where Np and D are the number of points 
        and dimension of each point, here D = 4, 3 spatial axes x, y, z in the  ego vehicle frame and an axis 
        for reflectance / intensity r.

    laser_detections: This tensor contains objects / agents detected with the LIDAR sensors as 3D bounding boxes mapped
        to the ego vehicle frame. The tensor is of shape [1 + Nd_lidar, 9], where the first detection pertains to the ego vehicle
        and Nd_lidar is the number of unique objects detected across the LIDAR sensors, and each index of the second dimension 
        corresponds to: [track_id, center_x, center_y, center_z, length, width, height, heading_angle(rad), object_type]

    ego_pose: A [4, 4] transformation matrix to rotate relative coordinates of global frame to ego vehicle frame

    cam_intrinsic: A 3 x 3 projection matrix for each camera to map from 3D camera frame to 2D image frame.
        The tensor is of shape [V_cam, 3, 3]

    cam_extrinsic: A 4 x 4 projection matrix for each camera to map from camera frame to ego vehicle frame.
        The tensor is of shape [V_cam, 4, 4]

    motion_tracks: This tensor contains the movement trajectory of dynamic agents in a given frame from timestep t to an
        arbitrary future timestep t+n. This tensor is of shape [N_agents, motion_timesteps, 2], as you might have guessed, 
        these tracks are gotten from the x and y values of the laser_detections from various timesteps. These motion
        tracks do not contain position data of the current timestep, unlike the occupancy map that contains the occupancy
        of the current timestep

    occupancy_map: This tensor contains the future occupancy of dynamic agents on a BEV (Bird Eye View) grid map. 
        The tensor is of shape [N_agents, occ_timesteps, H_bev, W_bev], where H_bev and W_bev are the height and width 
        of the BEV grid. This map contains the occupancy of the current timestep as the first map

    map_elements_polylines: This tensor contains the quantized vertices of polylines on the BEV grid. This tensor is of shape
        [num_elements, num_vertices, 2]

    map_elements_boxes: This tensor contains the bounding coord data and class label indexes of map elements, This tensor is 
        of shape [num_elements, 5] (4 for box data (x, y, w, h) and 1 for class label)

    ego_trajectory: This tensor contains movement trajectory (waypoints) of ego vehicle in ego vehicle coordinates, The tensor 
        is of shape [max(motion_timesteps, plan_timesteps), 2]. It only has two dimensions because it the trajectory corresponds
        to movement along the BEV frame, which is a 2D frame.
        The ego trajectory serves as targets to ego motion from the MotionFormer, and planned trajectory from the PlanFormer.

    command: This tensor encodes high-level commands for ego motion planning (like 'turn-left', 'turn-right', 'go-straight', etc), 
        it serves as input to the PlanFormer is of shape (1 x 1).
    """
    cam_views: torch.Tensor
    point_cloud: torch.Tensor
    laser_detections: torch.Tensor
    ego_pose: torch.Tensor
    cam_intrinsic: Optional[torch.Tensor] = None
    cam_extrinsic: Optional[torch.Tensor] = None
    motion_tracks: Optional[torch.Tensor] = None
    occupancy_map: Optional[torch.Tensor] = None
    map_elements_polylines: Optional[torch.Tensor] = None
    map_elements_boxes: Optional[torch.Tensor] = None
    ego_trajectory: Optional[torch.Tensor] = None
    command: Optional[torch.Tensor] = None


@dataclass
class MultiFrameData(FrameData):
    """
    This class stores multiple frame data from multiple timesteps
    """

    @classmethod
    def from_framedata_list(cls, frames: List[FrameData]) -> "MultiFrameData":
        sample_dict = dict(
            cam_views = [],
            point_cloud = [],
            laser_detections = [],
            ego_pose = [],
            cam_intrinsic = [],
            cam_extrinsic = [],
            motion_tracks = frames[-1].motion_tracks,
            occupancy_map = frames[-1].occupancy_map,
            map_elements_polylines = frames[-1].map_elements_polylines,
            map_elements_boxes = frames[-1].map_elements_boxes,
            ego_trajectory = frames[-1].ego_trajectory,
            command = frames[-1].command
        )

        for frame_idx in range(0, len(frames)):
            frame = frames[frame_idx]
            sample_dict["cam_views"].append(frame.cam_views)
            sample_dict["point_cloud"].append(frame.point_cloud)
            sample_dict["ego_pose"].append(frame.ego_pose)
            sample_dict["cam_intrinsic"].append(frame.cam_intrinsic)
            sample_dict["cam_extrinsic"].append(frame.cam_extrinsic)
            sample_dict["laser_detections"].append(frame.laser_detections)
        
        sample_dict["cam_views"] = torch.stack(sample_dict["cam_views"], dim=0)
        sample_dict["point_cloud"] = torch.stack(sample_dict['point_cloud'], dim=0)
        sample_dict["ego_pose"] = torch.stack(sample_dict["ego_pose"], dim=0)
        sample_dict["cam_intrinsic"] = torch.stack(sample_dict["cam_intrinsic"], dim=0)
        sample_dict["cam_extrinsic"] = torch.stack(sample_dict["cam_extrinsic"], dim=0)
        sample_dict["laser_detections"] = torch.stack(sample_dict["laser_detections"], dim=0)
        return cls(**sample_dict)



@dataclass
class BatchMultiFrameData(FrameData):
    """
    This class stores a batch of multiple frame data

    :timestep_mask: This tensor stores a padding mask for timesteps, it is of shape (batch_size, timesteps)
        if the frame at the corresponding timestep is a padding tensor, then timestep_mask[batch_idx, frame_idx]
        will be set to 0, else 1.
    """
    timestep_pad_mask: Optional[torch.Tensor]=None

    @classmethod
    def from_multiframedata_list(
        cls, 
        multi_frames: List[MultiFrameData], 
        frames_per_sample: int
        ) -> "BatchMultiFrameData":
        
        batch_dict = dict(
            timestep_pad_mask = [],
            cam_views = [],
            point_cloud = [],
            laser_detections = [],
            ego_pose = [],
            cam_intrinsic = [],
            cam_extrinsic = [],
            motion_tracks = [],
            occupancy_map = [],
            map_elements_polylines = [],
            map_elements_boxes = [],
            ego_trajectory = [],
            command = []
        )
        
        items_to_pad = ["cam_views", "point_cloud", "ego_pose"]

        for sample_idx in range(0, len(multi_frames)):

            for key in batch_dict:
                if key == "timestep_pad_mask":
                    continue

                if key in items_to_pad:
                    data = getattr(multi_frames[sample_idx], key)
                    pad_size = frames_per_sample - data.shape[0]
                    
                    tmask = torch.zeros(frames_per_sample, dtype=torch.bool)
                    tmask[pad_size:] = True

                    if pad_size > 0:
                        pad_data = torch.zeros_like(data[0])[None].tile(pad_size, *[1 for _ in range(0, data.ndim-1)])
                        data = torch.concat([pad_data, data], dim=0)
                    batch_dict[key].append(data)
                    batch_dict["timestep_pad_mask"].append(tmask)

                else:
                    batch_dict[key].append(getattr(multi_frames[sample_idx], key))
        
        for key in batch_dict:
            batch_dict[key] = torch.stack(batch_dict[key], dim=0)
        
        return cls(**batch_dict)