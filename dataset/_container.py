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
        for reflectance r.

    laser_detections: This tensor contains objects / agents detected with the LIDAR sensors as 3D bounding boxes mapped
        to the ego vehicle frame. The tensor is of shape [Nd_lidar, 10], where Nd_lidar is the number of unique objects 
        detected across the LIDAR sensors, and each index of the second dimension corresponds to:
        [batch_index, frame_index, center_x, center_y, center_z, length, width, height, heading_angle(rad), object_type]

    ego_pose: A [4, 4] transformation matrix to rotate relative coordinates of global frame to ego vehicle frame

    cam_intrinsic: A 3 x 3 projection matrix for each camera to map from 3D camera frame to 2D image frame.
        The tensor is of shape [V_cam, 3, 3]

    cam_extrinsics: A 4 x 4 projection matrix for each camera to map from camera frame to ego vehicle frame.
        The tensor is of shape [V_cam, 4, 4]

    motion_tracks: This tensor contains the movement trajectory of dynamic agents in a given frame from timestep t to an
        arbitrary future timestep t+n. This tensor is of shape [N_agents, motion_timesteps, 2], as you might have guessed, 
        these tracks are gotten from the x and y values of the laser_detections from various timesteps. These motion
        tracks do not contain position data of the current timestep, unlike the occupancy map that contains the occupancy
        of the current timestep

    occupancy_map: This tensor contains the future occupancy of dynamic agents on a BEV (Bird Eye View) grid map. 
        The tensor is of shape [N_agents, occ_timesteps, H_bev, W_bev], where H_bev and W_bev are the height and width 
        of the BEV grid. This map contains the occupancy of the current timestep as the first map

    map_elements_mask: This tensor contains mask segmentation of road components on the BEV grid. This tensor is of shape 
        [1, H_bev, W_bev].

    map_elements_polylines: This tensor contains the quantized vertices of polylines on the BEV grid. This tensor is of shape
        [num_elements, num_vertices, 2]

    map_elements_labels: This tensor contains the class label indexes of map elements, This tensor is of shape [num_elements, ]

    ego_trajectory: This tensor contains movement trajectory (waypoints) of ego vehicle in ego vehicle coordinates, The tensor 
        is of shape [plan_timesteps, 2]. It only has two dimensions because it the trajectory corresponds to movement along the
        BEV frame, which is a 2D frame
    """
    cam_views: torch.Tensor
    point_cloud: torch.Tensor
    laser_detections: torch.Tensor
    ego_pose: Optional[torch.Tensor] = None
    cam_intrinsic: Optional[torch.Tensor] = None
    cam_extrinsics: Optional[torch.Tensor] = None
    motion_tracks: Optional[torch.Tensor] = None
    occupancy_map: Optional[torch.Tensor] = None
    map_elements_mask: Optional[torch.Tensor] = None
    map_elements_polylines: Optional[torch.Tensor] = None
    map_elements_labels: Optional[torch.Tensor] = None
    ego_trajectory: Optional[torch.Tensor] = None


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
            ego_pose = frames[-1].ego_pose,
            cam_intrinsic = frames[-1].cam_intrinsic,
            cam_extrinsics = frames[-1].cam_extrinsics,
            motion_tracks = frames[-1].motion_tracks,
            occupancy_map = frames[-1].occupancy_map,
            map_elements_mask = frames[-1].map_elements_mask,
            map_elements_polylines = frames[-1].map_elements_polylines,
            map_elements_labels = frames[-1].map_elements_labels,
            ego_trajectory = frames[-1].ego_trajectory
        )

        for frame_idx in range(0, len(frames)):
            frame = frames[frame_idx]
            sample_dict["cam_views"].append(frame.cam_views)
            sample_dict["point_cloud"].append(frame.point_cloud)
            laser_detections = frame.laser_detections
            laser_detections[:, 1] = frame_idx
            sample_dict["laser_detections"].append(laser_detections)
        
        sample_dict["cam_views"] = torch.stack(sample_dict["cam_views"], dim=0)
        sample_dict["point_cloud"] = torch.stack(sample_dict['point_cloud'], dim=0)
        sample_dict["laser_detections"] = torch.concat(sample_dict["laser_detections"], dim=0)
        return cls(**sample_dict)



@dataclass
class BatchMultiFrameData(FrameData):
    """
    This class stores a batch of multiple frame data
    """

    @classmethod
    def from_multiframedata_list(cls, multi_frames: List[MultiFrameData]) -> "BatchMultiFrameData":
        batch_dict = dict(
            cam_views = [],
            point_cloud = [],
            laser_detections = [],
            ego_pose = [],
            cam_intrinsic = [],
            cam_extrinsics = [],
            motion_tracks = [],
            occupancy_map = [],
            map_elements_mask = [],
            map_elements_polylines = [],
            map_elements_labels = [],
            ego_trajectory = []
        )

        # TODO: Work on integrating map_elements_polylines and map_elements_labels data to batch

        for sample_idx in range(0, len(multi_frames)):
            for key in batch_dict:
                batch_dict[key].append(getattr(multi_frames[sample_idx], key))
        
        for key in batch_dict:
            if key == "laser_detections":
                batch_dict[key] = torch.concat(batch_dict[key], dim=0)

            elif key == "motion_tracks" or key == "occupancy_map" or key == "ego_trajectory":
                batch_dict[key] = torch.nested.nested_tensor(batch_dict[key], layout=torch.jagged)
                
            else:
                batch_dict[key] = torch.stack(batch_dict[key], dim=0)
        
        return cls(**batch_dict)