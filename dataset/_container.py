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

    tracks: This tensor contains objects / agents detected with the LIDAR sensors as 3D bounding boxes mapped
        to the ego vehicle frame. The tensor is of shape [1 + Nd_lidar, 9], where the first detection pertains to the ego vehicle
        and Nd_lidar is the number of unique objects detected across the LIDAR sensors, and each index of the second dimension 
        corresponds to: [track_id, center_x, center_y, center_z, length, width, height, heading_angle(rad), object_type]

    ego_pose: A [4, 4] transformation matrix to rotate relative coordinates of global frame to ego vehicle frame

    cam_intrinsic: A 3 x 3 projection matrix for each camera to map from 3D camera frame to 2D image frame.
        The tensor is of shape [V_cam, 3, 3]

    cam_extrinsic: A 4 x 4 projection matrix for each camera to map from camera frame to ego vehicle frame.
        The tensor is of shape [V_cam, 4, 4]

    agent_motions: This tensor contains the movement trajectory of dynamic agents in a given frame from timestep t to an
        arbitrary future timestep t+n. This tensor is of shape [N_agents, motion_timesteps+1, 6], 
        The (+1) is here because the current position of agents is included in the data unlike at the first timestamp
        however, it is not used to train the MotionFormer, only the subsequent points are.
        6 for [x, y, v_x, v_y, a_x, a_y]. These motion tracks do not contain position data of the current timestep, 

    occupancy_map: This tensor contains the future occupancy of dynamic agents on a BEV (Bird Eye View) grid map. 
        The tensor is of shape [N_agents, occ_timesteps, H_bev, W_bev], where H_bev and W_bev are the height and width 
        of the BEV grid. This map contains the occupancy of the current timestep as the first map

    map_elements_polylines: This tensor contains the quantized vertices of polylines on the BEV grid. This tensor is of shape
        [num_elements, num_vertices, 2]

    map_elements_boxes: This tensor contains the bounding coord data and class label indexes of map elements, This tensor is 
        of shape [num_elements, 5] (4 for box data (x, y, w, h) and 1 for class label)

    ego_motions: This tensor contains movement trajectory (waypoints) of ego vehicle in ego vehicle coordinates, The tensor 
        is of shape [max(motion_timesteps, plan_timesteps)+1, 6], 6 for [x, y, v_x, v_y, a_x, a_y].
        The extra timestep at the begining corresponds to the current position of the ego vehicle, which is not used as part of'
        the targets for the PlanFormer.
        The ego trajectory serves as targets to ego motion from the MotionFormer, and planned trajectory from the PlanFormer.

    command: This tensor encodes high-level commands for ego motion planning (like 'turn-left', 'turn-right', 'go-straight', etc), 
        it serves as input to the PlanFormer is of shape (1 x 1).
    """
    cam_views: torch.Tensor
    point_cloud: torch.Tensor
    tracks: torch.Tensor
    ego_pose: torch.Tensor
    cam_intrinsic: Optional[torch.Tensor] = None
    cam_extrinsic: Optional[torch.Tensor] = None
    agent_motions: Optional[torch.Tensor] = None
    occupancy_map: Optional[torch.Tensor] = None
    map_elements_polylines: Optional[torch.Tensor] = None
    map_elements_boxes: Optional[torch.Tensor] = None
    ego_motions: Optional[torch.Tensor] = None
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
            tracks = [],
            ego_pose = [],
            cam_intrinsic = [],
            cam_extrinsic = [],
            agent_motions = frames[-1].agent_motions,
            occupancy_map = frames[-1].occupancy_map,
            map_elements_polylines = frames[-1].map_elements_polylines,
            map_elements_boxes = frames[-1].map_elements_boxes,
            ego_motions = frames[-1].ego_motions,
            command = frames[-1].command
        )

        for frame_idx in range(0, len(frames)):
            frame = frames[frame_idx]
            sample_dict["cam_views"].append(frame.cam_views)
            sample_dict["point_cloud"].append(frame.point_cloud)
            sample_dict["ego_pose"].append(frame.ego_pose)
            sample_dict["cam_intrinsic"].append(frame.cam_intrinsic)
            sample_dict["cam_extrinsic"].append(frame.cam_extrinsic)
            sample_dict["tracks"].append(frame.tracks)
        
        sample_dict["cam_views"] = torch.stack(sample_dict["cam_views"], dim=0)
        sample_dict["point_cloud"] = torch.stack(sample_dict['point_cloud'], dim=0)
        sample_dict["ego_pose"] = torch.stack(sample_dict["ego_pose"], dim=0)
        sample_dict["cam_intrinsic"] = torch.stack(sample_dict["cam_intrinsic"], dim=0)
        sample_dict["cam_extrinsic"] = torch.stack(sample_dict["cam_extrinsic"], dim=0)
        sample_dict["tracks"] = torch.stack(sample_dict["tracks"], dim=0)
        return cls(**sample_dict)



@dataclass
class BatchMultiFrameData(FrameData):

    @classmethod
    def from_multiframedata_list(
        cls, 
        multi_frames: List[MultiFrameData], 
        frames_per_sample: int
        ) -> "BatchMultiFrameData":
        
        batch_dict = dict(
            cam_views = [],
            point_cloud = [],
            tracks = [],
            ego_pose = [],
            cam_intrinsic = [],
            cam_extrinsic = [],
            agent_motions = [],
            occupancy_map = [],
            map_elements_polylines = [],
            map_elements_boxes = [],
            ego_motions = [],
            command = []
        )
        
        items_to_bfill = [
            "cam_views", 
            "point_cloud", 
            "tracks", 
            "ego_pose", 
            "cam_intrinsic", 
            "cam_extrinsic"
        ]

        for sample_idx in range(0, len(multi_frames)):
            for key in batch_dict:
                if key in items_to_bfill:
                    data = getattr(multi_frames[sample_idx], key)
                    fill_size = frames_per_sample - data.shape[0]

                    if fill_size > 0:
                        filler = data[[0]].tile(fill_size, *[1 for _ in range(0, data.ndim-1)])
                        data = torch.concat([filler, data], dim=0)
                    batch_dict[key].append(data)

                else:
                    batch_dict[key].append(getattr(multi_frames[sample_idx], key))
        
        for key in batch_dict:
            batch_dict[key] = torch.stack(batch_dict[key], dim=0)
        
        return cls(**batch_dict)