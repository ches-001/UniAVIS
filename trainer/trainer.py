import os
import time
import tqdm
import math
import logging
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from utils.ddp_utils import ddp_sync_metrics
from utils.io_utils import load_yaml_file, save_yaml_file
from utils.img_utils import xywh_to_xyxy
from .base import *
from dataset import BatchMultiFrameData
from typing import Union, Optional, Dict, Any


LOGGER = logging.getLogger(__name__)

class E2ETrainer(BaseTrainer):
    def __init__(
        self, 
        bevformer: BEVFormer,
        optimizer: torch.optim.Optimizer,
        *,
        trackformer: Optional[TrackFormer]=None,
        track_lossfn: Optional[TrackLoss]=None,
        mapformer: Optional[Union[VectorMapFormer, RasterMapFormer]]=None,
        map_lossfn: Optional[Union[VectorMapLoss, RasterMapLoss]]=None,
        motionformer: Optional[MotionFormer]=None,
        motion_lossfn: Optional[MotionLoss]=None,
        occformer: Optional[OccFormer]=None,
        occ_lossfn: Optional[OccLoss]=None,
        planformer: Optional[PlanFormer]=None,
        plan_lossfn: Optional[PlanLoss]=None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]=None,
        lr_scheduler_step: int=10,
        checkpoints_path: Optional[str]=None,
        device_or_rank: Union[int, str]="cpu",
        ddp_mode: bool=False,
        config_or_path: Optional[Union[str, Dict[str, Any]]]=None
    ):
        super(E2ETrainer, self).__init__(ddp_mode, device_or_rank, checkpoints_path, config_or_path)

        self._set_module(bevformer, "bevformer", is_model=True)
        self._set_module(trackformer, "trackformer", is_model=True)
        self._set_module(mapformer, "mapformer", is_model=True)
        self._set_module(motionformer, "motionformer", is_model=True)
        self._set_module(occformer, "occformer", is_model=True)
        self._set_module(planformer, "planformer", is_model=True)

        self._set_module(track_lossfn, "track_lossfn", is_model=False)
        self._set_module(map_lossfn, "map_lossfn", is_model=False)
        self._set_module(motion_lossfn, "motion_lossfn", is_model=False)
        self._set_module(occ_lossfn, "occ_lossfn", is_model=False)
        self._set_module(plan_lossfn, "plan_lossfn", is_model=False)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_step = lr_scheduler_step
        self.last_epoch = 0

        # load checkpoints if any
        if self.checkpoints_path:
            self.load_checkpoints(self.checkpoints_path)
    

    def train(self, dataloader: DataLoader, verbose: bool=False) -> Dict[str, float]:
        metrics = self.step(dataloader, "train", verbose)
        self._train_metrics.append(metrics)

        if self.lr_scheduler and (self.last_epoch % self.lr_scheduler_step == 0):
            self.lr_scheduler.step()
            if verbose:
                lr = self.lr_scheduler.optimizer.param_groups[0]["lr"]
                print(f"Current LR: {lr :.6f}")
        self.last_epoch += 1

        return metrics
        

    def evaluate(self, dataloader: DataLoader, verbose: bool=False) -> Dict[str, float]:        
        with torch.no_grad():
            metrics = self.step(dataloader, "eval", verbose)
        self._eval_metrics.append(metrics)
        return metrics


    def step(self, dataloader: DataLoader, mode: str, verbose: bool=False) -> Dict[str, float]:
        if mode not in self._valid_modes:
            raise ValueError(f"Invalid mode {mode} expected either one of {self._valid_modes}")
        
        self._toggle_module_mode("bevformer", mode)
        self._toggle_module_mode("trackformer", mode)
        self._toggle_module_mode("mapformer", mode)
        self._toggle_module_mode("motionformer", mode)
        self._toggle_module_mode("occformer", mode)
        self._toggle_module_mode("planformer", mode)

        metrics = {}

        if self.ddp_mode:
            # invert progress bar position such that the last (rank n-1) is at
            # the top and the first (rank 0) at the bottom. This is because the
            # first rank will be the one logging all the metrics
            world_size = int(os.environ["WORLD_SIZE"])
            position = self.device_or_rank
            position = abs(position - (world_size - 1))
            pbar = tqdm.tqdm(enumerate(dataloader), position=position)
        else:
            total = math.ceil(len(dataloader.dataset) / dataloader.batch_size)
            pbar = tqdm.tqdm(enumerate(dataloader), total=total)

        # purpose of this line is for vscode highlight
        sample_batch: BatchMultiFrameData

        avg_track_loss, avg_map_loss, avg_motion_loss, avg_occ_loss, avg_plan_loss = [0.0] * 5

        for step, sample_batch in pbar:
            t_frames = sample_batch.cam_views.shape[1]
            cam_projections = sample_batch.cam_intrinsic @ torch.linalg.inv(sample_batch.cam_extrinsic)[..., :3, :]

            bev_features, track_queries, track_mask, track_ids = [None] * 4
            track_loss, map_loss, motion_loss, occ_loss, plan_loss = [0.0] * 5
            
            self.optimizer.zero_grad()

            for tidx in range(0, t_frames):
                t_imgs = sample_batch.cam_views[:, tidx]
                t_lidar = sample_batch.point_cloud[:, tidx]
                t_cam_projections = cam_projections[:, tidx]
                if bev_features is not None:
                    transition = torch.linalg.inv(sample_batch.ego_pose[:, tidx]) @ sample_batch.ego_pose[:, tidx - 1]


                bev_features = self.bevformer.forward(
                    imgs=t_imgs,
                    lidar_points=t_lidar,
                    projection=t_cam_projections,
                    transition=transition,
                    bev_histories=bev_features
                )

                ego_track_labels = sample_batch.tracks[:, tidx, 0]
                track_labels = sample_batch.tracks[:, tidx, 1:]

                if hasattr(self, "trackformer"):
                    track_queries, track_preds, track_ego_data = self.trackformer.forward(
                        bev_features=bev_features,
                        track_queries=track_queries,
                        track_queries_mask=track_mask
                    )
                    ego_track_query = track_ego_data["ego_query"]
                    ego_track_preds = track_ego_data["ego_detection"]

                    track_preds = torch.concat([ego_track_preds[:, None, :], track_preds], dim=1)
                    track_preds = TrackFormer.preds2scale(
                        preds=track_preds,
                        xyz_range=self.bevformer.point_net.pillar_gen.xyz_range,
                        angle_range=(-torch.pi, torch.pi)
                    )
                    ego_track_preds = track_preds[:, 0, :]
                    track_preds = track_preds[:, 1:, :]

                    if hasattr(self, "track_lossfn"):
                        t_track_loss, track_pred_indexes, track_target_indexes, track_ids, track_mask = self.track_lossfn.forward(
                            preds=track_preds,
                            targets=track_labels,
                            ego_preds=ego_track_preds,
                            ego_targets=ego_track_labels,
                            prev_track_ids=track_ids
                        )
                        track_loss = track_loss + (t_track_loss / t_frames)
                        avg_track_loss += track_loss

            map_box_labels = sample_batch.map_elements_boxes
            map_tgt_polylines = sample_batch.map_elements_polylines

            if hasattr(self, "mapformer"):
                bev_wh = (self.mapformer.bev_feature_hw[1], self.mapformer.bev_feature_hw[0])
                
                if isinstance(self.mapformer, RasterMapFormer):
                    assert isinstance(self.map_lossfn, RasterMapLoss)
                    map_queries, protos, pred_map_boxes = self.mapformer.forward(bev_features=bev_features)
                    pred_map_boxes = RasterMapFormer.boxes2scale(pred_map_boxes, bev_wh)
                    
                    if hasattr(self, "map_lossfn"):
                        for lidx in range(0, pred_map_boxes.shape[0]):
                            l_map_loss, map_pred_indexes, _, map_mask = self.map_lossfn.forward(
                                protos=protos, 
                                pred_detections=pred_map_boxes[lidx],
                                target_polylines=map_tgt_polylines,
                                target_detections=map_box_labels.float()
                            )
                            map_loss = map_loss + (l_map_loss / pred_map_boxes.shape[0])
                            avg_map_loss += map_loss

                elif isinstance(self.mapformer, VectorMapFormer):
                    assert isinstance(self.map_lossfn, VectorMapLoss)
                    map_tgt_kps = xywh_to_xyxy(map_box_labels[..., :4]).unflatten(dim=-1, sizes=(2, 2))
                    map_tgt_cls = map_box_labels[..., 5]

                    map_queries, pred_polyline_logits, pred_map_boxes = self.mapformer.forward(
                        bev_features=bev_features, 
                        tgt_kps=map_tgt_kps,
                        tgt_classes=map_tgt_cls,
                        tgt_vertices=map_tgt_polylines
                    )
                    pred_map_boxes = RasterMapFormer.boxes2scale(pred_map_boxes, bev_wh)

                    if hasattr(self, "map_lossfn"):
                        map_loss, map_pred_indexes, _, map_mask = self.map_lossfn.forward(
                            pred_polylines=pred_polyline_logits,
                            pred_detections=pred_map_boxes,
                            target_polylines=map_tgt_polylines,
                            target_detections=map_box_labels.float()
                        )
                        avg_map_loss += map_loss
                else:
                    raise Exception(f"Invalid mapformer type {type(self.mapformer)}")

            if hasattr(self, "motionformer"):
                agent2scene_transform = MotionFormer.create_agent2scene_transforms(
                    angles=track_preds[..., 6], locs=track_preds[..., :3]
                )

                track_queries = torch.gather(track_queries, dim=2, index=track_pred_indexes[..., None])
                track_queries = track_queries[:, :, 0]

                map_queries = torch.gather(map_queries, dim=2, index=map_pred_indexes[..., None])
                map_queries = map_queries[:, :, 0]

                motion_queries, pred_motions, mode_scores, ego_motion_data = self.motionformer.forward(
                    agent_current_pos=track_preds[..., :2],
                    bev_features=bev_features,
                    ego_query=ego_track_query,
                    track_queries=track_queries,
                    map_queries=map_queries,
                    transform=agent2scene_transform,
                    track_pad_mask=track_mask,
                    map_pad_mask=map_mask,
                )
                ego_motion_query = ego_motion_data["ego_query"]

                if hasattr(self, "motion_lossfn"):
                    target_motions = sample_batch.agent_motions[:, 1:, ...]
                    ego_target_motions = sample_batch.agent_motions[:, 0, ...]

                    target_motions = torch.gather(target_motions, dim=2, index=track_target_indexes[..., None])
                    target_motions = target_motions[:, :, 0]

                    ego_pred_motions = ego_motion_data["ego_mode_traj"]
                    ego_mode_scores = ego_motion_data["ego_mode_score"]

                    motion_loss = self.motion_lossfn.forward(
                        pred_motions=pred_motions,
                        pred_mode_scores=mode_scores,
                        target_motions=target_motions,
                        ego_pred_motions=ego_pred_motions,
                        ego_pred_mode_scores=ego_mode_scores,
                        ego_target_motions=ego_target_motions,
                        transform=agent2scene_transform
                    )
                    avg_motion_loss += motion_loss

            if hasattr(self, "occformer"):
                pred_occ, pred_occ_attn_mask = self.occformer.forward(
                    bev_features=bev_features,
                    track_queries=track_queries,
                    motion_queries=motion_queries,
                    pad_mask=track_mask
                )
                if hasattr(self, "occ_lossfn"):
                    target_occ = sample_batch.occupancy_map
                    target_occ = torch.gather(target_occ, dim=2, index=track_target_indexes[..., None])
                    target_occ = target_occ[:, :, 0]
                    occ_loss = self.occ_lossfn.forward(
                        pred_occ=pred_occ, attn_mask=pred_occ_attn_mask, target_occ=target_occ
                    )
                    avg_occ_loss += occ_loss

            if hasattr(self, "planformer"):
                commands = sample_batch.command
                
                pred_plan_motion = self.planformer.forward(
                    commands=commands, 
                    bev_features=bev_features, 
                    ego_track_query=ego_track_query, 
                    ego_motion_query=ego_motion_query
                )

                if hasattr(self, "plan_lossfn"):
                    target_plan_motion = sample_batch.ego_motions[:, :self.planformer.pred_horizon, :]

                    plan_loss = self.plan_lossfn.forward(
                        pred_motion=pred_plan_motion,
                        target_motion=target_plan_motion,
                        ego_size=ego_track_labels[..., 4:6],
                        multiagent_size=track_labels[..., 4:6],
                        multiagents_motions=target_motions,
                        transform=None
                    )
                    avg_plan_loss += plan_loss

            loss = track_loss + map_loss + motion_loss + occ_loss + plan_loss
            
            loss.backward()
            self.optimizer.step()

        metrics = {
            "track_loss": avg_track_loss, 
            "map_loss": avg_map_loss, 
            "motion_loss": avg_motion_loss, 
            "occ_loss": avg_occ_loss,
            "plan_loss": avg_plan_loss, 
        }
        
        for k in metrics:
            if isinstance(metrics[k], torch.Tensor):
                metrics[k] = metrics[k].item()
            metrics[k] /= (step + 1)

        if self.ddp_mode:
            metrics = ddp_sync_metrics(metrics, self.device_or_rank)

        if verbose:
            log_msg = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            time = datetime.strftime(datetime.now(), "%d/%m/%Y %H:%M:%S")
            print(f"[Step {step} @ {time}] {log_msg}")
            
        return metrics