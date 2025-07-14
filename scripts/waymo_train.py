import os
import sys
import argparse
import logging
import random
import torch
import numpy as np
from e2e_trainers.waymo_e2e import WaymoE2ETrainer
from modules.bevformer import BEVFormer
from modules.trackformer import TrackFormer
from modules.mapformer import RasterMapFormer, VectorMapFormer
from modules.motionformer import MotionFormer
from modules.occformer import OccFormer
from modules.planformer import PlanFormer
from modules.lossfns.track_loss import TrackLoss
from modules.lossfns.map_loss import RasterMapLoss, VectorMapLoss
from modules.lossfns.motion_loss import MotionLoss
from modules.lossfns.occ_loss import OccLoss
from modules.lossfns.plan_loss import PlanLoss
from utils.io_utils import load_yaml_file
from utils.ddp_utils import ddp_setup, ddp_destroy, ddp_broadcast, ddp_log
from torch.utils.data import DataLoader, DistributedSampler
from dataset.waymo_dataset import WaymoDataset
from typing import Dict, Any, Union

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = "config/modules.yaml"
DEFAULT_DATA_PATH = "data/waymo"

def run(args: argparse.Namespace):
    # TODO: Handle logging

    if args.use_ddp:
        ddp_setup()

    assert os.path.isfile(args.config_path)

    config = load_yaml_file(args.config_path)
    train_path = os.path.join(args.data_path, "training.pickle")
    eval_path = os.path.join(args.data_path, "validation.pickle")
    metadata_path = os.path.join(args.data_path, "metadata.json")
    num_workers = args.num_cpu_workers

    assert os.path.isfile(train_path)
    assert os.path.isfile(eval_path)

    dataset_config = config["dataset"]
    train_dataset = WaymoDataset(train_path, metadata_path, **dataset_config)
    eval_dataset = WaymoDataset(eval_path, metadata_path, **dataset_config)
    train_sampler = None
    eval_sampler = None
    shuffle = True
    device_or_rank = "cuda" if torch.cuda.is_available() else "cpu"

    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
        shuffle = False
        try:
            device_or_rank = int(os.environ["LOCAL_RANK"])
        except KeyError as e:
            LOGGER.error(
                f"{e}. This LOCAL_RANK key not existing in the environment variable is a clear "
                "indication that you need to execute this script with torchrun if you wish to "
                "use the DDP mode (ddp=True)"
            )
            sys.exit(0)

    collate_fn = lambda batch : WaymoDataset.collate_fn(batch, frames_per_sample=train_dataset.frames_per_sample)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=train_sampler, collate_fn=collate_fn
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=eval_sampler, collate_fn=collate_fn
    )
    
    trainer = WaymoE2ETrainer(device_or_rank=device_or_rank, ddp_mode=args.use_ddp, config_or_path=args.config_path)
    
    bevformer = BEVFormer(**config["bevformer"])
    trainer.set_module(bevformer, "bevformer", is_model=True)

    if args.perception_train:
        trackformer = TrackFormer(**config["trackformer"])
        track_lossfn = TrackLoss(**config["track_loss"])
        
        if args.use_vectormap:
            mapformer = VectorMapFormer(**config["vectormapformer"])
            map_lossfn = VectorMapLoss(**config["vectormap_loss"])
        else:
            mapformer = RasterMapFormer(**config["rastermapformer"])
            map_lossfn = RasterMapLoss(**config["rastermap_loss"])
            
        trainer.set_module(trackformer, "trackformer", is_model=True)
        trainer.set_module(track_lossfn, "track_lossfn", is_model=False)
        trainer.set_module(mapformer, "mapformer", is_model=True)
        trainer.set_module(map_lossfn, "map_lossfn", is_model=False)

    if args.e2e_train:
        motionformer = MotionFormer(**config["motionformer"])
        motion_lossfn = MotionLoss(**config["motion_loss"])

        occformer = OccFormer(**config["occformer"])
        occ_lossfn = OccLoss(**config["occ_loss"])

        planformer = PlanFormer(**config["planformer"])
        plan_lossfn = PlanLoss(**config["plan_loss"])

        cluster_config = config["motion_cluster"]
        if args.use_ddp:
            if device_or_rank == 0:
                motion_anchors, *_ = WaymoDataset.cluster_agent_motion_endpoints(
                    train_dataset.data,
                    device=device_or_rank,
                    **cluster_config
                )
            else:
                motion_anchors = torch.zeros(cluster_config["num_clusters"], 2, dtype=torch.float32, device=device_or_rank)
            ddp_broadcast(motion_anchors, src_rank=0)
        else:
            motion_anchors, *_ = WaymoDataset.cluster_agent_motion_endpoints(
                    train_dataset.data,
                    device=device_or_rank,
                    **cluster_config
                )
        motionformer.set_agent_anchors(motion_anchors)

        trainer.set_module(motionformer, "motionformer", is_model=True)
        trainer.set_module(motion_lossfn, "motion_lossfn", is_model=False)
        trainer.set_module(planformer, "planformer", is_model=True)
        trainer.set_module(plan_lossfn, "plan_lossfn", is_model=False)
        trainer.set_module(occformer, "occformer", is_model=True)
        trainer.set_module(occ_lossfn, "occ_lossfn", is_model=False)

        optimizer_name = config["optimizer"]["name"]
        optimizer_kwargs = config["optimizer"]["kwargs"]
        trainer.set_optimizer(optimizer_name, **optimizer_kwargs)

        lr_scheduler_config = config.get("lr_scheduler")
        if lr_scheduler_config:
            lr_scheduler_name = lr_scheduler_config["name"]
            lr_scheduler_interval = lr_scheduler_config["interval"]
            lr_scheduler_kwargs = lr_scheduler_config["kwargs"]
            trainer.set_lr_scheduler(lr_scheduler_name, lr_scheduler_interval, **lr_scheduler_kwargs)
    
    best_loss = torch.inf
    best_epoch = None
    for epoch in range(0, args.epochs):
        trainer.train(train_dataloader, verbose=args.verbose)
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            eval_metrics = trainer.eval(eval_dataloader, verbose=args.verbose)

            eval_loss = eval_metrics["loss"]
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_epoch = epoch
                trainer.save_best()

        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs - 1:
            trainer.save_checkpoints(epoch)

    trainer.save_metrics(save_plot=True, figsize=(15, 60))

    if args.use_ddp:
        ddp_destroy()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"UniAVIS (on Waymo)")

    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH, help="Config path")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Dataset path")
    parser.add_argument("--use_ddp", action="store_true", help="Switch to DDP mode if True")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_cpu_workers", type=int, default=1, help="Number of CPU workers")
    parser.add_argument("--perception_train", action="store_true", help="Train Perception (TrackFormer and Mapformer) if True")
    parser.add_argument("--e2e_train", action="store_true", help="Train End2End (Perception, Prediction and Planning) if True")
    parser.add_argument("--use_vectormap", action="store_true", help="Use VectorMapFormer if True else RasterMapFormer")
    parser.add_argument("--verbose", action="store_true", help="If True, log training outputs")
    parser.add_argument("--eval_interval", type=int, default=5, help="Number of epochs to skip till the next evaluation")
    parser.add_argument("--checkpoint_interval", type=int, default=3, help="Number of epochs to skip till the next chedkpoint is saved")
    parser.add_argument("--rng_seed", type=int, default=42, help="Seed for Random Number Generator (RNG)")

    args = parser.parse_args()

    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # For single GPU / CPU training:: scripts/waymo_train.py
    # For multiple GPU training:: torchrun --standalone --nproc_per_node=gpu scripts/waymo_train.py --use_ddp
    run(args)