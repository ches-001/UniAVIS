import os
import yaml
import time
import tqdm
import math
import logging
import torch
import torch.nn as nn
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from utils.ddp_utils import ddp_sync_metrics
from utils.io_utils import load_yaml_file, save_yaml_file
from .base import BaseTrainer
from typing import *


LOGGER = logging.getLogger(__name__)

class PerceptionTrainer(BaseTrainer):

    def __init__(
        self, 
        bevformer: nn.Module,
        trackformer: nn.Module,
        mapformer: nn.Module,
        checkpoint_path: Optional[str]=None,
        device_or_rank: Union[int, str]="cpu",
        ddp_mode: bool=False,
        config_or_path: Optional[str]=None
    ):
        super(PerceptionTrainer, self).__init__(ddp_mode, device_or_rank)

        self.bevformer = bevformer
        self.trackformer = trackformer
        self.mapformer = mapformer

        self.bevformer.to(self.device_or_rank)
        self.trackformer.to(self.device_or_rank)
        self.mapformer.to(self.device_or_rank)

        if self.ddp_mode:
            self.bevformer = DDP(self.bevformer, device_ids=[self.device_or_rank, ])
            self.trackformer = DDP(self.trackformer, device_ids=[self.device_or_rank, ])
            self.mapformer = DDP(self.mapformer, device_ids=[self.device_or_rank, ])

        self.config = load_yaml_file(config_or_path) if isinstance(config_or_path, str) else config_or_path

        self.last_epoch = 0
        
        self.checkpoints_dir = os.path.join(self.checkpoints_dir, str(int(time.time())))

        self._save_config_copy(config_or_path, to_checkpoint_dir=True)
        self._save_config_copy(config_or_path, to_checkpoint_dir=False)

        # load checkpoint if any
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    

    def _save(self, path: str, snapshot_mode: bool=True):
        pass
    

    def _save_config_copy(self, config_path: str, to_checkpoint_dir: bool):
        pass


    def save_best_model(self):
        # If DDP mode, we only need to ensure that the model is only being saved at
        # rank-0 device. It does not neccessarily matter the rank, as long as the
        # model saving only happens on one rank only (one device) since the model
        # is exactly the same across all
        pass


    def save_checkpoint(self):
        # similar to the `save_best_model` method, the model for only one device needs
        # to be saved.
        pass
    

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        # if DDP mode, we do not need to only do this for rank-0 devices but
        # across all devices to ensure that the model and optimizer states 
        # begin at same point
        pass
    

    def train(self, dataloader: DataLoader, verbose: bool=False) -> Dict[str, float]:
        result = self.step(dataloader, "train", verbose)
        if self.lr_scheduler and (self.last_epoch % self.lr_schedule_interval == 0):
            self.lr_scheduler.step()
        self.last_epoch += 1
        return result
        

    def evaluate(self, dataloader: DataLoader, verbose: bool=False) -> Dict[str, float]:        
        with torch.no_grad():
            return self.step(dataloader, "eval", verbose)


    def step(self, dataloader: DataLoader, mode: str, verbose: bool=False) -> Dict[str, float]:
        if mode not in self._valid_modes:
            raise ValueError(f"Invalid mode {mode} expected either one of {self._valid_modes}")
        getattr(self.bevformer, mode)()
        getattr(self.trackformer, mode)()
        getattr(self.mapformer, mode)()
        metrics = {}
        if self.ddp_mode:
            # invert progress bar position such that the last (rank n-1) is at
            # the top and the first (rank 0) at the bottom. This is because the
            # first rank will be the one logging all the metrics
            world_size = int(os.environ["WORLD_SIZE"])
            position = (
                self.device_or_rank 
                if not isinstance(self.device_or_rank, str) else 
                int(self.device_or_rank.replace("cuda:", ""))
            )
            position = abs(position - (world_size - 1))
            pbar = tqdm.tqdm(enumerate(dataloader), position=position)
        else:
            total = math.ceil(len(dataloader.dataset) / dataloader.batch_size)
            pbar = tqdm.tqdm(enumerate(dataloader), total=total)

        for count, sample_data in pbar:
            pass

        return metrics
    

    