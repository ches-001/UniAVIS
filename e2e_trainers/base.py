import os
import shutil
import torch
import pandas as pd
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from modules.bevformer import BEVFormer
from modules.trackformer import TrackFormer
from modules.mapformer import VectorMapFormer, RasterMapFormer
from modules.motionformer import MotionFormer
from modules.occformer import OccFormer
from modules.planformer import PlanFormer
from modules.lossfns.track_loss import TrackLoss
from modules.lossfns.map_loss import VectorMapLoss, RasterMapLoss
from modules.lossfns.motion_loss import MotionLoss
from modules.lossfns.occ_loss import OccLoss
from modules.lossfns.plan_loss import PlanLoss
from matplotlib import pyplot as plt
from typing import List, Tuple, Dict, Union, Optional, Any, Iterable
from utils.io_utils import save_yaml_file

class BaseTrainer:
    # purpose of the next 12 lines is for vscode highlight
    bevformer: BEVFormer
    optimizer: torch.optim.Optimizer
    trackformer: Optional[Union[TrackFormer, DDP]]
    track_lossfn: Optional[TrackLoss]
    mapformer: Optional[Union[VectorMapFormer, RasterMapFormer, DDP]]
    map_lossfn: Optional[Union[VectorMapLoss, RasterMapLoss]]
    motionformer: Optional[Union[MotionFormer, DDP]]
    motion_lossfn: Optional[MotionLoss]
    occformer: Optional[Union[OccFormer, DDP]]
    occ_lossfn: Optional[OccLoss]
    planformer: Optional[Union[PlanFormer, DDP]]
    plan_lossfn: Optional[PlanLoss]
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]

    metrics_dir = "metrics"
    checkpoints_dir = "checkpoints"
    best_dir = "best_checkpoints"
    config_dir = "configs"

    def __init__(
            self, 
            ddp_mode: bool=False, 
            device_or_rank: Union[int, str]="cpu", 
            checkpoints_path: Optional[str]=None,
            config_or_path: Optional[Union[str, Dict[str, Any]]]=None
        ):
        
        if config_or_path:
            assert isinstance(config_or_path, (str, dict))

        self.ddp_mode = ddp_mode
        self.device_or_rank = device_or_rank
        self.checkpoints_path = checkpoints_path
        self.config_or_path = config_or_path
        self._valid_modes = ["train", "eval"]

        self.start_time = datetime.strftime(datetime.now(), "%d/%m/%Y %H-%M-%S")
        self.metrics_dir = os.path.join("saved", self.start_time, self.metrics_dir)
        self.checkpoints_dir = os.path.join("saved", self.start_time, self.checkpoints_dir)
        self.best_dir = os.path.join("saved", self.start_time, self.best_dir)
        self.config_dir = os.path.join("saved", self.start_time, self.config_dir)

        # collect metrics in this list of dicts
        self._train_metrics: List[Dict[str, float]] = []
        self._eval_metrics: List[Dict[str, float]] = []

        self._save_config()

    
    def train(self, dataloader: DataLoader, verbose: bool=False) -> Dict[str, float]:
        raise NotImplementedError
    

    def eval(self, dataloader: DataLoader, verbose: bool=False) -> Dict[str, float]:
        raise NotImplementedError
    

    def step(self, dataloader: DataLoader, mode: str, verbose: bool=False) -> Dict[str, float]:
        raise NotImplementedError


    def set_module(self, module: Optional[nn.Module], name: str, is_model: bool):
        if module is not None:
            setattr(self, name, module)
            if is_model:
                getattr(self, name).to(self.device_or_rank)
                if self.ddp_mode:
                    setattr(self, name, DDP(getattr(self, name), device_ids=[self.device_or_rank, ]))

    
    def set_optimizer(self, optim_name: str, **kwargs):
        params = []
        if hasattr(self, "bevformer"):
            params.append(self.bevformer.parameters())
        if hasattr(self, "trackformer"):
            params.append(self.trackformer.parameters())
        if hasattr(self, "mapformer"):
            params.append(self.mapformer.parameters())
        if hasattr(self, "motionformer"):
            params.append(self.motionformer.parameters())
        if hasattr(self, "occformer"):
            params.append(self.occformer.parameters())
        if hasattr(self, "planformer"):
            params.append(self.planformer.parameters())
        if len(params) == 0:
            raise ValueError("model params for optimizer cannot be empty")
        
        setattr(self, "optimizer", getattr(torch.optim, optim_name)(params=params, **kwargs))


    def set_lr_scheduler(self, scheduler_name: str, interval: int, **kwargs):
        if not hasattr(self, "optimizer") or self.optimizer is not None:
            raise ValueError("self.optimizer must be set to a non-NoneType value before setting a scheduler")
        assert isinstance(interval, int)
        setattr(self, "lr_scheduler", getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer=self.optimizer, *kwargs))
        setattr(self, "lr_scheduler_interval", interval)
                    

    def _toggle_module_mode(self, module_name: str, mode: str):
        if hasattr(self, module_name):
            getattr(getattr(self, module_name), mode)()


    def _save_module(self, module_name: str, as_best: bool=False, epoch: Optional[int]=None):
        if not hasattr(self, module_name):
            return
        module: nn.Module = getattr(self, module_name)
        
        if self.ddp_mode:
            if self.device_or_rank != 0:
                return
            module: nn.Module = getattr(self, module_name).module
        state_dict = module.state_dict()
        
        if as_best:
            path = self.best_dir
        else:
            assert epoch is not None
            path = os.path.join(self.checkpoints_dir, f"epoch_{str(epoch).zfill(4)}")
        torch.save(state_dict, os.path.join(path, f"{module_name}_state.pth.tar"))


    def _save_optimizer(self, as_best: bool=False, epoch: Optional[int]=None):
        state_dict = self.optimizer.state_dict()
        if as_best:
            path = self.best_dir
        else:
            assert epoch is not None
            path = os.path.join(self.checkpoints_dir, f"epoch_{str(epoch).zfill(4)}")
        torch.save(state_dict, os.path.join(path, f"optimizer_state.pth.tar"))

    
    def _save_config(self):
        if not self.config_or_path:
            return
        
        if isinstance(self.config_or_path, str):
            shutil.copy(self.config_or_path, self.config_dir)
            return
        save_yaml_file(self.config_or_path, self.config_dir)


    def _load_state_dict(self, name: str, path: str, is_model: bool):
        # if DDP mode, we do not only need to do this for rank-0 devices but
        # across all devices to ensure that the model and optimizer states 
        # begin at same point
        if not hasattr(self, name):
            return
        file = os.path.join(path, f"{name}_state.pth.tar")

        if not os.path.isfile(file):
            return
        state_dict = torch.load(file)

        if not self.ddp_mode or not is_model:
            getattr(self, name).load_state_dict(state_dict)
        else:
            getattr(self, name).module.load_state_dict(state_dict)

    
    def save_best(self):
        self._save_module("bevformer", as_best=True)
        self._save_module("trackformer", as_best=True)
        self._save_module("mapformer", as_best=True)
        self._save_module("motionformer", as_best=True)
        self._save_module("occformer", as_best=True)
        self._save_module("planformer", as_best=True)
        self._save_optimizer(as_best=True)


    def save_checkpoints(self, epoch: int):
        self._save_module("bevformer", as_best=False, epoch=epoch)
        self._save_module("trackformer", as_best=False, epoch=epoch)
        self._save_module("mapformer", as_best=False, epoch=epoch)
        self._save_module("motionformer", as_best=False, epoch=epoch)
        self._save_module("occformer", as_best=False, epoch=epoch)
        self._save_module("planformer", as_best=False, epoch=epoch)
        self._save_optimizer(as_best=False, epoch=epoch)


    def load_checkpoints(self, path: str):
        self._load_state_dict("bevformer", path, is_model=True)
        self._load_state_dict("trackformer", path, is_model=True)
        self._load_state_dict("mapformer", path, is_model=True)
        self._load_state_dict("motionformer", path, is_model=True)
        self._load_state_dict("occformer", path, is_model=True)
        self._load_state_dict("planformer", path, is_model=True)
        self._load_state_dict("optimizer", path, is_model=False)
            

    def save_metrics(self, save_plot: bool, figsize: Tuple[float, float]=(15, 60)):
        # metrics ought to be saved by only one device process since they will be collected
        # and synced across all devices during training anyways
        if self.ddp_mode and self.device_or_rank != 0:
            return
        train_metrics_df, eval_metrics_df = self._metrics_to_csv()
        
        if save_plot:
            self._plot_and_save_metrics_df(train_metrics_df, figsize)
            self._plot_and_save_metrics_df(eval_metrics_df, figsize)
    

    def _metrics_to_csv(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.ddp_mode and self.device_or_rank != 0:
            return
        if not os.path.isdir(self.metrics_dir):
            os.makedirs(self.metrics_dir, exist_ok=True)
        train_metrics_df = pd.DataFrame(self._train_metrics)
        train_metrics_df.to_csv(os.path.join(self.metrics_dir, "train_metrics.csv"), index=False)

        eval_metrics_df = pd.DataFrame(self._eval_metrics)
        eval_metrics_df.to_csv(os.path.join(self.metrics_dir, "eval_metrics.csv"), index=False)

        return train_metrics_df, eval_metrics_df


    def _plot_and_save_metrics_df(self, df: pd.DataFrame, mode: str, figsize: Tuple[float, float]=(15, 60)):        
        valid_modes = self._valid_modes
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        nrows = len(df.columns)
        fig, axs = plt.subplots(nrows, 1, figsize=figsize)

        if nrows == 1:
            label = df.columns[0]
            axs.plot(df[df.columns[0]].to_numpy())
            axs.grid(visible=True)
            axs.set_xlabel("Epoch")
            axs.set_ylabel(label)
            axs.set_title(f"[{mode.title()}] {label} vs Epoch", fontsize=24)
            axs.tick_params(axis='x', which='major', labelsize=20)
        else:
            for i, col in enumerate(df.columns):
                label = col.replace("_", " ").title()
                axs[i].plot(df[col].to_numpy())
                axs[i].grid(visible=True)
                axs[i].set_xlabel("Epoch")
                axs[i].set_ylabel(label)
                axs[i].set_title(f"[{mode.title()}] {label} vs Epoch", fontsize=24)
                axs[i].tick_params(axis='x', which='major', labelsize=20)

        if os.path.isdir(self.metrics_dir): os.makedirs(self.metrics_dir, exist_ok=True)
        fig.savefig(os.path.join(self.metrics_dir, f"{mode}_metrics_plot.jpg"))
        fig.clear()
        plt.close(fig)