import os
import pandas as pd
import torch.nn as nn
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
from typing import List, Tuple, Dict, Union, Optional

class BaseTrainer:
    # purpose of the next 11 lines is for vscode highlight
    bevformer: BEVFormer
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

    metrics_dir = "metrics/detection"
    checkpoints_dir = "saved_model/detection/checkpoints"
    best_model_dir = "saved_model/detection/best_model"

    def __init__(self, ddp_mode: bool=False, device_or_rank: Union[str, float]="cpu"):
        self.ddp_mode = ddp_mode
        self.device_or_rank = device_or_rank
        self._valid_modes = ["train", "eval"]

        # collect metrics in this list of dicts
        self._train_metrics: List[Dict[str, float]] = []
        self._eval_metrics: List[Dict[str, float]] = []

    def _toggle_module_mode(self, module_name: str, mode: str):
        if hasattr(self, module_name):
            getattr(getattr(self, module_name), mode)()

    def _zero_module_grads(self, module_name: str):
        if hasattr(self, module_name):
            getattr(self, module_name).zero_grad()

    def _set_module(self, module: Optional[nn.Module], name: str, is_model: bool):
        if module is not None:
            setattr(self, name, module)
            if is_model:
                getattr(self, name).to(self.device_or_rank)
                if self.ddp_mode:
                    setattr(self, name, DDP(getattr(self, name), device_ids=[self.device_or_rank, ]))
            
    def save_metrics_plots(self, figsize: Tuple[float, float]=(15, 60)):
        # metrics ought to be saved by only one device process since they will be collected
        # and synced across all devices involved
        if not self.ddp_mode or (self.ddp_mode and self.device_or_rank in [0, f"cuda:0"]):
            self._save_metrics_plots("train", figsize)
            self._save_metrics_plots("eval", figsize)
    
    def metrics_to_csv(self):
        if not self.ddp_mode or (self.ddp_mode and self.device_or_rank in [0, f"cuda:0"]):
            if not os.path.isdir(self.metrics_dir): 
                os.makedirs(self.metrics_dir, exist_ok=True)
            pd.DataFrame(self._train_metrics).to_csv(os.path.join(self.metrics_dir, "train_metrics.csv"), index=False)
            pd.DataFrame(self._eval_metrics).to_csv(os.path.join(self.metrics_dir, "eval_metrics.csv"), index=False)

    def _save_metrics_plots(self, mode: str, figsize: Tuple[float, float]=(15, 60)):        
        valid_modes = self._valid_modes
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        df = pd.DataFrame(getattr(self, f"_{mode}_metrics"))
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