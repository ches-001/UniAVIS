import torch
import numpy as np
from enum import Enum
import torch.distributed as distr
from typing import Dict, Union, List, Optional

def ddp_setup():
    distr.init_process_group(backend="nccl")

def ddp_destroy():
    distr.destroy_process_group()

def ddp_broadcast(val: torch.Tensor, src_rank: Union[int, str]):
    handle = distr.broadcast(val, src=src_rank, async_op=True)
    handle.wait()

def ddp_sync_vals(
        val: Union[int, float, torch.Tensor, List[Union[int, float]], np.ndarray],
        rank: int,
        op: Optional[Enum]=None
    ) -> torch.Tensor:
    if isinstance(val, np.ndarray):
        val = torch.from_numpy(val).to(device=f"cuda:{rank}")
    elif isinstance(val, torch.Tensor):
        val = val.to(device=f"cuda:{rank}")
    else:
        val = torch.tensor(val, device=f"cuda:{rank}")
    # distr.all_reduce(...) is a technique used to collect tensors across multiple
    # devices, to reduce them by a single operation (sum, average, product, etc)
    handle = distr.all_reduce(val, op=(op or distr.ReduceOp.SUM), async_op=True)
    # handle.Wait(...) ensures the operation is enqueued, but not necessarily complete.
    handle.wait()
    return val

def ddp_sync_metrics(metrics: Dict[str, float], rank: int, op: Enum=distr.ReduceOp.AVG) -> Dict[str, float]:
    # Code runs on each device (rank).
    keys = list(metrics.keys())
    metrics_vals = list(metrics.values())
    metrics_vals = torch.tensor(metrics_vals, dtype=torch.float32, device=f"cuda:{rank}")
    metrics_vals = ddp_sync_vals(rank, metrics_vals, op=op)
    metrics = {k:v.item() for k, v in zip(keys, metrics_vals)}
    return metrics


def ddp_log(log: str, device: Union[str, int], is_ddp: bool, device_to_log: Union[str, int]=-1):
    if is_ddp:
        if device_to_log != -1:
            if device == device_to_log:
                print(log)
        else:
            print(log)
        return
    print(log)