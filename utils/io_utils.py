import os
import cv2
import json
import yaml
import shutil
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def delete_path(path: str):
    if not os.path.exists(path): return

    if os.path.isfile(path):
        os.remove(path)
    
    elif os.path.isdir(path):
        shutil.rmtree(path)

def save_pickle_file(data: Any, path: str):
    parent = Path(path).parent
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_pickle_file(path: str) -> Any:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_json_file(data: Dict[str, Any], path: str):
    parent = Path(path).parent
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)

def load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_yaml_file(data: Dict[str, Any], path: str):
    parent = Path(path).parent
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=True)

def load_yaml_file(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def load_and_process_img(
        img_path: str, 
        img_size: Optional[Tuple[int, int]]=None, 
        scale: bool=True,
        read_flags: int=cv2.COLOR_RGB2BGR
    ) -> torch.Tensor:

    img = cv2.imread(img_path, flags=read_flags)
    if img_size is not None:
        img = cv2.resize(img, dsize=(img_size[1], img_size[0]))
    img = np.asarray(img).copy()
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)
    if scale:
        img = (img / 255).to(dtype=torch.float32)
    return img