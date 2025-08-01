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

def save_pickle_file(data: Any, path: str, **kwargs):
    parent = Path(path).parent
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, **kwargs)

def load_pickle_file(path: str, **kwargs) -> Any:
    with open(path, "rb") as f:
        data = pickle.load(f, **kwargs)
    return data

def save_json_file(data: Dict[str, Any], path: str, **kwargs):
    parent = Path(path).parent
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, **kwargs)

def load_json_file(path: str, **kwargs) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f, **kwargs)
    return data

def save_yaml_file(data: Dict[str, Any], path: str, **kwargs):
    parent = Path(path).parent
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, **kwargs)

def load_yaml_file(path: str, **kwargs) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f, **kwargs)
    return data

def load_img(img_path: str, read_flags: int=cv2.COLOR_RGB2BGR) -> np.ndarray:
    img = cv2.imread(img_path, flags=read_flags)
    return img