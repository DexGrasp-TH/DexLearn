from typing import Dict
import json
import torch
import random
import numpy as np
import os


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_json(file_path):
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            json_params = json.load(file_p)
    else:
        json_params = file_path
    return json_params


def write_json(data: Dict, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=1)


def load_scene_cfg(scene_path):
    scene_cfg = np.load(scene_path, allow_pickle=True).item()

    def update_relative_path(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                update_relative_path(v)
            elif k.endswith("_path") and isinstance(v, str):
                d[k] = os.path.join(os.path.dirname(scene_path), v)
        return

    update_relative_path(scene_cfg["scene"])

    return scene_cfg
