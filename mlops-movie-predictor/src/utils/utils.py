import os
import random

import numpy as np
import torch


def init_seed():
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def project_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_dir(model_name):
    return os.path.join(
        project_path(),
        "models",
        model_name
    )
    
def auto_increment_run_suffix(name: str, pad=3):
    suffix = name.split("-")[-1]
    next_suffix = str(int(suffix) + 1).zfill(pad)
    return name.replace(suffix, next_suffix)