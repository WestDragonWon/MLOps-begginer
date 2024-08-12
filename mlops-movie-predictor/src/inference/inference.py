import os
import sys
import glob

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
import torch.nn as nn

from src.utils.utils import model_dir
from src.model.movie_predictor import MoviePredictor


def load_checkpoint():
    target_dir = model_dir(MoviePredictor.name)
    models_path = os.path.join(target_dir, "*.pth")
    latest_model = glob.glob(models_path)[-1]
    
    checkpoint = torch.load(latest_model)
    return checkpoint

def init_model(checkpoint):
    model = MoviePredictor(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    scaler = checkpoint["scaler"]
    contents_id_map = checkpoint["contents_id_map"]
    return model, criterion, scaler, contents_id_map