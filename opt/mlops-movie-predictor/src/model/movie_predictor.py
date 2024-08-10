import torch.nn as nn

import os
import datetime

# ...
import torch

from utils.utils import model_dir


def model_save(model, model_params, epoch, optimizer, loss, scaler, contents_id_map, ext="pth"):
    save_dir = model_dir(model.name)
    os.makedirs(save_dir, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    dst = os.path.join(save_dir, f"E{epoch}_T{current_time}.{ext}")  # ext 부분 수정
    if ext == "pth":
        torch.save({
            "epoch": epoch,
            "model_params": model_params,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "scaler": scaler,
            "contents_id_map": contents_id_map,
        }, dst)
    elif ext == "onnx":
        dummy_input = torch.randn(1, model.input_dim)
        torch.onnx.export(
            model,
            dummy_input,
            dst,
            export_params=True
        )
    else:
        raise ValueError(f"Invalid model export extension : {ext}")

class MoviePredictor(nn.Module):
    name = "movie_predictor"
    
    def __init__(self, input_dim, num_classes):
        super(MoviePredictor, self).__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x
