import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import fire
import wandb
from dotenv import load_dotenv
import numpy as np

from src.dataset.watch_log import get_datasets
from src.model.movie_predictor import MoviePredictor, model_save
from src.utils.utils import init_seed, auto_increment_run_suffix
from src.train.train import train
from src.evaluate.evaluate import evaluate
from src.utils.constant import Optimizers, Models
from src.inference.inference import load_checkpoint, init_model, inference, recommend_to_df
from src.postprocess.postprocess import write_db


init_seed()
load_dotenv()

def get_runs(project_name):
    return wandb.Api().runs(path=project_name, order="-created_at")


def get_latest_run(project_name):
    runs = get_runs(project_name)
    if not runs:
        return f"{project_name}-000"

    return runs[0].name


def run_train(model_name, optimizer, batch_size=64, num_epochs=10, lr=0.001, model_ext="pth"):
    """
    run train task for mlops model

    :param model_name : as
    """
    Models.validation(model_name)
    Optimizers.validation(optimizer) 

    api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=api_key)

    project_name = "my-mlops"
    run_name = get_latest_run(project_name)
    next_run_name = auto_increment_run_suffix(run_name)

    wandb.init(
        project=project_name,
        id=next_run_name,
        name=next_run_name,
        notes="content-based movie recommend model",
        tags=["content-based", "movie", "recommend"],
        config=locals(),
    )

    # 데이터셋 및 DataLoader 생성
    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model_params = {
        "input_dim": train_dataset.features_dim,
        "num_classes": train_dataset.num_classes,
    }

    # 모델 초기화
    # model = MoviePredictor(**model_params)
    model_class = Models[model_name.upper()].value
    model = model_class(**model_params)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer_class = Optimizers[optimizer.upper()].value
    optimizer = optimizer_class(model.parameters(), lr=lr)

    epoch = 0
    train_loss = 0
    for epoch in tqdm(range(num_epochs)):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, _ = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Train-Val Loss : {train_loss - val_loss:.4f}")
        wandb.log({"Loss/Train": train_loss})
        wandb.log({"Loss/Valid": val_loss})

    model_save(
        model=model,
        model_params=model_params,
        epoch=epoch,
        optimizer=optimizer,
        loss=train_loss,
        scaler=train_dataset.scaler,
        contents_id_map=train_dataset.contents_id_map,
        ext=model_ext,
    )


def run_preprocessing(date="240809"):
    pass


def run_inference(data=None, batch_size=64):
    checkpoint = load_checkpoint()
    model, criterion, scaler, contents_id_map = init_model(checkpoint)

    if data is None:
        data = []

    data = np.array(data)

    recommend = inference(model, criterion, scaler, contents_id_map, data, batch_size)
    print(recommend)

    write_db(recommend_to_df(recommend), "mlops", "recommend")


if __name__ == "__main__":
    fire.Fire({
        "preprocessing": run_preprocessing,
        "train": run_train,
        "inference": run_inference,
    })