import os
import sys

# 작업 디렉터리를 명시적으로 설정 (여기에 컨테이너 내부의 정확한 경로를 입력)
project_dir = "/opt/mlops-movie-predictor/src"

# sys.path에 프로젝트 경로를 추가
sys.path.append(project_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import fire

# 절대 경로를 통해 모듈을 임포트
from utils.constant import Optimizers, Models
from dataset.watch_log import get_datasets
from model.movie_predictor import MoviePredictor
from utils.utils import init_seed
from train.train import train
from evaluate.evaluate import evaluate
from model.movie_predictor import MoviePredictor, model_save  # model_save 추가


init_seed()

def run_train(model_name, optimizer, num_epochs=10, lr=0.001, model_ext="pth"):
    Models.validation(model_name)
    Optimizers.validation(optimizer)

if __name__ == '__main__':
    fire.Fire({
        "train": run_train,
    })
    # 데이터셋 및 DataLoader 생성
    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    # 모델 초기화
    model_class = Models[model_name.upper()].value
    model = model_class(**model_params)
    model_params = {
        "input_dim": train_dataset.features_dim,
        "num_classes": train_dataset.num_classes
    }
    model = MoviePredictor(**model_params)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer_class = Optimizers[optimizer.upper()].value
    optimizer = optimizer_class(model.parameters(), lr=lr)

    epoch = 0
    train_loss = 0

    # 학습 루프
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, _ = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val-Train Loss: {val_loss - train_loss:.4f}")


    model_save(
        model=model,
        model_params=model_params,
        epoch=num_epochs,
        optimizer=optimizer,
        loss=train_loss,
        scaler=train_dataset.scaler,
        contents_id_map=train_dataset.contents_id_map,
    )

    # 테스트
    model.eval()
    test_loss, predictions = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    print([train_dataset.decode_content_id(idx) for idx in predictions])

    model_ext = "onnx"  # or "pth"
    model_save(
        model=model,
        model_params=model_params,
        epoch=num_epochs,
        optimizer=optimizer,
        loss=train_loss,
        scaler=train_dataset.scaler,
        contents_id_map=train_dataset.contents_id_map,
        ext=model_ext,
    )
  
  
