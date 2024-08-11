import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset.watch_log import get_datasets
from src.model.movie_predictor import MoviePredictor
from src.utils.utils import init_seed
from src.train.train import train
from src.evaluate.evaluate import evaluate

from src.model.movie_predictor import MoviePredictor, model_save

init_seed()


if __name__ == '__main__':
    # 데이터셋 및 DataLoader 생성
    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    # 모델 초기화
    model_params = {
        "input_dim": train_dataset.features_dim,
        "num_classes": train_dataset.num_classes
    }
    model = MoviePredictor(**model_params)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 루프
    num_epochs = 15

    epoch = 0
    train_loss = 0
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, _ = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val-Train Loss : {val_loss-train_loss:.4f}")
    
    model_ext = "onnx"
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
    # 테스트
    model.eval()
    test_loss, predictions = evaluate(model, test_loader, criterion)
    print(f"Test Loss : {test_loss:.4f}")
    print([train_dataset.decode_content_id(idx) for idx in predictions])