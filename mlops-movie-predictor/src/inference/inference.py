import os
import sys
import glob

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )   
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import onnx
import onnxruntime
from icecream import ic
from dotenv import load_dotenv

from src.utils.utils import model_dir, calculate_hash, read_hash
from src.model.movie_predictor import MoviePredictor
from src.dataset.watch_log import WatchLogDataset, get_datasets
from src.evaluate.evaluate import evaluate
from src.postprocess.postprocess import write_db

def model_validation(model_path):
    original_hash = read_hash(model_path)
    current_hash = calculate_hash(model_path)
    if original_hash == current_hash:
        ic("validation success")
        return True
    else:
        return False

def load_checkpoint():
    target_dir = model_dir(MoviePredictor.name)
    models_path = os.path.join(target_dir, "*.pth")
    latest_model = glob.glob(models_path)[-1]
    
    if model_validation(latest_model):
        checkpoint = torch.load(latest_model)
        return checkpoint
    else:
        raise FileExistsError("Not found or invalid model file")

def init_model(checkpoint):
    model = MoviePredictor(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    scaler = checkpoint["scaler"]
    contents_id_map = checkpoint["contents_id_map"]
    return model, criterion, scaler, contents_id_map

def make_inference_df(data):
    columns = "user_id content_id watch_seconds rating popularity".split()
    return pd.DataFrame(
        data=[data],
        columns=columns
    )

def inference(model, criterion, scaler, contents_id_map, data: np.array, batch_size=1):
    if data.size > 0:
        df = make_inference_df(data)
        dataset = WatchLogDataset(df, scaler=scaler)
    else:
        _, _, dataset = get_datasets()

    dataset.contents_id_map = contents_id_map
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    loss, predictions = evaluate(model, dataloader, criterion)
    ic(loss)
    ic(predictions)
    return [dataset.decode_content_id(idx) for idx in predictions]


def inference_onnx(scaler, contents_id_map, data):
    df = make_inference_df(data)
    dataset = WatchLogDataset(df, scaler=scaler)
    dataset.contents_id_map = contents_id_map
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    latest_model = get_latest_model(model_ext="onnx")
    ort_session = onnxruntime.InferenceSession(latest_model, providers=["CPUExecutionProvider"])

    predictions = []
    for data, labels in dataloader:
        ort_inputs = {ort_session.get_inputs()[0].name: data.numpy()}
        ort_outs = [ort_session.get_outputs()[0].name]

        output = ort_session.run(ort_outs, ort_inputs)
        predicted = np.argmax(output[0], 1)[0]
        predictions.append(predicted)

    return dataset.decode_content_id(predictions[0])

def recommend_to_df(recommend):
    return pd.DataFrame(
        data=recommend,
        columns="recommend_content_id".split()
    )

if __name__ == '__main__':
    load_dotenv()
    checkpoint = load_checkpoint()
    model, criterion, scaler, contents_id_map = init_model(checkpoint)
    data = np.array([1, 1209290, 4508, 7.577, 1204.764])
    recommend = inference(
        model, criterion, scaler, contents_id_map, data=np.array([]), batch_size=64
    )
    ic(recommend) # {"d1: {"d2": {"d3": value}}} #print(a["d1"]["d2"]["d3"])
    recommend_df = recommend_to_df(recommend)
    write_db(recommend_df, "mlops", "recommend")