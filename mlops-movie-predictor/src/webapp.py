import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.inference.inference import load_checkpoint, init_model, inference
from src.postprocess.postprocess import read_db


app = FastAPI()

load_dotenv()
checkpoint = load_checkpoint()
model, criterion, scaler, contents_id_map = init_model(checkpoint)


class InferenceInput(BaseModel):
    user_id: int
    content_id: int
    watch_seconds: int
    rating: float
    popularity: float

# https://api.endpoint/movie-predictor/inference?user_id=12345&content_id=456...

@app.post("/predict")
async def predict(input_data: InferenceInput):
    try:
        data = np.array([
            input_data.user_id,
            input_data.content_id,
            input_data.watch_seconds,
            input_data.rating,
            input_data.popularity
        ])
        recommend = inference(model, criterion, scaler, contents_id_map, data)
        return {"recommended_content_id": recommend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
        
@app.get("/batch-predict")
async def batch_predict(k: int = 5):
    try:
        recommend = read_db("mlops", "recommend", k=k)
        return {"recommended_content_id": recommend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)