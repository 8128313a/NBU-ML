from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version


app = FastAPI()


class Input(BaseModel):
    input: array


class PredictionOut(BaseModel):
    classification: int


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: Input)
    classification= predict_pipeline(payload.Input)
    return {"classifier": classification}