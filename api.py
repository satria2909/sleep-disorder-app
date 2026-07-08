from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas import PredictionRequest
from services import predict_sleep

app = FastAPI(
    title="Sleep Disorder API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Sleep Disorder Prediction API"
    }


@app.get("/health")
def health():
    return {
        "status": "OK"
    }


@app.post("/predict")
def predict(data: PredictionRequest):

    return predict_sleep(data)
