"""Serviço FastAPI para análise de sentimentos facial."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fer import FER
from pydantic import BaseModel


@dataclass
class EmotionPrediction:
    """Representa a emoção predominante detectada em um rosto."""

    label: str
    confidence: float
    box: Tuple[int, int, int, int]


class HealthResponse(BaseModel):
    status: str


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class PredictionItem(BaseModel):
    label: str
    confidence: float
    box: BoundingBox


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]


@lru_cache(maxsize=1)
def get_detector() -> FER:
    """Retorna uma instância compartilhada do detector de emoções."""

    # O detector FER utiliza PyTorch/TensorFlow e fará uso de GPU quando disponível
    # nas dependências instaladas. O parâmetro mtcnn melhora a detecção facial.
    return FER(mtcnn=True)


def predict_emotions(detector: FER, frame: np.ndarray) -> list[EmotionPrediction]:
    """Retorna as emoções predominantes para cada rosto no quadro fornecido."""

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_emotions(rgb_frame) or []

    predictions: list[EmotionPrediction] = []
    for result in results:
        x, y, w, h = result["box"]
        emotions: Dict[str, float] = result["emotions"]
        label, confidence = max(emotions.items(), key=lambda item: item[1])
        predictions.append(
            EmotionPrediction(
                label=label,
                confidence=float(confidence),
                box=(int(x), int(y), int(w), int(h)),
            )
        )
    return predictions


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decodifica bytes em um frame OpenCV."""

    image_array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Os bytes fornecidos não representam uma imagem válida.")
    return frame


app = FastAPI(
    title="Análise de Sentimentos Facial",
    version="1.0.0",
    description="API para detectar emoções em imagens faciais usando a biblioteca FER.",
)


@app.get("/", response_model=HealthResponse, tags=["status"])
async def root() -> HealthResponse:
    """Endpoint básico para verificar se a API está acessível."""

    return HealthResponse(status="ok")


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def healthcheck() -> HealthResponse:
    """Retorna o status de saúde do serviço."""

    return HealthResponse(status="healthy")


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Recebe uma imagem e retorna as emoções detectadas em cada rosto."""

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem válido.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="O arquivo enviado está vazio.")

    try:
        frame = load_image_from_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detector = get_detector()
    predictions = predict_emotions(detector, frame)

    return PredictionResponse(
        predictions=[
            PredictionItem(
                label=pred.label,
                confidence=pred.confidence,
                box=BoundingBox(
                    x=pred.box[0],
                    y=pred.box[1],
                    width=pred.box[2],
                    height=pred.box[3],
                ),
            )
            for pred in predictions
        ]
    )


__all__ = [
    "app",
    "EmotionPrediction",
    "BoundingBox",
    "PredictionItem",
    "PredictionResponse",
    "predict_emotions",
    "load_image_from_bytes",
]
