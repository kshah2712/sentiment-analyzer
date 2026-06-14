import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
from typing import List
import time

# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analyzer API",
    description="Analyzes sentiment of text using DistilBERT",
    version="1.0.0"
)

# ── Load model ─────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, "models/sentiment_model")

print("Loading sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=MODEL_PATH,
    truncation=True,
    max_length=128
)
print("Model loaded!")

# ── Schemas ────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      description="Text to analyze")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic! Best film I've seen this year."
            }
        }


class BatchInput(BaseModel):
    texts: List[str] = Field(..., min_length=1,
                              description="List of texts to analyze")

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Amazing product, highly recommend!",
                    "Terrible quality, complete waste of money.",
                    "It was okay, nothing special."
                ]
            }
        }


# ── Routes ─────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "Sentiment Analyzer API",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "endpoints": {
            "health":        "GET  /health",
            "predict":       "POST /predict",
            "predict_batch": "POST /predict/batch",
            "docs":          "GET  /docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": MODEL_PATH
    }


@app.post("/predict")
def predict(data: TextInput):
    try:
        start = time.time()
        result = sentiment_pipeline(data.text)[0]
        latency = round(time.time() - start, 3)

        label = result["label"]
        score = round(result["score"], 4)

        return {
            "text": data.text[:100] + "..." if len(data.text) > 100 else data.text,
            "sentiment": label,
            "confidence": score,
            "scores": {
                "positive": score if label == "POSITIVE" else round(1 - score, 4),
                "negative": score if label == "NEGATIVE" else round(1 - score, 4)
            },
            "emoji": "😊" if label == "POSITIVE" else "😞",
            "latency_seconds": latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(data: BatchInput):
    try:
        if len(data.texts) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 texts per batch request"
            )

        start   = time.time()
        results = sentiment_pipeline(data.texts, batch_size=16)
        latency = round(time.time() - start, 3)

        predictions = []
        positive_count = 0

        for text, result in zip(data.texts, results):
            label = result["label"]
            score = round(result["score"], 4)
            if label == "POSITIVE":
                positive_count += 1
            predictions.append({
                "text": text[:80] + "..." if len(text) > 80 else text,
                "sentiment": label,
                "confidence": score,
                "emoji": "😊" if label == "POSITIVE" else "😞"
            })

        return {
            "total": len(data.texts),
            "positive_count": positive_count,
            "negative_count": len(data.texts) - positive_count,
            "positive_percentage": round(positive_count / len(data.texts) * 100, 1),
            "predictions": predictions,
            "latency_seconds": latency
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))