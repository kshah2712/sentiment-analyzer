import os
import json
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

# ── Config ─────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_DIR = "models/sentiment_model"


def evaluate_model():
    print("="*50)
    print("  Evaluating Pretrained Sentiment Model")
    print("="*50)

    # load test data
    test_df = pd.read_csv("data/raw/test.csv")
    print(f"\nTest samples: {len(test_df)}")
    print(f"Label distribution:\n{test_df['label'].value_counts()}")

    # load pretrained pipeline
    print(f"\n=== LOADING PRETRAINED MODEL ===")
    print(f"Model: {MODEL_NAME}")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        truncation=True,
        max_length=128
    )
    print("Model loaded successfully!")

    # evaluate on test set
    print(f"\n=== EVALUATING ON TEST SET ===")
    print("Running predictions (this may take a few minutes)...")

    # predict in batches
    texts = list(test_df["text"])
    true_labels = list(test_df["label"])

    results = sentiment_pipeline(texts, batch_size=32)

    # convert predictions to 0/1
    pred_labels = [
        1 if r["label"] == "POSITIVE" else 0
        for r in results
    ]

    # calculate metrics
    acc = accuracy_score(true_labels, pred_labels)
    f1  = f1_score(true_labels, pred_labels, average="weighted")

    print(f"\n=== RESULTS ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        true_labels, pred_labels,
        target_names=["Negative", "Positive"]
    ))

    # save model locally
    print(f"\n=== SAVING MODEL ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to: {OUTPUT_DIR}")

    # save results
    results_dict = {
        "model": MODEL_NAME,
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "test_samples": len(test_df)
    }
    with open("models/results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"✅ Results saved to: models/results.json")

    return sentiment_pipeline


if __name__ == "__main__":
    evaluate_model()