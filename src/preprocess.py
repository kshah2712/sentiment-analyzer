import pandas as pd
import numpy as np
import os
import re
from transformers import AutoTokenizer

# ── Config ─────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


def clean_text(text: str) -> str:
    """
    Basic text cleaning —
    remove HTML tags and extra whitespace
    """
    # remove HTML tags like <br />, <p>, etc.
    text = re.sub(r"<[^>]+>", " ", text)

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_data(train_path="data/raw/train.csv",
              test_path="data/raw/test.csv"):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print("=== LOADING DATASET ===")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"\nLabel distribution (train):\n{train_df['label'].value_counts()}")
    print(f"\nSample text:\n{train_df['text'][0][:200]}...")

    return train_df, test_df


def preprocess(train_path="data/raw/train.csv",
               test_path="data/raw/test.csv"):

    train_df, test_df = load_data(train_path, test_path)

    # clean text
    print("\n=== CLEANING TEXT ===")
    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"]  = test_df["text"].apply(clean_text)
    print("HTML tags removed, whitespace cleaned")

    # load tokenizer
    print(f"\n=== LOADING TOKENIZER ===")
    print(f"Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # sample tokenization to show how it works
    sample = train_df["text"][0][:100]
    tokens = tokenizer.tokenize(sample)
    print(f"\nSample text: '{sample}'")
    print(f"Tokenized:   {tokens}")
    print(f"Token IDs:   {tokenizer.convert_tokens_to_ids(tokens)}")

    # tokenize full dataset
    print(f"\n=== TOKENIZING DATASET ===")
    print(f"Max sequence length: {MAX_LENGTH}")

    train_encodings = tokenizer(
        list(train_df["text"]),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    test_encodings = tokenizer(
        list(test_df["text"]),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    train_labels = list(train_df["label"])
    test_labels  = list(test_df["label"])

    print(f"Train input_ids shape: {train_encodings['input_ids'].shape}")
    print(f"Test input_ids shape:  {test_encodings['input_ids'].shape}")

    return train_encodings, test_encodings, train_labels, test_labels, tokenizer


if __name__ == "__main__":
    preprocess()