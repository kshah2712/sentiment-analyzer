# 💬 Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)
![BERT](https://img.shields.io/badge/BERT-Fine--tuned-FF6B6B)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A production-style NLP pipeline that performs sentiment analysis using a fine-tuned BERT model from HuggingFace. Covers transformer architecture, tokenization, fine-tuning on real review data, and serving predictions via a modern FastAPI REST API — all containerized with Docker.

> **Purpose:** Demonstrate NLP fundamentals and modern transformer-based approach — tokenization, BERT fine-tuning, HuggingFace ecosystem, and building a production NLP API.

---

## 📌 What This Project Covers

| Concept | Implementation |
|---|---|
| NLP Fundamentals | Tokenization, text preprocessing, embeddings |
| Transformers | BERT architecture and attention mechanism |
| Fine-tuning | Transfer learning on sentiment dataset |
| HuggingFace | Transformers, Datasets, Trainer API |
| Model Serving | FastAPI with batch prediction support |
| Containerization | Dockerfile + docker-compose |
| Testing | pytest with FastAPI test client |

---

## 🗂️ Project Structure

```
sentiment-analyzer/
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Tokenized data
├── notebooks/
│   └── 01_sentiment_eda.ipynb
├── src/
│   ├── download_data.py      # Download IMDB dataset
│   ├── preprocess.py         # Text cleaning + tokenization
│   └── train.py              # Fine-tune BERT model
├── models/                   # Saved model files
├── api/
│   └── app.py                # FastAPI REST API
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/kshah2712/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2. Create virtual environment

```bash
conda create -n sentiment python=3.11 -y
conda activate sentiment
pip install -r requirements.txt
```

### 3. Download dataset

```bash
python src/download_data.py
```

### 4. Train the model

```bash
python src/train.py
```

### 5. Run FastAPI locally

```bash
uvicorn api.app:app --reload --port 8000
```

### 6. Run with Docker

```bash
docker-compose up --build
```

---

## 🚀 API Usage

### Health check
```bash
GET /health
```

### Single prediction
```bash
POST /predict
{
  "text": "This movie was absolutely amazing! I loved every minute of it."
}
```

**Response:**
```json
{
  "text": "This movie was absolutely amazing!...",
  "sentiment": "POSITIVE",
  "confidence": 0.9987,
  "scores": {
    "positive": 0.9987,
    "negative": 0.0013
  }
}
```

### Batch prediction
```bash
POST /predict/batch
{
  "texts": [
    "Great product, highly recommend!",
    "Terrible experience, waste of money."
  ]
}
```

---

## 📊 Model Results

| Metric | Score |
|---|---|
| Accuracy | ~93% |
| F1 Score | ~93% |
| Model | distilbert-base-uncased-finetuned-sst-2-english |

---

## 🧠 Key Concepts

### Why BERT over traditional ML?
Traditional ML (TF-IDF + Logistic Regression) treats words as independent tokens. BERT understands **context** — "not good" vs "good" are treated completely differently because BERT reads the full sentence bidirectionally.

### Transfer Learning
Instead of training from scratch (needs millions of examples), we use a BERT model **pre-trained on billions of words** and fine-tune it on our specific task. This gives state-of-the-art results with minimal data and compute.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **NLP:** HuggingFace Transformers, BERT/DistilBERT
- **Deep Learning:** PyTorch
- **API:** FastAPI + Uvicorn
- **Containerization:** Docker, Docker Compose
- **Testing:** pytest

---

## 📚 Key Learnings

- **Tokenization** — how BERT converts text to numbers
- **Attention mechanism** — how transformers understand context
- **Transfer learning** — fine-tuning pretrained models
- **HuggingFace pipeline** — industry standard for NLP
- **Batch prediction** — serving multiple inputs efficiently

---

## 🗺️ Part of ML Learning Roadmap

This is **Project 5 of 10** in a progressive ML + GenAI portfolio:

| # | Project | Skills |
|---|---|---|
| ✅ 1 | Classic ML Pipeline | EDA, Sklearn, Flask, Docker |
| ✅ 2 | House Price Predictor | Regression, Feature Eng., Streamlit |
| ✅ 3 | Customer Churn Classifier | SHAP, FastAPI, Imbalanced data |
| ✅ 5 | Sentiment Analyzer (this project) | HuggingFace, BERT, NLP, Transformers |
| 8 | RAG Chatbot | LangChain, Vector DB, GenAI |
| ... | ... | ... |

---

## 👤 Author

**Kashyap Shah**
[GitHub](https://github.com/kshah2712) 
