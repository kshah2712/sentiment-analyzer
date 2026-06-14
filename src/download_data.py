import pandas as pd
import os
from datasets import load_dataset

os.makedirs("data/raw", exist_ok=True)

print("=== DOWNLOADING IMDB SENTIMENT DATASET ===")
print("Loading from HuggingFace datasets...")

# load IMDB dataset from HuggingFace
dataset = load_dataset("stanfordnlp/imdb")

print(f"\nDataset structure: {dataset}")
print(f"\nTrain samples: {len(dataset['train'])}")
print(f"Test samples:  {len(dataset['test'])}")

# convert to pandas
train_df = pd.DataFrame(dataset["train"])
test_df  = pd.DataFrame(dataset["test"])

print(f"\nColumns: {list(train_df.columns)}")
print(f"\nLabel distribution (train):\n{train_df['label'].value_counts()}")
print(f"\nSample review:\n{train_df['text'][0][:300]}...")

# save small subset for faster training
# 0 = negative, 1 = positive
train_small = train_df.groupby("label").head(2000)  # 2000 per class = 4000 total
test_small  = test_df.groupby("label").head(500)    # 500 per class = 1000 total

train_small.to_csv("data/raw/train.csv", index=False)
test_small.to_csv("data/raw/test.csv", index=False)

print(f"\n✅ Train subset saved: {len(train_small)} samples")
print(f"✅ Test subset saved:  {len(test_small)} samples")
print("\nDone! Check data/raw/ folder")