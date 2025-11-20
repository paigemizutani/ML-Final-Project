# ----------------------------------------------------------
# model.py — BERT for MBTI Personality Classification
# Supports BOTH Toy Dataset and Kaggle Dataset
# ----------------------------------------------------------

# REQUIREMENTS:
# pip install kagglehub[pandas-datasets]
# pip install transformers datasets torch scikit-learn

import os
import pandas as pd
import kagglehub
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch


# ==========================================================
# 0. CONFIGURATION SWITCH
# ==========================================================
USE_TOY_DATA = True   # ← Toggle to False to use Kaggle dataset


# ==========================================================
# 1. TOY DATASET (FAST DEVELOPMENT)
# ==========================================================
def load_toy_dataset():
    print("\n⚡ Using TOY DATASET (fast prototyping)...")

    data = {
        "type": ["INTJ", "ENFP", "ISTP", "INFP", "ENTJ", "INTP"],
        "posts": [
            "I love thinking about abstract theories.",
            "I enjoy meeting new people and being spontaneous!",
            "I prefer hands-on problem solving.",
            "I write poetry and reflect deeply about life.",
            "I like leading teams and making decisions.",
            "I analyze systems logically for fun."
        ]
    }

    df = pd.DataFrame(data)
    print(df)
    return df


# ==========================================================
# 2. KAGGLE DATASET LOADING
# ==========================================================
def load_kaggle_dataset():
    print("\n⬇️ Downloading dataset from Kaggle (requires API auth)...")

    dataset_handle = "minhaozhang1/reddit-mbti-dataset"

    dataset_path = kagglehub.dataset_download(dataset_handle)
    print("Dataset downloaded to:", dataset_path)

    # List files
    files = os.listdir(dataset_path)
    print("Files:", files)

    # Find CSV file
    csv_files = [f for f in files if f.endswith(".csv")]
    if not csv_files:
        raise ValueError("❌ No CSV file found in Kaggle dataset!")
    csv_filename = csv_files[0]

    print("Loading:", csv_filename)

    df = kagglehub.load_dataset(dataset_handle, path=csv_filename)

    print("\nSample data:")
    print(df.head())
    return df


# ==========================================================
# 3. MAIN LOADING LOGIC
# ==========================================================
if USE_TOY_DATA:
    df = load_toy_dataset()
else:
    df = load_kaggle_dataset()


# ==========================================================
# 4. PREPROCESSING
# ==========================================================
expected_columns = ["type", "posts"]
if not all(col in df.columns for col in expected_columns):
    raise ValueError(f"Dataset must contain: {expected_columns}")

# Encode MBTI types → integers 0–15
le = LabelEncoder()
df["label"] = le.fit_transform(df["type"])

# Train-test split
train_texts, test_texts, y_train, y_test = train_test_split(
    df["posts"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42
)

print("\n📊 Data split:")
print("Train samples:", len(train_texts))
print("Test samples:", len(test_texts))


# ==========================================================
# 5. TOKENIZATION (BERT)
# ==========================================================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Create HuggingFace datasets
train_ds = Dataset.from_dict({"text": train_texts, "label": y_train})
test_ds = Dataset.from_dict({"text": test_texts, "label": y_test})

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds = train_ds.remove_columns(["text"])
test_ds = test_ds.remove_columns(["text"])

train_ds.set_format("torch")
test_ds.set_format("torch")


# ==========================================================
# 6. BERT MODEL SETUP
# ==========================================================
num_classes = len(df["type"].unique())

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_classes
)


# ==========================================================
# 7. TRAINING CONFIG
# ==========================================================
training_args = TrainingArguments(
    output_dir="./bert-mbti",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4 if USE_TOY_DATA else 8,
    per_device_eval_batch_size=4 if USE_TOY_DATA else 8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

print("\n🚀 Starting training...")
trainer.train()


# ==========================================================
# 8. EVALUATION
# ==========================================================
print("\n📈 Evaluation Results:")
print(trainer.evaluate())


# ==========================================================
# 9. SAVE MODEL
# ==========================================================
model_dir = "./bert-mbti-model"
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print(f"\n💾 Model saved to: {model_dir}")


# ==========================================================
# 10. PREDICTION FUNCTION
# ==========================================================
def predict_mbti(text):
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return le.inverse_transform([pred])[0]


# Example
example = "I love brainstorming and exploring new ideas."
print("\nExample prediction:")
print("Text:", example)
print("Predicted MBTI:", predict_mbti(example))
