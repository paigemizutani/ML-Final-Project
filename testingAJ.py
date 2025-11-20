# ==========================================================
# MBTI BERT — Minimal Pipeline for Mac
# ==========================================================
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from datasets import Dataset
from sklearn.metrics import accuracy_score

# ==========================================================
# CONFIG
# ==========================================================
NUM_EPOCHS = 2
MAX_LEN = 128
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "prajjwal1/bert-tiny"  # small and fast for Mac

# ==========================================================
# 1. TOY DATA
# ==========================================================
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

# Split MBTI into four dimensions
df["E_I"] = df["type"].apply(lambda x: 0 if x[0] == "I" else 1)
df["S_N"] = df["type"].apply(lambda x: 0 if x[1] == "S" else 1)
df["T_F"] = df["type"].apply(lambda x: 0 if x[2] == "T" else 1)
df["J_P"] = df["type"].apply(lambda x: 0 if x[3] == "J" else 1)

# ==========================================================
# 2. TOKENIZE
# ==========================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
dataset = Dataset.from_pandas(df)

def tokenize(batch):
    return tokenizer(batch["posts"], padding="max_length", truncation=True, max_length=MAX_LEN)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "E_I", "S_N", "T_F", "J_P"])

# ==========================================================
# 3. TRAIN/TEST SPLIT
# ==========================================================
train_test = dataset.train_test_split(test_size=0.2)
train_loader = DataLoader(train_test['train'], batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(train_test['test'], batch_size=BATCH_SIZE)

# ==========================================================
# 4. TRAIN & EVALUATE EACH DIMENSION
# ==========================================================
dimensions = ["E_I", "S_N", "T_F", "J_P"]

for dim in dimensions:
    print(f"\nTraining for {dim}...")

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch[dim].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} finished, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch[dim].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    print(f"Accuracy for {dim}: {acc:.2f}")
