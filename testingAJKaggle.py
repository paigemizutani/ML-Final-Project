import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from datasets import Dataset
from sklearn.metrics import accuracy_score

# --- NEW: Import kagglehub to load Kaggle datasets ---
import kagglehub
from kagglehub import KaggleDatasetAdapter

# CONFIG
NUM_EPOCHS = 2
MAX_LEN = 128
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "prajjwal1/bert-tiny"

# --- 0. OPTIONAL: LOAD KAGGLE DATASET ---
file_path = ""  # leave empty if single CSV
df_kaggle = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "minhaozhang1/reddit-mbti-dataset",
    file_path
)

# Sample a smaller subset for safe testing
df_kaggle = df_kaggle.sample(n=500, random_state=42)
print("Kaggle dataset shape (sampled):", df_kaggle.shape)
print("First 5 rows:", df_kaggle.head())

# Add MBTI dimension columns
df_kaggle["E_I"] = df_kaggle["type"].apply(lambda x: 0 if x[0] == "I" else 1)
df_kaggle["S_N"] = df_kaggle["type"].apply(lambda x: 0 if x[1] == "S" else 1)
df_kaggle["T_F"] = df_kaggle["type"].apply(lambda x: 0 if x[2] == "T" else 1)
df_kaggle["J_P"] = df_kaggle["type"].apply(lambda x: 0 if x[3] == "J" else 1)

# --- 1. TOY DATA (OR SWITCH TO KAGGLE DATA) ---
use_kaggle = True  # Change to False to use toy data

if use_kaggle:
    df = df_kaggle
else:
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
    df["E_I"] = df["type"].apply(lambda x: 0 if x[0] == "I" else 1)
    df["S_N"] = df["type"].apply(lambda x: 0 if x[1] == "S" else 1)
    df["T_F"] = df["type"].apply(lambda x: 0 if x[2] == "T" else 1)
    df["J_P"] = df["type"].apply(lambda x: 0 if x[3] == "J" else 1)

# 2. TOKENIZE
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
dataset = Dataset.from_pandas(df)

def tokenize(batch):
    return tokenizer(batch["posts"], padding="max_length", truncation=True, max_length=MAX_LEN)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "E_I", "S_N", "T_F", "J_P"])

# 3. TRAIN/TEST SPLIT
train_test = dataset.train_test_split(test_size=0.2)
train_loader = DataLoader(train_test['train'], batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(train_test['test'], batch_size=BATCH_SIZE)

# 4. TRAIN & PREDICT IN-MEMORY
dimensions = ["E_I", "S_N", "T_F", "J_P"]
dim_map = {"E_I": ["I","E"], "S_N": ["S","N"], "T_F": ["T","F"], "J_P": ["J","P"]}
trained_models = {}

for dim in dimensions:
    print(f"\nTraining for {dim}...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=5e-5)

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

    # Keep model in memory
    trained_models[dim] = model

# 5. PREDICT NEW POST
new_post = "I don't like people talking too much"
encoded = tokenizer(new_post, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
input_ids = encoded["input_ids"].to(DEVICE)
attention_mask = encoded["attention_mask"].to(DEVICE)

predicted_mbtis = {}
for dim, model in trained_models.items():
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        predicted_mbtis[dim] = dim_map[dim][prediction]

predicted_type = "".join([predicted_mbtis[dim] for dim in dimensions])
print(f"Predicted MBTI for the new post: {predicted_type}")
