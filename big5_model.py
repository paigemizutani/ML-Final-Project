import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from datasets import load_dataset, DatasetDict

dataset = load_dataset("jingjietan/pandora-big5")

total_size = sum(len(dataset[s]) for s in ["train", "validation", "test"])
TARGET_TOTAL = 368000

split_sizes = {
    split: int(len(dataset[split]) / total_size * TARGET_TOTAL)
    for split in ["train", "validation", "test"]
}

sampled_dataset = DatasetDict({
    split: dataset[split].shuffle(seed=42).select(
        range(split_sizes[split])
    )
    for split in ["train", "validation", "test"]
})

for split in ["train", "validation", "test"]:
    sampled_dataset[split].to_pandas().to_csv(f"big5_{split}.csv", index=False)


# -------------------------
# CONFIG
# -------------------------
NUM_EPOCHS = 7
MAX_LEN = 128
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-uncased"
PATIENCE_ES = 2
LR = 1e-5

dimensions = ['O', 'C', 'E', 'A', 'N']

# -------------------------
# LOAD DATA
# -------------------------
train_df = pd.read_csv("big5_train.csv")
val_df   = pd.read_csv("big5_validation.csv")
test_df  = pd.read_csv("big5_test.csv")

dimensions = ['O', 'C', 'E', 'A', 'N']

for dim in dimensions:
    train_df[dim] = train_df[dim].astype(float)
    val_df[dim]   = val_df[dim].astype(float)
    test_df[dim]  = test_df[dim].astype(float)


# -------------------------
# TOKENIZER
# -------------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def encode(texts):
    texts = texts.astype(str)   # FORCE strings
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

train_enc = encode(train_df['text'])
val_enc   = encode(val_df['text'])
test_enc  = encode(test_df['text'])

# -------------------------
# DATASETS & LOADERS
# -------------------------
train_dataset = TensorDataset(
    train_enc['input_ids'],
    train_enc['attention_mask'],
    torch.tensor(train_df[dimensions].values, dtype=torch.float)
)

val_dataset = TensorDataset(
    val_enc['input_ids'],
    val_enc['attention_mask'],
    torch.tensor(val_df[dimensions].values, dtype=torch.float)
)

test_dataset = TensorDataset(
    test_enc['input_ids'],
    test_enc['attention_mask'],
    torch.tensor(test_df[dimensions].values, dtype=torch.float)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# -------------------------
# MODEL (REGRESSION)
# -------------------------
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=5,
    problem_type="regression"
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=1,
    verbose=True
)

# -------------------------
# TRAINING LOOP
# -------------------------
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for input_ids, attention_mask, labels in train_loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1} | "
        f"Train MSE: {train_loss:.4f} | "
        f"Val MSE: {val_loss:.4f}"
    )

    scheduler.step(val_loss)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_big5_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE_ES:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# -------------------------
# LOAD BEST MODEL
# -------------------------
model.load_state_dict(
    torch.load("best_big5_model.pt", weights_only=True)
)
model.eval()

# -------------------------
# EVALUATION
# -------------------------
all_true, all_pred = [], []

with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_true.append(labels.cpu().numpy())
        all_pred.append(outputs.logits.cpu().numpy())

y_true = np.vstack(all_true)
y_pred = np.vstack(all_pred)

print("\nBig Five Regression Results:")
for i, dim in enumerate(dimensions):
    mse = mean_squared_error(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    r, _ = pearsonr(y_true[:, i], y_pred[:, i])

    print(f"{dim}: MSE={mse:.4f}, MAE={mae:.4f}, r={r:.3f}")

# -------------------------
# PREDICT NEW TEXT
# -------------------------
new_posts = [
    "Ok ya not sure if this model is actually predicting anything",
    "I love leading teams and planning long-term goals.",
    "I really enjoy spending time alone thinking about ideas.",
    "whatever this is sooo annoying",
    "Wait Im actually so excited for thiss",
    "I love spending time with strangers",
    "I love organizing plans and thinking ahead.",
    "I get stressed easily and worry about small things.",
]

encoded = tokenizer(
    new_posts,
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)

input_ids = encoded["input_ids"].to(DEVICE)
attention_mask = encoded["attention_mask"].to(DEVICE)

with torch.no_grad():
    preds = model(input_ids=input_ids, attention_mask=attention_mask).logits

for post, p in zip(new_posts, preds.cpu().numpy()):
    print(f"\nPost: {post}")
    print(
        f"O={p[0]:.2f}, "
        f"C={p[1]:.2f}, "
        f"E={p[2]:.2f}, "
        f"A={p[3]:.2f}, "
        f"N={p[4]:.2f}"
    )

