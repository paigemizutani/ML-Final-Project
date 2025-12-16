import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib
matplotlib.use("Agg")  # force non-GUI backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset, DatasetDict
import torch.nn as nn
import seaborn as sns

# -------------------------
# LOAD & SAMPLE DATA
# -------------------------
dataset = load_dataset("jingjietan/pandora-big5")
dimensions = ['O', 'C', 'E', 'A', 'N']

total_size = sum(len(dataset[s]) for s in ["train", "validation", "test"])
TARGET_TOTAL = 150000

split_sizes = {split: int(len(dataset[split]) / total_size * TARGET_TOTAL)
               for split in ["train", "validation", "test"]}

sampled_dataset = DatasetDict({
    split: dataset[split].shuffle(seed=42).select(range(split_sizes[split]))
    for split in ["train", "validation", "test"]
})

for split in ["train", "validation", "test"]:
    sampled_dataset[split].to_pandas().to_csv(f"big5_{split}.csv", index=False)

# -------------------------
# CONFIG
# -------------------------
NUM_EPOCHS = 10
MAX_LEN = 128
BATCH_SIZE = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-uncased"
PATIENCE_ES = 2
LR = 4e-5  

# -------------------------
# LOAD DATA
# -------------------------
train_df = pd.read_csv("big5_train.csv")
val_df   = pd.read_csv("big5_validation.csv")
test_df  = pd.read_csv("big5_test.csv")

for dim in dimensions:
    train_df[dim] = train_df[dim].astype(float)
    val_df[dim]   = val_df[dim].astype(float)
    test_df[dim]  = test_df[dim].astype(float)

# -------------------------
# NORMALIZE TARGETS
# -------------------------
scaler = MinMaxScaler()
train_df[dimensions] = scaler.fit_transform(train_df[dimensions])
val_df[dimensions]   = scaler.transform(val_df[dimensions])
test_df[dimensions]  = scaler.transform(test_df[dimensions])

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def encode(texts):
    texts = texts.astype(str)
    return tokenizer(texts.tolist(), padding="max_length", truncation=True,
                     max_length=MAX_LEN, return_tensors="pt")

train_enc = encode(train_df['text'])
val_enc   = encode(val_df['text'])
test_enc  = encode(test_df['text'])

# -------------------------
# DATASETS & LOADERS
# -------------------------
train_dataset = TensorDataset(
    train_enc['input_ids'], train_enc['attention_mask'],
    torch.tensor(train_df[dimensions].values, dtype=torch.float)
)
val_dataset = TensorDataset(
    val_enc['input_ids'], val_enc['attention_mask'],
    torch.tensor(val_df[dimensions].values, dtype=torch.float)
)
test_dataset = TensorDataset(
    test_enc['input_ids'], test_enc['attention_mask'],
    torch.tensor(test_df[dimensions].values, dtype=torch.float)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# -------------------------
# MODEL
# -------------------------
class PsychBERT_Regressor(nn.Module):
    def __init__(self, model_name, num_labels=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.regressor(pooled)

model = PsychBERT_Regressor(MODEL_NAME).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = nn.SmoothL1Loss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

# -------------------------
# TRAINING LOOP (per-trait tracking)
# -------------------------
best_val_loss = float("inf")
epochs_no_improve = 0
train_losses, val_losses, val_r2s = [], [], []
val_losses_per_trait = []  # store per-trait validation loss
val_r2s_per_trait = []     # store per-trait R²

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    all_true, all_pred = [], []
    per_trait_loss = np.zeros(len(dimensions))
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
            all_true.append(labels.cpu().numpy())
            all_pred.append(logits.cpu().numpy())
            per_trait_loss += np.mean((logits - labels)**2, axis=0)  # per-trait MSE

    val_loss /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    per_trait_loss /= len(val_loader)
    val_losses_per_trait.append(per_trait_loss)

    y_true_val = np.vstack(all_true)
    y_pred_val = np.vstack(all_pred)
    r2_vals = r2_score(y_true_val, y_pred_val, multioutput='raw_values')  # per-trait R²
    val_r2s_per_trait.append(r2_vals)
    val_r2s.append(np.mean(r2_vals))

    print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Mean R2: {np.mean(r2_vals):.3f}")

    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "best_psychbert_big5.pt")
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE_ES:
            print("Early stopping triggered.")
            break

# -------------------------
# LOAD BEST MODEL
# -------------------------
model.load_state_dict(torch.load("best_psychbert_big5.pt"))
model.eval()

# -------------------------
# TEST EVALUATION
# -------------------------
all_true, all_pred = [], []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        all_true.append(labels.cpu().numpy())
        all_pred.append(logits.cpu().numpy())

y_true = scaler.inverse_transform(np.vstack(all_true))
y_pred = scaler.inverse_transform(np.vstack(all_pred))

# -------------------------
# METRICS
# -------------------------
metrics = []
print("\nBig Five Regression Metrics:")
for i, dim in enumerate(dimensions):
    mse = mean_squared_error(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    r2  = r2_score(y_true[:, i], y_pred[:, i])
    metrics.append([dim, mse, mae, r2])
    print(f"{dim}: MSE={mse:.3f} | MAE={mae:.3f} | R2={r2:.3f}")

metrics_df = pd.DataFrame(metrics, columns=['Trait','MSE','MAE','R2'])
metrics_df.to_csv("big5_regression_metrics.csv", index=False)

# -------------------------
# Scatter plots per trait
# -------------------------
for i, dim in enumerate(dimensions):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true[:,i], y_pred[:,i], alpha=0.5, color='teal', edgecolor='k', s=20)
    plt.plot([0,100],[0,100],'r--', linewidth=2)
    plt.title(f"{dim} (R²={metrics_df['R2'][i]:.2f})")
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"scatter_{dim}.png")
    plt.close()

# -------------------------
# Training Curve (Loss + R² per trait)
# -------------------------
plt.figure(figsize=(12,6))
epochs = range(1, len(val_losses_per_trait)+1)
colors = ['C0','C1','C2','C3','C4']

# Plot per-trait validation loss
for i, dim in enumerate(dimensions):
    plt.plot(epochs, [v[i] for v in val_losses_per_trait], label=f"{dim} Loss", color=colors[i], linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss per Big Five Trait")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("val_loss_per_trait.png", dpi=300)
plt.close()

# Optional: per-trait R² plot
plt.figure(figsize=(12,6))
for i, dim in enumerate(dimensions):
    plt.plot(epochs, [v[i] for v in val_r2s_per_trait], label=f"{dim} R²", color=colors[i], linestyle='--', linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Validation R²")
plt.title("Validation R² per Big Five Trait")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("val_r2_per_trait.png", dpi=300)
plt.close()
