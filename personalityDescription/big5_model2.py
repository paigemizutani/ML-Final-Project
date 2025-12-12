import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset, DatasetDict
import torch.nn as nn
from collections import Counter
import seaborn as sns

# -------------------------
# LOAD & SAMPLE DATA
# -------------------------
dataset = load_dataset("jingjietan/pandora-big5")

total_size = sum(len(dataset[s]) for s in ["train", "validation", "test"])
TARGET_TOTAL = 300000 

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
NUM_EPOCHS = 10
MAX_LEN = 128
BATCH_SIZE = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-uncased"

PATIENCE_ES = 2
LR = 4e-5  
dimensions = ['O', 'C', 'E', 'A', 'N']

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
# MODEL: PsychBERT + MLP Regression Head
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
# TRAINING LOOP
# -------------------------
best_val_loss = float("inf")
epochs_no_improve = 0
train_losses, val_losses, val_r2s = [], [], []

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = (
            input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
        )
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
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask, labels = (
                input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
            )
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
            all_true.append(labels.cpu().numpy())
            all_pred.append(logits.cpu().numpy())
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    val_r2 = r2_score(y_true, y_pred)
    val_r2s.append(val_r2)
    print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | R2: {val_r2:.3f}")

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

# Load best
model.load_state_dict(torch.load("best_psychbert_big5.pt"))
model.eval()

# -------------------------
# TEST EVALUATION
# -------------------------
all_true, all_pred = [], []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids, attention_mask, labels = (
            input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
        )
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        all_true.append(labels.cpu().numpy())
        all_pred.append(logits.cpu().numpy())

y_true = scaler.inverse_transform(np.vstack(all_true))
y_pred = scaler.inverse_transform(np.vstack(all_pred))

print("\nBig Five Regression Results:")
metrics = []
for i, dim in enumerate(dimensions):
    mse = mean_squared_error(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    r2  = r2_score(y_true[:, i], y_pred[:, i])
    print(f"{dim}: MSE={mse:.4f} | MAE={mae:.4f} | R2={r2:.3f}")
    metrics.append([dim, mse, mae, r2])

metrics_df = pd.DataFrame(metrics, columns=['Trait', 'MSE', 'MAE', 'R²'])
metrics_df.to_csv("big5_regression_metrics.csv", index=False)

# -------------------------
# Scatter plots: True vs Predicted
# -------------------------
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=False, sharey=False)
for i, dim in enumerate(dimensions):
    ax = axes[i]
    ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, color='teal', edgecolor='k', s=20)
    ax.plot([y_true[:, i].min(), y_true[:, i].max()],
            [y_true[:, i].min(), y_true[:, i].max()],
            'r--', linewidth=2)
    ax.set_title(f"{dim} (R²={metrics_df['R²'][i]:.2f})", fontsize=12)
    ax.set_xlabel("True", fontsize=10)
    ax.set_ylabel("Predicted", fontsize=10)
    ax.grid(alpha=0.3)
plt.suptitle("PsychBERT Big Five Regression: True vs Predicted", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("big5_scatter_plot.png", dpi=300)
plt.show()

# -------------------------
# Training curves + R² bar chart
# -------------------------
plt.figure(figsize=(16,6))
plt.subplot(1,3,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training / Validation Loss"); plt.legend(); plt.grid(alpha=0.3)

plt.subplot(1,3,2)
plt.plot(val_r2s, marker='o', label='Val R²')
plt.xlabel("Epoch"); plt.ylabel("R²")
plt.title("Validation R²"); plt.grid(alpha=0.3)

plt.subplot(1,3,3)
plt.bar(dimensions, metrics_df['R²'], color='teal')
plt.xlabel("Trait"); plt.ylabel("R²"); plt.title("R² by Trait"); plt.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("psychbert_training_and_metrics.png", dpi=300)
plt.show()


# -------------------------
# Top 15 4-word phrases for 20th / 80th percentile per trait (no nltk)
# -------------------------
for dim in dimensions:
    low_thresh = train_df[dim].quantile(0.2)
    high_thresh = train_df[dim].quantile(0.8)
    low_texts = train_df[train_df[dim] <= low_thresh]['text']
    high_texts = train_df[train_df[dim] >= high_thresh]['text']

    for label, texts in zip(['20th Percentile', '80th Percentile'], [low_texts, high_texts]):
        words = " ".join(texts.astype(str)).lower().split()
        fourgrams = [" ".join(words[i:i+4]) for i in range(len(words)-3)]
        common_phrases = Counter(fourgrams).most_common(15)

        df_phrases = pd.DataFrame(common_phrases, columns=['Phrase','Count'])
        print(f"\nTop 15 4-word phrases for {dim} ({label}):")
        print(df_phrases)

        plt.figure(figsize=(8,4))
        sns.barplot(x='Count', y='Phrase', data=df_phrases, color='teal')
        plt.title(f"{dim} {label} - Top 15 4-word Phrases")
        plt.tight_layout()
        plt.show()


# -------------------------
# Predict new posts
# -------------------------
def predict_new_posts(texts):
    enc = tokenizer(
        [str(t) for t in texts],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)

    with torch.no_grad():
        preds = model(input_ids=input_ids, attention_mask=attention_mask)  # tensor directly
        preds = scaler.inverse_transform(preds.cpu().numpy())

    return pd.DataFrame(preds, columns=dimensions)


# Example usage:
new_posts = [
    "Ok ya not sure if this model is actually predicting anything",
    "I love leading teams and planning long-term goals.",
    "I really enjoy spending time alone thinking about ideas.",
    "Whatever this is sooo annoying",
    "Wait I'm actually so excited for this",
    "I love spending time with strangers",
    "I love organizing plans and thinking ahead.",
    "I get stressed easily and worry about small things.",
]

new_preds = predict_new_posts(new_posts)

for post, p in zip(new_posts, new_preds.values):
    print(f"\nPost: {post}")
    print(
        f"O={p[0]:.2f}, "
        f"C={p[1]:.2f}, "
        f"E={p[2]:.2f}, "
        f"A={p[3]:.2f}, "
        f"N={p[4]:.2f}"
    )
# -------------------------
# Compute relative percentiles
# -------------------------
def get_relative_scores(post_pred, all_preds_df):
    relative_scores = {}
    for dim in dimensions:
        all_scores = all_preds_df[dim].values
        relative_scores[dim] = (all_scores < post_pred[dim]).mean()  # fraction below
    return relative_scores

# Example: assume y_pred is all dataset predictions (shape [n_posts, 5])
all_preds_df = pd.DataFrame(y_pred, columns=dimensions)  # dataset predictions

print("\nOriginal Predictions:")
for post, p in zip(new_posts, new_preds.values):
    print(f"\nPost: {post}")
    print(
        f"O={p[0]:.2f}, "
        f"C={p[1]:.2f}, "
        f"E={p[2]:.2f}, "
        f"A={p[3]:.2f}, "
        f"N={p[4]:.2f}"
    )

print("\n" + "-"*60 + "\nRelative Percentile Predictions:")
for post, p in zip(new_posts, new_preds.values):
    relative = get_relative_scores(pd.Series(p, index=dimensions), all_preds_df)
    print(f"\nPost: {post}")
    print(
        f"O={relative['O']:.2f}, "
        f"C={relative['C']:.2f}, "
        f"E={relative['E']:.2f}, "
        f"A={relative['A']:.2f}, "
        f"N={relative['N']:.2f}"
    )

