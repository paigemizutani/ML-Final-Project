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
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize

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
# TRAINING LOOP
# -------------------------
best_val_loss = float("inf")
epochs_no_improve = 0
train_losses, val_losses, val_r2s = [], [], []

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
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
            all_true.append(labels.cpu().numpy())
            all_pred.append(logits.cpu().numpy())
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    y_true_val = np.vstack(all_true)
    y_pred_val = np.vstack(all_pred)
    val_r2 = r2_score(y_true_val, y_pred_val)
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
# Scatter plots saved
# -------------------------
for i, dim in enumerate(dimensions):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true[:,i], y_pred[:,i], alpha=0.5, color='teal', edgecolor='k', s=20)
    plt.plot([0,100],[0,100],'r--', linewidth=2)
    plt.title(f"{dim} (R²={metrics_df['R2'][i]:.2f})")
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.tight_layout()
   # plt.savefig(f"scatter_{dim}.png")
    plt.close()

# -------------------------
# SAFE BINNING (10 bins, 0-9)
# -------------------------
def bin_score_10_safe(x):
    x = np.clip(x, 0, 100)
    return min(int(x // 10), 9)

binned_true = np.vectorize(bin_score_10_safe)(y_true)
binned_pred = np.vectorize(bin_score_10_safe)(y_pred)

bin_metrics = []
for i, dim in enumerate(dimensions):
    heatmap_df = pd.DataFrame(0, index=range(10), columns=range(10))
    for t,p in zip(binned_true[:,i], binned_pred[:,i]):
        heatmap_df.loc[t,p] += 1

    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='YlGnBu')
    plt.xlabel("Predicted Bin")
    plt.ylabel("True Bin")
    plt.title(f"{dim} - True vs Predicted (10x10 bins)")
    plt.tight_layout()
    plt.savefig(f"heatmap_{dim}.png")
    plt.close()

    mse_bin = mean_squared_error(binned_true[:, i], binned_pred[:, i])
    mae_bin = mean_absolute_error(binned_true[:, i], binned_pred[:, i])
    r2_bin  = r2_score(binned_true[:, i], binned_pred[:, i])
    bin_metrics.append([dim, mse_bin, mae_bin, r2_bin])

bin_metrics_df = pd.DataFrame(bin_metrics, columns=['Trait','MSE_bin','MAE_bin','R2_bin'])
bin_metrics_df.to_csv("big5_regression_metrics_bins_10.csv", index=False)

# -------------------------
# Distribution plots saved
# -------------------------
for i, dim in enumerate(dimensions):
    plt.figure(figsize=(6,4))
    sns.histplot(y_true[:,i], bins=10, color='skyblue', kde=False, alpha=0.6, label='True')
    sns.histplot(y_pred[:,i], bins=10, color='salmon', kde=False, alpha=0.6, label='Predicted')
    plt.title(f"{dim} Distribution")
    plt.xlabel("Score"); plt.ylabel("Count"); plt.legend()
    plt.tight_layout()
   # plt.savefig(f"dist_{dim}.png")
    plt.close()

# -------------------------
# Pairwise predicted traits saved
# -------------------------
sns.pairplot(pd.DataFrame(y_pred, columns=dimensions), kind='scatter', corner=True, plot_kws={'alpha':0.3, 's':20})
plt.suptitle("Predicted OCEAN Trait Pairwise Plots", y=1.02)
#plt.savefig("pairwise_pred.png")
plt.close()





'''
# -------------------------
# COMBINED HEATMAP FIGURE
# -------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Create a 5x5 figure: 1 row, 5 columns
fig, axes = plt.subplots(1, 5, figsize=(25,5))

for i, dim in enumerate(dimensions):
    # Create 10x10 bin counts
    heatmap_df = pd.DataFrame(0, index=range(10), columns=range(10))
    for t, p in zip(binned_true[:, i], binned_pred[:, i]):
        heatmap_df.loc[t, p] += 1

    ax = axes[i]
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        cbar=(i==4),  # only show colorbar on last subplot
        ax=ax
    )
    ax.set_xlabel("Predicted Bin")
    ax.set_ylabel("True Bin")
    ax.set_title(dim)

plt.suptitle("Predicted vs True Big Five Traits (10 Bins)", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
plt.savefig("results_summary.png", dpi=300)
plt.show()
'''



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
    "Yes I completely agree with you! That is a great first step to take to be successful in the computer science department, please let me know if you need more help!",
    "I don’t think that approach will work. We need to be more logical and stick to the data.",
    "Wow, that’s such a cool idea! I never thought of it that way before.",
    "I’ve made a detailed plan for the project timeline. Everyone, please review it carefully.",
    "Honestly, I just go with my gut on things. Planning too much stresses me out.",
    "That’s really insightful. I think there’s a bigger pattern here if we look closely.",
    "I love brainstorming all the possibilities. No idea is too wild.",
    "Please make sure to follow the instructions exactly. Precision matters.",
    "I prefer to work alone on this task. Group work slows me down.",
    "Honestly, I don’t care about the rules as long as we have fun doing it.",
    "I’ve noticed a trend in the data that might help us predict outcomes more accurately.",
    "We should just try everything and see what works best."
]


new_preds = predict_new_posts(new_posts)

for post, p in zip(new_posts, new_preds.values):
    # Binary labels based on >=50 → High, <50 → Low
    O_label = "High" if p[0] >= 50 else "Low"
    C_label = "High" if p[1] >= 50 else "Low"
    E_label = "High" if p[2] >= 50 else "Low"
    A_label = "High" if p[3] >= 50 else "Low"
    N_label = "High" if p[4] >= 50 else "Low"

    print(f"\nPost: {post}")
    print(
        f"O={p[0]:.2f} ({O_label}), "
        f"C={p[1]:.2f} ({C_label}), "
        f"E={p[2]:.2f} ({E_label}), "
        f"A={p[3]:.2f} ({A_label}), "
        f"N={p[4]:.2f} ({N_label})"
    )





    # -------------------------
# Training Accuracy / Loss Curve
# -------------------------
plt.figure(figsize=(8,5))
epochs_range = range(1, len(train_losses)+1)

# Plot train & validation loss
plt.plot(epochs_range, train_losses, 'o-', color='teal', label='Train Loss')
plt.plot(epochs_range, val_losses, 'o-', color='salmon', label='Validation Loss')

# Plot validation R² on secondary axis
ax2 = plt.gca().twinx()
ax2.plot(epochs_range, val_r2s, 's--', color='purple', label='Validation R²')
ax2.set_ylabel("R²", color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

plt.title("Training / Validation Loss and Validation R² over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig("training_curve.png", dpi=300)
plt.close()



# -------------------------
# Plot per-trait loss and R² over epochs
# -------------------------
'''plt.figure(figsize=(10,6))

epochs = range(1, len(train_losses)+1)
colors = ['C0','C1','C2','C3','C4']

# Per-trait validation R² (we need to compute per-trait per epoch)
# Assuming you have stored per-trait R²s per epoch in val_r2s_per_trait list of dicts
# Example: val_r2s_per_trait[epoch_idx] = {'O':0.1,'C':0.2,...}

# For demonstration, we compute a simple proxy: overall R² per epoch (replace with per-trait if available)
for i, dim in enumerate(dimensions):
    plt.plot(epochs, [v[i] for v in val_losses], label=f"{dim} Loss", color=colors[i], linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss per Big Five Trait")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("val_loss_per_trait.png", dpi=300)
plt.close()

'''
# =========================================================
# OCEAN CLASSIFICATION-STYLE METRICS (MBTI-LIKE)
# =========================================================

print("\n==============================")
print("OCEAN ACCURACY & PR AUC SCORES")
print("==============================")

# ---------------------------------------------------------
# RAW SCORES → BINARY (LOW / HIGH)
# ---------------------------------------------------------
print("\n--- RAW OCEAN (Binary: Low vs High) ---")

raw_results = []

for i, dim in enumerate(dimensions):
    # Ground truth & predictions: High (>=50) vs Low (<50)
    y_true_bin = (y_true[:, i] >= 50).astype(int)
    y_pred_bin = (y_pred[:, i] >= 50).astype(int)

    acc = accuracy_score(y_true_bin, y_pred_bin)

    # PR AUC uses continuous confidence
    pr_auc = average_precision_score(
        y_true_bin,
        y_pred[:, i] / 100.0
    )

    raw_results.append([dim, acc, pr_auc])
    print(f"{dim}: Accuracy={acc:.4f} | PR AUC={pr_auc:.4f}")

raw_acc_df = pd.DataFrame(
    raw_results,
    columns=["Trait", "Accuracy_raw", "PR_AUC_raw"]
)
raw_acc_df.to_csv("big5_accuracy_raw.csv", index=False)

# ---------------------------------------------------------
# BINNED SCORES → 10-CLASS CLASSIFICATION
# ---------------------------------------------------------
print("\n--- BINNED OCEAN (10 Bins) ---")

bin_results = []
num_bins = 10
classes = list(range(num_bins))

for i, dim in enumerate(dimensions):
    y_true_b = binned_true[:, i]
    y_pred_b = binned_pred[:, i]

    # Exact-bin accuracy
    acc = accuracy_score(y_true_b, y_pred_b)

    # PR AUC (macro, one-vs-rest)
    y_true_oh = label_binarize(y_true_b, classes=classes)
    y_pred_oh = label_binarize(y_pred_b, classes=classes)

    pr_auc = average_precision_score(
        y_true_oh,
        y_pred_oh,
        average="macro"
    )

    bin_results.append([dim, acc, pr_auc])
    print(f"{dim}: Accuracy={acc:.4f} | PR AUC={pr_auc:.4f}")

bin_acc_df = pd.DataFrame(
    bin_results,
    columns=["Trait", "Accuracy_bin", "PR_AUC_bin"]
)
bin_acc_df.to_csv("big5_accuracy_binned.csv", index=False)








# =========================================================
# RESULTS FIGURE: OCEAN Accuracy & PR AUC
# =========================================================

# Load metric CSVs (already saved above)
raw_df = pd.read_csv("big5_accuracy_raw.csv")
bin_df = pd.read_csv("big5_accuracy_binned.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

traits = raw_df["Trait"]
x = np.arange(len(traits))
width = 0.35

import matplotlib.pyplot as plt
import numpy as np

# Merge metrics into a single DataFrame for plotting
plot_df = pd.DataFrame({
    "Trait": dimensions,
    "Binary Accuracy": raw_acc_df["Accuracy_raw"],
    "Binary PR-AUC": raw_acc_df["PR_AUC_raw"],
    "Binned Accuracy": bin_acc_df["Accuracy_bin"],
    "Binned PR-AUC": bin_acc_df["PR_AUC_bin"]
})

# Plot settings
x = np.arange(len(dimensions))  # Trait positions
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(10,6))

# Bars
ax.bar(x - 1.5*width, plot_df["Binary Accuracy"], width, label="Binary Accuracy", color="teal")
ax.bar(x - 0.5*width, plot_df["Binary PR-AUC"], width, label="Binary PR-AUC", color="cyan")
ax.bar(x + 0.5*width, plot_df["Binned Accuracy"], width, label="Binned Accuracy", color="salmon")
ax.bar(x + 1.5*width, plot_df["Binned PR-AUC"], width, label="Binned PR-AUC", color="orange")

# Labels
ax.set_xticks(x)
ax.set_xticklabels(plot_df["Trait"])
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Performance Across OCEAN Traits")
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("ocean_bar_chart.png", dpi=300)
plt.show()

