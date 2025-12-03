import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
import torch.nn as nn
import torch.nn.functional as F

NUM_EPOCHS = 4
MAX_LEN = 128
BATCH_SIZE = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-uncased"

# LOAD DATA
df = pd.read_csv("data/reddit_post.csv", on_bad_lines='skip')
df = df[['mbti', 'body']].dropna().astype(str)
df = df.sample(n=80000, random_state=42)

# CREATE MBTI DIMENSIONS
df['E_I'] = df['mbti'].apply(lambda x: 0 if x[0] == 'I' else 1)
df['S_N'] = df['mbti'].apply(lambda x: 0 if x[1] == 'S' else 1)
df['T_F'] = df['mbti'].apply(lambda x: 0 if x[2] == 'T' else 1)
df['J_P'] = df['mbti'].apply(lambda x: 0 if x[3] == 'J' else 1)

# TOKENIZATION
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
def encode(texts):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

# TRAIN/TEST SPLIT
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['body'], df[['E_I','S_N','T_F','J_P']], test_size=0.35, random_state=42
)
train_encodings = encode(train_texts)
test_encodings = encode(test_texts)

train_dataset = TensorDataset(
    train_encodings['input_ids'], train_encodings['attention_mask'],
    torch.tensor(train_labels.values, dtype=torch.float)
)
test_dataset = TensorDataset(
    test_encodings['input_ids'], test_encodings['attention_mask'],
    torch.tensor(test_labels.values, dtype=torch.float)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# -------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------
def eval_and_plot(model, test_loader, dim_index):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            b_true = labels[:, dim_index].cpu().numpy()
            logits = model(input_ids.to(DEVICE), attention_mask.to(DEVICE))
            probs = torch.sigmoid(logits[:, dim_index]).cpu().numpy()

            y_true.extend(b_true)
            y_prob.extend(probs)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"PR AUC:  {pr_auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve {dim_index} (AUC = {pr_auc:.3f})")
    plt.savefig(f"pr_curve_{dim_index}.png", dpi=300)
    plt.close()

# -------------------------------------------------------------
# MULTITASK MODEL + FOCAL LOSS
# -------------------------------------------------------------
class MultiTaskBERT(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # 4 classification heads
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(4)])

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)

        logits = []
        for head in self.heads:
            logits.append(head(pooled))   # shape [batch, 1]

        return torch.cat(logits, dim=1)   # shape [batch, 4]

# FOCAL LOSS FOR BINARY TASKS
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt)**self.gamma * bce
        return focal.mean()

# -------------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------------
model = MultiTaskBERT(MODEL_NAME).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fct = FocalLoss(alpha=1.0, gamma=2.0)

for epoch in range(NUM_EPOCHS):
    model.train()
    for input_ids, attention_mask, labels in train_loader:
        optimizer.zero_grad()

        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(input_ids, attention_mask)   # [batch_size, 4]
        loss = loss_fct(logits, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} finished. Loss = {loss.item():.4f}")

# -------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------
for dim_idx in range(4):
    print(f"\nEvaluating dimension {dim_idx}...")
    eval_and_plot(model, test_loader, dim_idx)

# -------------------------------------------------------------
# PREDICTION
# -------------------------------------------------------------
dimensions = ['E_I','S_N','T_F','J_P']
dim_map = {"E_I":["I","E"], "S_N":["S","N"], "T_F":["T","F"], "J_P":["J","P"]}

new_posts = [
    "Ok ya not sure if this model is actually predicting anything",
    "I love leading teams and planning long-term goals.",
    "I really enjoy spending time alone thinking about ideas.",
    "whatever this is sooo annoying",
    "Wait Im actually so excited for thiss",
    "Don't worry its going to be ok its difficult for most people",
    "I don't even know what I'm going to eat for lunch tomorrow tbh",
    "I already have my entire life planned out so I'm chilling for now thankfully"
]

encoded = tokenizer(new_posts, padding="max_length", truncation=True, 
                    max_length=MAX_LEN, return_tensors="pt")

input_ids = encoded["input_ids"].to(DEVICE)
attention_mask = encoded["attention_mask"].to(DEVICE)

model.eval()
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    probs = torch.sigmoid(logits)

predicted_types = []
for i in range(len(new_posts)):
    letters = []
    for dim_idx, dim in enumerate(dimensions):
        pred = int(probs[i, dim_idx] >= 0.5)
        letters.append(dim_map[dim][pred])
    predicted_types.append("".join(letters))

for post, mbti in zip(new_posts, predicted_types):
    print(f"\nPost: {post}\nPredicted MBTI: {mbti}")
