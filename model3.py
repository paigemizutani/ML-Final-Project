import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score

NUM_EPOCHS = 7
MAX_LEN = 128
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-uncased"

# LOAD CSVs WE MADE
train_df = pd.read_csv("data/balanced/train_balanced.csv")
val_df   = pd.read_csv("data/balanced/val_balanced.csv")
test_df  = pd.read_csv("data/balanced/test_balanced.csv")

dimensions = ['E_I','S_N','T_F','J_P']
for df_ in [train_df, val_df, test_df]:
    if not all(dim in df_.columns for dim in dimensions):
        df_['E_I'] = df_['mbti'].apply(lambda x: 0 if x[0]=='I' else 1)
        df_['S_N'] = df_['mbti'].apply(lambda x: 0 if x[1]=='S' else 1)
        df_['T_F'] = df_['mbti'].apply(lambda x: 0 if x[2]=='T' else 1)
        df_['J_P'] = df_['mbti'].apply(lambda x: 0 if x[3]=='J' else 1)

# TOKENIZER
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
def encode(texts):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

train_enc = encode(train_df['body'])
val_enc   = encode(val_df['body'])
test_enc  = encode(test_df['body'])

# TENSOR DATASETS
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

# MODEL
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    problem_type="multi_label_classification"
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=2e-5)

# CLASS WEIGHTS for BCE
weights = []
for dim in dimensions:
    cw = compute_class_weight(
        "balanced",
        classes=np.array([0,1]),
        y=train_df[dim]
    )
    weights.append(cw[1])  # positive class weight

class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

# TRAINING LOOP
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # VALIDATION
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f}")

# TEST EVALUATION
model.eval()
all_true, all_probs = [], []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()

        all_true.append(labels.numpy())
        all_probs.append(probs)

y_true = np.vstack(all_true)
y_prob = np.vstack(all_probs)

dim_map = {"E_I": ["I","E"], "S_N": ["S","N"], "T_F": ["T","F"], "J_P": ["J","P"]}

for i, dim in enumerate(dimensions):
    y_t = y_true[:, i]
    y_p = y_prob[:, i]

    y_pred = (y_p >= 0.5).astype(int)
    acc = accuracy_score(y_t, y_pred)
    pr_auc = average_precision_score(y_t, y_p)

    print(f"\n{dim}: Accuracy={acc:.4f}, PR AUC={pr_auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_t, y_p)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{dim} PR Curve (AUC = {pr_auc:.3f})")
    plt.savefig(f"pr_curve_{dim}.png", dpi=300)
    plt.close()

# PREDICT NEW POSTS
new_posts = [
    "Ok ya not sure if this model is actually predicting anything",
    "I love leading teams and planning long-term goals.",
    "I really enjoy spending time alone thinking about ideas.",
    "whatever this is sooo annoying",
    "Wait Im actually so excited for thiss",
    "I love spending time with strangers"
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

model.eval()
with torch.no_grad():
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    probs = torch.sigmoid(logits).cpu().numpy()

for post, p in zip(new_posts, probs):
    letters = []
    for i, dim in enumerate(dimensions):
        bit = 1 if p[i] >= 0.5 else 0
        letters.append(dim_map[dim][bit])
    print(f"\nPost: {post}\nPredicted MBTI: {''.join(letters)}")