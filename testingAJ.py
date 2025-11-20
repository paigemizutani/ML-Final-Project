import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight  # <-- added
import numpy as np  # <-- added

# --------------------------
# CONFIG
# --------------------------
NUM_EPOCHS = 5
MAX_LEN = 128
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "prajjwal1/bert-tiny"

# --------------------------
# LOAD DATA
# --------------------------
df = pd.read_csv("data/reddit_post.csv")
df = df[['mbti', 'body']].dropna().astype(str)

df = df.sample(n=15000, random_state=42)

# --------------------------
# CREATE MBTI DIMENSIONS
# --------------------------
df['E_I'] = df['mbti'].apply(lambda x: 0 if x[0] == 'I' else 1)
df['S_N'] = df['mbti'].apply(lambda x: 0 if x[1] == 'S' else 1)
df['T_F'] = df['mbti'].apply(lambda x: 0 if x[2] == 'T' else 1)
df['J_P'] = df['mbti'].apply(lambda x: 0 if x[3] == 'J' else 1)

# --------------------------
# TOKENIZATION
# --------------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def encode(texts):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

# --------------------------
# TRAIN/TEST SPLIT
# --------------------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['body'], df[['E_I','S_N','T_F','J_P']], test_size=0.2, random_state=42
)

train_encodings = encode(train_texts)
test_encodings = encode(test_texts)

train_dataset = TensorDataset(
    train_encodings['input_ids'], train_encodings['attention_mask'],
    torch.tensor(train_labels.values, dtype=torch.long)
)

test_dataset = TensorDataset(
    test_encodings['input_ids'], test_encodings['attention_mask'],
    torch.tensor(test_labels.values, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# --------------------------
# TRAIN MODELS PER DIMENSION
# --------------------------
dimensions = ['E_I','S_N','T_F','J_P']
dim_map = {"E_I": ["I","E"], "S_N": ["S","N"], "T_F": ["T","F"], "J_P": ["J","P"]}
trained_models = {}

# ---- NEW: COMPUTE CLASS WEIGHTS ----
class_weights = {}
for dim in dimensions:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),  # <-- use numpy array
        y=df[dim]
    )

    class_weights[dim] = torch.tensor(weights, dtype=torch.float).to(DEVICE)

# --------------------------------------

for i, dim in enumerate(dimensions):
    print(f"\nTraining model for {dim}...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch[0].to(DEVICE)
            attention_mask = batch[1].to(DEVICE)
            labels = batch[2][:,i].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # ---- NEW: APPLY CLASS WEIGHTS TO LOSS ----
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights[dim])
            loss = loss_fct(outputs.logits, labels)
            # ------------------------------------------

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done, Loss: {loss.item():.4f}")

    trained_models[dim] = model

# --------------------------
# PREDICT NEW POST
# --------------------------
new_posts = [
    "Ok ya not sure if this model is actually predicting anything",
    "I love leading teams and planning long-term goals.",
    "I really enjoy spending time alone thinking about ideas.",
    "whatever this is sooo annoying",
    "Wait Im actually so excited for thiss"
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

predicted_types = []

for i in range(len(new_posts)):
    predicted_mbtis = {}
    for dim in dimensions:
        model = trained_models[dim]
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[i].unsqueeze(0),
                attention_mask=attention_mask[i].unsqueeze(0)
            )
            pred = torch.argmax(outputs.logits, dim=-1).item()
            predicted_mbtis[dim] = dim_map[dim][pred]

    predicted_types.append("".join(predicted_mbtis[dim] for dim in dimensions))

for post, mbti in zip(new_posts, predicted_types):
    print(f"\nPost: {post}\nPredicted MBTI: {mbti}")
