import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# --------------------------
# CONFIG
# --------------------------
NUM_EPOCHS = 2
MAX_LEN = 128
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "prajjwal1/bert-tiny"

# --------------------------
# LOAD DATA
# --------------------------
df = pd.read_csv("data/reddit_post.csv")  # adjust path if needed
df = df[['mbti', 'body']].dropna().astype(str)

df = df.sample(n=5000, random_state=42)

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
            labels = batch[2][:,i].to(DEVICE)  # select dimension

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done, Loss: {loss.item():.4f}")

    trained_models[dim] = model

# --------------------------
# PREDICT NEW POST
# --------------------------
new_post = "I enjoy analyzing complex systems and finding logical solutions."
encoded = tokenizer(new_post, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
input_ids = encoded["input_ids"].to(DEVICE)
attention_mask = encoded["attention_mask"].to(DEVICE)

predicted_mbtis = {}
for i, dim in enumerate(dimensions):
    model = trained_models[dim]
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        predicted_mbtis[dim] = dim_map[dim][pred]

predicted_type = "".join([predicted_mbtis[dim] for dim in dimensions])
print(f"\nPredicted MBTI: {predicted_type}")

