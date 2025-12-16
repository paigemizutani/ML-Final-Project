import pandas as pd
from sklearn.model_selection import train_test_split
import os

CSV_PATH = "data/reddit_post.csv"    # path to raw CSV
OUT_DIR = "data/balanced"            # where to write balanced data
n_per_type = 23000                   # number of rows we want per MBTI type  
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df[['mbti', 'body']].dropna().astype(str)

# count per MBTI
counts = df['mbti'].value_counts().sort_index()
print("Class counts (original):")
print(counts)

print(f"\nSampling {n_per_type:,} rows per MBTI type...")

balanced_parts = []
for mbti_type, group in df.groupby('mbti'):
    sampled = group.sample(n=n_per_type, random_state=RANDOM_STATE)
    balanced_parts.append(sampled)

balanced_df = pd.concat(balanced_parts).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

print("\nCounts after balancing:")
print(balanced_df['mbti'].value_counts().sort_index())

# Save full balanced dataset 
balanced_csv = os.path.join(OUT_DIR, f"balanced_data.csv")
balanced_df.to_csv(balanced_csv, index=False)
print(f"\nSaved balanced CSV to: {balanced_csv}")

# make train/val/test splits
train_df, temp_df = train_test_split(balanced_df, test_size=0.30, stratify=balanced_df['mbti'], random_state=RANDOM_STATE)
val_df, test_df = train_test_split(temp_df, test_size=2/3, stratify=temp_df['mbti'], random_state=RANDOM_STATE)

print("\nFinal splits (counts per MBTI):")
print("TRAIN:", train_df['mbti'].value_counts().sort_index().iloc[:5])
print("VAL:", val_df['mbti'].value_counts().sort_index().iloc[:5])
print("TEST:", test_df['mbti'].value_counts().sort_index().iloc[:5])

train_df.to_csv(os.path.join(OUT_DIR, f"train_balanced.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, f"val_balanced.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, f"test_balanced.csv"), index=False)

print("\nSaved train/val/test CSVs to:", OUT_DIR)
