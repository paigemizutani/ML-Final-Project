from openai import OpenAI

client = OpenAI(api_key="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
")

import json
import os

# -----------------------------
# Initialize Client
# -----------------------------


# -----------------------------
# Load JSON files
# -----------------------------
with open("/Users/ashleyjeon/Desktop/Applied ML/ML-Final-Project/personalityDescription/mbti_descriptions.json", "r") as f:
    mbti_descriptions = json.load(f)

with open("/Users/ashleyjeon/Desktop/Applied ML/ML-Final-Project/personalityDescription/ocean_descriptions.json", "r") as f:
    ocean_descriptions = json.load(f)

# -----------------------------
# User Inputs
# -----------------------------
post_text = input("Enter post text: ")
predicted_mbti = input("Enter predicted MBTI (e.g., INFJ): ")

scores = {}
for trait in ["O", "C", "E", "A", "N"]:
    scores[trait] = float(input(f"Enter {trait} score (0–100): "))

# -----------------------------
# Helper function
# -----------------------------
def scaled_desc(score, trait_key):
    if score < 40:
        return f"leans toward tendencies associated with low {trait_key}."
    elif score > 60:
        return f"leans toward tendencies associated with high {trait_key}."
    else:
        return f"shows a balanced expression of {trait_key}."

# -----------------------------
# Build the Prompt
# -----------------------------
mbti_info = mbti_descriptions.get(predicted_mbti, "")

trait_block = "\n".join(
    [f"- {t}: {scaled_desc(scores[t], t)}" for t in ["O", "C", "E", "A", "N"]]
)

prompt = f"""
You are a psychology expert specializing in personality analysis.

Write a natural, human-readable two-paragraph personality description.
Avoid numbers and avoid referencing scores. Describe only traits and behaviors.

Post:
\"\"\"{post_text}\"\"\"

Predicted MBTI:
{predicted_mbti}

MBTI Description:
{mbti_info}

OCEAN Interpretation:
{trait_block}

Instructions:
- Blend MBTI and OCEAN traits smoothly
- No technical or statistical language
- No percentages or numbers
- Make the writing sound personal and intuitive
"""

# -----------------------------
# Generate Response
# -----------------------------
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=400,
    temperature=0.7
)

generated_text = response.choices[0].message.content.strip()

print("\n------ Generated Personality Description ------\n")
print(generated_text)
