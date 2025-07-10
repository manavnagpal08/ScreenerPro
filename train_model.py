import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sentence_transformers import SentenceTransformer
import joblib
import re

# Load scaled dataset
df = pd.read_csv("resume_jd_labeled_data_scaled.csv")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define feature extraction
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def extract_features(jd, resume):
    jd_clean = clean_text(jd)
    resume_clean = clean_text(resume)

    jd_embed = model.encode(jd_clean)
    resume_embed = model.encode(resume_clean)

    # Extra features
    resume_words = set(re.findall(r'\b\w+\b', resume_clean))
    jd_words = set(re.findall(r'\b\w+\b', jd_clean))
    keyword_overlap = len(resume_words & jd_words)
    resume_len = len(resume_clean.split())

    core_skills = ['sql', 'excel', 'python', 'tableau', 'powerbi', 'r', 'aws']
    matched_skills = sum(1 for skill in core_skills if skill in resume_clean)

    extra_feats = np.array([keyword_overlap, resume_len, matched_skills])
    return np.concatenate([jd_embed, resume_embed, extra_feats])

# Generate features
X = np.array([
    extract_features(jd, resume)
    for jd, resume in zip(df["jd_text"], df["resume_text"])
])
y = df["score"].values

# Train enhanced ML model
reg = GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=5, random_state=42)
reg.fit(X, y)

# Save model
joblib.dump(reg, "ml_screening_model.pkl")
print("✅ ML model retrained with extra features and saved as ml_screening_model.pkl")
