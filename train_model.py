# train_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer
import joblib

df = pd.read_csv("resume_jd_labeled_data.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

X = np.array([
    np.concatenate([model.encode(jd), model.encode(resume)])
    for jd, resume in zip(df["jd_text"], df["resume_text"])
])
y = df["score"]

reg = Ridge()
reg.fit(X, y)
joblib.dump(reg, "ml_screening_model.pkl")
print("✅ ML model trained and saved as ml_screening_model.pkl")
