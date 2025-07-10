import pandas as pd
import numpy as np
import re
from datetime import datetime # Still needed for extract_years_of_experience helper, though not used in extract_features for regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer
import joblib
import os

# --- Configuration ---
MODEL_SAVE_PATH = "ml_screening_model.pkl"
DATASET_PATH = "resume_jd_labeled_data_scaled.csv" # Path to your uploaded dataset

# Load embedding model
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading SentenceTransformer model: {e}")
    print("Please ensure you have an active internet connection or the model is cached locally.")
    exit() # Exit if the embedding model cannot be loaded

# --- Helper Functions (Copied/Adapted from your Streamlit app) ---
def clean_text(text):
    """Cleans text by removing newlines, extra spaces, and non-ASCII characters."""
    text = re.sub(r'\n', ' ', text) # Remove newlines
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Remove non-ASCII characters
    return text.strip().lower()

# This function is kept for completeness/potential future use, but its output
# is NOT currently included in the features for the GradientBoostingRegressor.
def extract_years_of_experience(text):
    """Extracts years of experience from a given text by parsing date ranges or keywords."""
    text = text.lower()
    total_months = 0
    job_date_ranges = re.findall(
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\s*(?:to|–|-)\s*(present|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})',
        text
    )

    for start, end in job_date_ranges:
        try:
            start_date = datetime.strptime(start.strip(), '%b %Y')
        except ValueError:
            try:
                start_date = datetime.strptime(start.strip(), '%B %Y')
            except ValueError:
                continue

        if end.strip() == 'present':
            end_date = datetime.now()
        else:
            try:
                end_date = datetime.strptime(end.strip(), '%b %Y')
            except ValueError:
                try:
                    end_date = datetime.strptime(end.strip(), '%B %Y')
                except ValueError:
                    continue

        delta_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        total_months += max(delta_months, 0)

    if total_months == 0:
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\+)?\s*(year|yrs|years)\b', text)
        if not match:
            match = re.search(r'experience[^\d]{0,10}(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

    return round(total_months / 12, 1)


# --- Load Dataset ---
# Now loading from your provided CSV file!
print(f"Loading dataset from {DATASET_PATH}...")
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset '{DATASET_PATH}' loaded successfully.")
    print(f"Dataset head:\n{df.head()}")
    if 'jd_text' not in df.columns or 'resume_text' not in df.columns or 'score' not in df.columns:
        print("❌ Error: Dataset must contain 'jd_text', 'resume_text', and 'score' columns.")
        exit()
except FileNotFoundError:
    print(f"❌ Error: The file '{DATASET_PATH}' was not found.")
    print("Please ensure the CSV file is in the same directory as this script.")
    exit()
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()


# --- Feature Extraction ---
def extract_features(jd_text, resume_text):
    """
    Extracts features from job description and resume text for model training.
    Features include sentence embeddings, keyword overlap, resume length, and matched core skills.
    """
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    # Generate embeddings
    jd_embed = embedding_model.encode(jd_clean)
    resume_embed = embedding_model.encode(resume_clean)

    # Extract extra numerical features
    resume_words = set(re.findall(r'\b\w+\b', resume_clean))
    jd_words = set(re.findall(r'\b\w+\b', jd_clean))
    keyword_overlap = len(resume_words & jd_words)
    resume_len = len(resume_clean.split())

    # Define core skills (can be expanded based on common job requirements)
    core_skills = ['sql', 'excel', 'python', 'tableau', 'powerbi', 'r', 'aws', 'java', 'spring', 'docker', 'kubernetes', 'figma', 'jira']
    matched_skills = sum(1 for skill in core_skills if skill in resume_clean)

    # Concatenate all features into a single vector
    # The order of features must be consistent between training and prediction
    extra_feats = np.array([keyword_overlap, resume_len, matched_skills])
    return np.concatenate([jd_embed, resume_embed, extra_feats])

# Generate features for the entire dataset
print("Extracting features from the dataset...")
# Use a list comprehension for efficiency
X = np.array([
    extract_features(jd, resume)
    for jd, resume in zip(df["jd_text"], df["resume_text"])
])
y = df["score"].values # Assuming 'score' is the target column in your CSV

print(f"Total samples: {len(X)}")
print(f"Feature vector shape: {X.shape}")

# --- Train-Test Split ---
# It's good practice to split your data to evaluate the model on unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --- Model Training ---
print("Training the Gradient Boosting Regressor model...")
# Initialize and train the GradientBoostingRegressor
# Parameters (n_estimators, learning_rate, max_depth) can be tuned for better performance
reg = GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=5, random_state=42)
reg.fit(X_train, y_train) # Fit on training data
print("✅ Model training complete.")

# --- Model Evaluation ---
print("\nEvaluating the model on the test set...")
y_pred = reg.predict(X_test)

# Evaluate using regression metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# You might also want to check the range of predictions
print(f"Predicted score range: {np.min(y_pred):.2f} to {np.max(y_pred):.2f}")

# --- Save the Trained Model ---
try:
    joblib.dump(reg, MODEL_SAVE_PATH)
    print(f"\n✅ ML model retrained with extra features and saved as {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"❌ Error saving model: {e}")

print("\nTraining script finished. You can now use 'ml_screening_model.pkl' in your Streamlit app.")
