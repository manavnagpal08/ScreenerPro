import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
import joblib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from email_sender import send_email_to_candidate
from login import login_section

# --- Load Embedding + ML Model ---
model = SentenceTransformer("all-MiniLM-L6-v2")
try:
    ml_model = joblib.load("ml_screening_model.pkl")
except Exception as e:
    st.error(f"❌ Failed to load ML model: {e}")
    ml_model = None

# --- Helpers ---
def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip().lower()

def extract_text_from_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return ''.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        return f"[ERROR] {str(e)}"

def extract_years_of_experience(text):
    text = text.lower()
    total_months = 0
    job_date_ranges = re.findall(
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\s*(?:to|–|-)\s*(present|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})',
        text
    )

    for start, end in job_date_ranges:
        try:
            start_date = datetime.strptime(start.strip(), '%b %Y')
        except:
            try:
                start_date = datetime.strptime(start.strip(), '%B %Y')
            except:
                continue

        if end.strip() == 'present':
            end_date = datetime.now()
        else:
            try:
                end_date = datetime.strptime(end.strip(), '%b %Y')
            except:
                try:
                    end_date = datetime.strptime(end.strip(), '%B %Y')
                except:
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

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def smart_score(resume_text, jd_text, years_exp):
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    jd_words = set(re.findall(r'\b\w+\b', jd_text.lower()))
    overlap = resume_words & jd_words
    base = min(len(overlap), 25)
    score = (base * 3) + min(years_exp, 10)
    return round(min(score, 100), 2), ", ".join(overlap), "", []

def semantic_score(resume_text, jd_text, years_exp):
    if ml_model is None:
        return 0.0
    try:
        # --- Clean text ---
        def clean_text(t):
            t = re.sub(r'\n', ' ', t)
            t = re.sub(r'\s+', ' ', t)
            return t.strip().lower()

        jd_clean = clean_text(jd_text)
        resume_clean = clean_text(resume_text)

        # --- Embeddings ---
        jd_embed = model.encode(jd_clean)
        resume_embed = model.encode(resume_clean)

        # --- Extra numerical features ---
        resume_words = set(re.findall(r'\b\w+\b', resume_clean))
        jd_words = set(re.findall(r'\b\w+\b', jd_clean))
        keyword_overlap = len(resume_words & jd_words)
        resume_len = len(resume_clean.split())

        core_skills = ['sql', 'excel', 'python', 'tableau', 'powerbi', 'r', 'aws']
        matched_skills = sum(1 for skill in core_skills if skill in resume_clean)

        extra_feats = np.array([keyword_overlap, resume_len, matched_skills])
        features = np.concatenate([jd_embed, resume_embed, extra_feats])

        # --- Predict ---
        score = ml_model.predict([features])[0]
        score = float(np.clip(score, 0, 100))

        # Fallback
        if score < 10:
            from screener import smart_score
            score, _, _, _ = smart_score(resume_text, jd_text, years_exp)

        return round(score, 2)
    except Exception as e:
        print("❌ semantic_score error:", e)
        return 0.0


# --- Streamlit UI ---
st.title("🧠 ScreenerPro – AI Resume Screener")

jd_text = ""
job_roles = {"Upload my own": None}
if os.path.exists("data"):
    for fname in os.listdir("data"):
        if fname.endswith(".txt"):
            job_roles[fname.replace(".txt", "").replace("_", " ").title()] = os.path.join("data", fname)

jd_option = st.selectbox("📌 Select Job Role or Upload Your Own JD", list(job_roles.keys()))
if jd_option == "Upload my own":
    jd_file = st.file_uploader("Upload Job Description (TXT)", type="txt")
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
else:
    jd_path = job_roles[jd_option]
    if jd_path and os.path.exists(jd_path):
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

resume_files = st.file_uploader("📄 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
cutoff = st.slider("📈 Score Cutoff", 0, 100, 80)
min_experience = st.slider("💼 Minimum Experience Required", 0, 15, 2)

if jd_text and resume_files:
    results = []
    resume_text_map = {}
    for file in resume_files:
        text = extract_text_from_pdf(file)
        if text.startswith("[ERROR]"):
            st.error(f"Could not process {file.name}")
            continue

        exp = extract_years_of_experience(text)
        email = extract_email(text)
        score = semantic_score(text, jd_text, exp)
        summary = f"{exp}+ years exp. | {text.strip().splitlines()[0]}" if text else f"{exp}+ years exp."

        results.append({
            "File Name": file.name,
            "Score (%)": score,
            "Years Experience": exp,
            "Summary": summary,
            "Email": email or "Not found"
        })
        resume_text_map[file.name] = text

    df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False)
    df['Tag'] = df.apply(lambda row: "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3 else (
        "✅ Good Fit" if row['Score (%)'] >= 75 else "⚠️ Needs Review"), axis=1)
    shortlisted = df[(df['Score (%)'] >= cutoff) & (df['Years Experience'] >= min_experience)]

    st.metric("📊 Avg. Score", f"{df['Score (%)'].mean():.2f}%")
    st.metric("✅ Shortlisted", len(shortlisted))

    st.markdown("### 🏆 Top Candidates")
    for _, row in df.head(3).iterrows():
        st.subheader(f"{row['File Name']} — {row['Score (%)']}%")
        st.text(row['Summary'])
        st.caption(f"Email: {row['Email']}")
        with st.expander("📄 Resume Preview"):
            st.code(resume_text_map.get(row['File Name'], ""))

    st.download_button("📥 Download Results CSV", df.to_csv(index=False).encode("utf-8"), file_name="results.csv")

    st.markdown("### ✉️ Send Emails to Shortlisted Candidates")
    email_ready = shortlisted[shortlisted['Email'].str.contains("@", na=False)]

    subject = st.text_input("Email Subject", "You're Shortlisted - Next Steps")
    body = st.text_area("Email Body", """
Dear {{name}},

We are pleased to inform you that you've been shortlisted for the next round.
Your profile scored {{score}}% in our AI-powered screening system.

Regards,  
Recruitment Team
""")

    if st.button("📧 Send Emails"):
        for _, row in email_ready.iterrows():
            msg = body.replace("{{name}}", row['File Name'].replace(".pdf", "")).replace("{{score}}", str(row['Score (%)']))
            send_email_to_candidate(
                name=row["File Name"],
                score=row["Score (%)"],
                feedback=row["Tag"],
                recipient=row["Email"],
                subject=subject,
                message=msg
            )
        st.success("✅ Emails sent to shortlisted candidates!")
