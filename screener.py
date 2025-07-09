import streamlit as st
from login import login_section
from email_sender import send_email_to_candidate
import pdfplumber
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dateutil import parser
from datetime import datetime
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load SpaCy Model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")

st.set_page_config(page_title="Resume Screener Pro", layout="wide")
st.title("📂 Resume Screener Pro")

# --- JD Upload ---
job_roles = {"Upload my own": None}
jd_dir = "data"
if os.path.exists(jd_dir):
    for fname in os.listdir(jd_dir):
        if fname.endswith(".txt"):
            job_roles[fname.replace(".txt", "").replace("_", " ").title()] = os.path.join(jd_dir, fname)

jd_option = st.selectbox("📌 Select Job Role or Upload Your Own JD", list(job_roles.keys()))
jd_text = ""
if jd_option == "Upload my own":
    jd_file = st.file_uploader("Upload Job Description (TXT)", type="txt")
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
else:
    jd_path = job_roles[jd_option]
    if jd_path:
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

resume_files = st.file_uploader("📄 Upload Resumes", type="pdf", accept_multiple_files=True)
cutoff = st.slider("📈 Minimum Match Score (%)", 0, 100, 80)
min_exp = st.slider("💼 Minimum Experience (years)", 0, 15, 2)
min_cgpa = st.slider("🎓 Minimum CGPA (out of 10)", 0.0, 10.0, 7.5, step=0.1)

# --- Functions ---
def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            return ''.join([page.extract_text() or '' for page in pdf.pages])
    except:
        return ""

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else "Not Found"

def extract_profile_links(text):
    # Accepts varied formats
    linkedin = re.search(r'(https?://)?(www\.)?linkedin\.com/in/[^\s)>,\]]+', text, re.IGNORECASE)
    github = re.search(r'(https?://)?(www\.)?github\.com/[^\s)>,\]]+', text, re.IGNORECASE)
    return (linkedin.group(0) if linkedin else None), (github.group(0) if github else None)

def extract_scores(text):
    cgpa = re.search(r'([\d.]+)\s*/\s*10\s*(CGPA|C.G.P.A)?', text, re.IGNORECASE)
    try:
        return float(cgpa.group(1)) if cgpa else 0.0
    except:
        return 0.0

def extract_experience(text):
    pattern = r'([A-Za-z]{3,9}\s\d{4})\s*[-–—]\s*([A-Za-z]{3,9}\s\d{4}|present)'
    ranges = re.findall(pattern, text, re.IGNORECASE)
    total_months = 0
    for start, end in ranges:
        try:
            s = parser.parse(start)
            e = parser.parse(end) if "present" not in end.lower() else datetime.today()
            total_months += (e.year - s.year) * 12 + (e.month - s.month)
        except:
            continue
    return round(total_months / 12, 1)

def tfidf_score(jd, resume):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([jd, resume])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

def spacy_score(jd, resume):
    return round(nlp(jd).similarity(nlp(resume)) * 100, 2)

def skill_match_score(jd, resume, years):
    jd_words = set(re.findall(r'\b\w+\b', jd.lower()))
    resume_words = set(re.findall(r'\b\w+\b', resume.lower()))
    core_skills = ['python', 'java', 'sql', 'html', 'css', 'javascript', 'react', 'machine', 'learning']
    matched = [word for word in jd_words if word in resume_words]
    total_weight = sum([3 if word in core_skills else 1 for word in matched])
    possible = sum([3 if word in core_skills else 1 for word in jd_words])
    score = (total_weight / possible) * 100 if possible else 0
    score += min(years, 10)
    return round(score, 2), matched, [w for w in core_skills if w in jd_words and w not in resume_words]

def final_score(tfidf, spacy, skill):
    return round(tfidf * 0.3 + spacy * 0.3 + skill * 0.4, 2)

def is_shortlisted(score, exp, cgpa, linkedin, github):
    return score >= cutoff and exp >= min_exp and cgpa >= min_cgpa and linkedin and github

# --- Main Screening ---
if jd_text and resume_files:
    st.info("🔍 Screening resumes...")
    results = []
    for file in resume_files:
        text = extract_text_from_pdf(file)
        email = extract_email(text)
        linkedin, github = extract_profile_links(text)
        cgpa = extract_scores(text)
        exp = extract_experience(text)
        tfidf = tfidf_score(jd_text, text)
        spacy_sim = spacy_score(jd_text, text)
        skill_score, matched, missing = skill_match_score(jd_text, text, exp)
        score = final_score(tfidf, spacy_sim, skill_score)

        tag = "✅ Good Fit"
        if score > 90 and exp >= 3:
            tag = "🔥 Top Talent"
        if not linkedin or not github:
            tag = "⚠️ Missing Profile"
        if not is_shortlisted(score, exp, cgpa, linkedin, github):
            tag = "❌ Not Shortlisted"

        results.append({
            "Name": file.name,
            "Score": score,
            "Experience": exp,
            "CGPA": cgpa,
            "Email": email,
            "LinkedIn": linkedin or "❌",
            "GitHub": github or "❌",
            "Tag": tag,
            "Missing Skills": ", ".join(missing)
        })

    df = pd.DataFrame(results)
    st.success("✅ Screening complete.")
    st.dataframe(df, use_container_width=True)

    st.download_button("📥 Download Results", data=df.to_csv(index=False), file_name="screening_results.csv")

    if any(df["LinkedIn"] == "❌") or any(df["GitHub"] == "❌"):
        st.warning("⚠️ Some candidates are missing LinkedIn or GitHub links.")

    if any(df["Tag"] == "❌ Not Shortlisted"):
        st.error("🚫 Some resumes failed the minimum criteria and were not shortlisted.")
