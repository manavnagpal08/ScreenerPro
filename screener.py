# screener.py

import streamlit as st
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

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")

st.set_page_config(page_title="Resume Screener Pro", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("📂 Resume Screener Pro")

# Load job descriptions
job_roles = {"Upload my own": None}
if os.path.exists("data"):
    for file in os.listdir("data"):
        if file.endswith(".txt"):
            job_roles[file.replace(".txt", "").replace("_", " ").title()] = os.path.join("data", file)

jd_option = st.selectbox("📌 Select Job Role or Upload JD", list(job_roles.keys()))
jd_text = ""
if jd_option == "Upload my own":
    jd_file = st.file_uploader("Upload Job Description", type="txt")
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
else:
    with open(job_roles[jd_option], "r", encoding="utf-8") as f:
        jd_text = f.read()

resume_files = st.file_uploader("📄 Upload Resumes", type="pdf", accept_multiple_files=True)
cutoff = st.slider("Minimum Match Score (%)", 0, 100, 80)
min_exp = st.slider("Minimum Experience (Years)", 0, 15, 2)
min_cgpa = st.slider("Minimum CGPA", 0.0, 10.0, 7.5)

# ----------------------- Extraction Functions -----------------------
def extract_text(file):
    try:
        with pdfplumber.open(file) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)
    except:
        return ""

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else "Not found"

def extract_links(text):
    linkedin = re.search(r'https?://(www\.)?linkedin\.com/in/[^\s,)\]]+', text, re.IGNORECASE)
    github = re.search(r'https?://(www\.)?github\.com/[^\s,)\]]+', text, re.IGNORECASE)
    return linkedin.group(0) if linkedin else "❌", github.group(0) if github else "❌"

def extract_experience(text):
    ranges = re.findall(r'([A-Za-z]{3,9}\s\d{4})\s*[-–—]\s*([A-Za-z]{3,9}\s\d{4}|present)', text, re.IGNORECASE)
    total_months = 0
    for start, end in ranges:
        try:
            s = parser.parse(start)
            e = datetime.today() if "present" in end.lower() else parser.parse(end)
            total_months += (e.year - s.year) * 12 + (e.month - s.month)
        except:
            continue
    return round(total_months / 12, 1)

def extract_cgpa(text):
    cgpa_match = re.search(r'(\d\.\d{1,2})\s*/\s*10', text)
    try:
        return float(cgpa_match.group(1)) if cgpa_match else 0.0
    except:
        return 0.0

# ---------------------- Scoring Functions ---------------------------
def tfidf_score(jd, resume):
    vect = TfidfVectorizer(stop_words="english")
    tfidf = vect.fit_transform([jd, resume])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

def spacy_score(jd, resume):
    return round(nlp(jd).similarity(nlp(resume)) * 100, 2)

def skill_score(jd, resume, exp):
    jd_words = set(re.findall(r'\b\w+\b', jd.lower()))
    resume_words = set(re.findall(r'\b\w+\b', resume.lower()))
    core = ['python', 'java', 'sql', 'html', 'css', 'javascript', 'react', 'machine', 'learning']
    matched = [word for word in jd_words if word in resume_words]
    weight = sum(3 if w in core else 1 for w in matched)
    total = sum(3 if w in core else 1 for w in jd_words)
    score = (weight / total) * 100 if total else 0
    score += min(exp, 10)
    missing = [w for w in core if w in jd_words and w not in resume_words]
    return round(score, 2), matched, missing

def final_score(tfidf, spacy, skill):
    return round(tfidf * 0.3 + spacy * 0.3 + skill * 0.4, 2)

# ----------------------- Processing Logic ---------------------------
if jd_text and resume_files:
    results = []
    for file in resume_files:
        text = extract_text(file)
        exp = extract_experience(text)
        email = extract_email(text)
        cgpa = extract_cgpa(text)
        linkedin, github = extract_links(text)
        tfidf = tfidf_score(jd_text, text)
        spacy_sim = spacy_score(jd_text, text)
        skill, matched, missing = skill_score(jd_text, text, exp)
        score = final_score(tfidf, spacy_sim, skill)

        tag = "✅ Good Fit"
        if score > 90 and exp >= 3:
            tag = "🔥 Top Talent"
        if not linkedin or not github:
            tag = "⚠️ Missing Profile"
        if score < cutoff or exp < min_exp or cgpa < min_cgpa:
            tag = "❌ Not Shortlisted"

        results.append({
            "Name": file.name,
            "Score": score,
            "Exp": exp,
            "CGPA": cgpa,
            "Email": email,
            "LinkedIn": linkedin,
            "GitHub": github,
            "Tag": tag,
            "Matched Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing)
        })

    df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    st.success("✅ Screening Complete")
    st.dataframe(df, use_container_width=True)

    # 📊 Metrics
    st.markdown("### 📈 Metrics & Charts")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Score", f"{df['Score'].mean():.1f}%")
    col2.metric("Avg Experience", f"{df['Exp'].mean():.1f} yrs")
    col3.metric("Avg CGPA", f"{df['CGPA'].mean():.2f}")

    # Tag pie chart
    tag_counts = df["Tag"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(tag_counts, labels=tag_counts.index, autopct='%1.1f%%', startangle=140)
    ax1.axis("equal")
    st.pyplot(fig1)

    # Wordcloud
    st.markdown("### ☁️ Word Cloud of Missing Skills")
    all_words = " ".join(df["Missing Skills"].dropna())
    wc = WordCloud(width=1000, height=400, background_color="white").generate(all_words)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2)

    # Email Sending
    st.markdown("### ✉️ Manual Email Sender")
    send_df = df[df["Email"].str.contains("@")]
    selected = st.multiselect("Choose Emails to Send", options=send_df["Email"].tolist())
    subj = st.text_input("Subject", value="🎉 You’ve been shortlisted!")
    body = st.text_area("Email Template", value="""
Dear {{name}},

You've been shortlisted for the position.  
Your score: {{score}}%.  

Regards,  
HR Team
""", height=160)

    if st.button("📤 Send Emails"):
        for email in selected:
            row = send_df[send_df["Email"] == email].iloc[0]
            msg = body.replace("{{name}}", row["Name"]).replace("{{score}}", str(row["Score"]))
            send_email_to_candidate(
                name=row["Name"], score=row["Score"], feedback=row["Tag"],
                recipient=row["Email"], subject=subj, message=msg
            )
        st.success("📬 Emails sent successfully!")

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="screening_results.csv", mime="text/csv")
