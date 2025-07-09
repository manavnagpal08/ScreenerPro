import streamlit as st
import pandas as pd
import os
import re
import pdfplumber
from datetime import datetime
from dateutil import parser
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from email_sender import send_email_to_candidate

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")

st.set_page_config(page_title="Resume Screener Pro", layout="wide")

# --- Global CSS ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.metric-box {
    background: #f9f9f9;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📂 Resume Screener Pro")

# --- Load Job Descriptions ---
job_roles = {"Upload my own": None}
if os.path.exists("data"):
    for file in os.listdir("data"):
        if file.endswith(".txt"):
            job_roles[file.replace(".txt", "").title()] = os.path.join("data", file)

jd_option = st.selectbox("📌 Select Job Role or Upload Your Own JD", list(job_roles.keys()))
jd_text = ""
if jd_option == "Upload my own":
    jd_file = st.file_uploader("Upload JD (.txt)", type="txt")
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
else:
    with open(job_roles[jd_option], "r", encoding="utf-8") as f:
        jd_text = f.read()

# Upload Resumes
resume_files = st.file_uploader("📄 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
cutoff = st.slider("📈 Score Cutoff (%)", 0, 100, 75)
min_exp = st.slider("💼 Min Experience (Years)", 0, 15, 2)
min_cgpa = st.slider("🎓 Min CGPA (out of 10)", 0.0, 10.0, 7.5)

# === Utility Functions ===
def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            return ''.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        return ""

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else "❌"

def extract_links(text):
    linkedin_match = re.search(r'(https?:\/\/)?(www\.)?linkedin\.com\/in\/[^\s\)\]]+', text, re.IGNORECASE)
    github_match = re.search(r'(https?:\/\/)?(www\.)?github\.com\/[^\s\)\]]+', text, re.IGNORECASE)
    return linkedin_match.group(0) if linkedin_match else "❌", github_match.group(0) if github_match else "❌"

def extract_scores(text):
    cgpa = re.search(r'(\d{1,2}\.\d{1,2})\s*\/\s*10\s*', text)
    try:
        cgpa_val = float(cgpa.group(1)) if cgpa else 0
    except:
        cgpa_val = 0
    return cgpa_val

def extract_experience(text):
    text = text.lower()
    years = []
    for match in re.findall(r'(\d{1,2})\+?\s*(years?|yrs?|year)', text):
        try:
            years.append(int(match[0]))
        except:
            continue

    total_months = 0
    date_ranges = re.findall(r'([A-Za-z]{3,9}\s\d{4})\s*[-–—]\s*([A-Za-z]{3,9}\s\d{4}|present)', text)
    for start, end in date_ranges:
        try:
            s = parser.parse(start)
            e = parser.parse(end) if 'present' not in end.lower() else datetime.today()
            months = (e.year - s.year) * 12 + (e.month - s.month)
            total_months += months
        except:
            continue

    date_based = round(total_months / 12, 1)
    return max(date_based, max(years) if years else 0)

def tfidf_score(jd, resume):
    vect = TfidfVectorizer(stop_words='english')
    tfidf = vect.fit_transform([jd, resume])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

def spacy_score(jd, resume):
    return round(nlp(jd).similarity(nlp(resume)) * 100, 2)

def skill_score(jd, resume, years):
    jd_words = set(re.findall(r'\b\w+\b', jd.lower()))
    resume_words = set(re.findall(r'\b\w+\b', resume.lower()))
    core_skills = ['python', 'java', 'sql', 'html', 'css', 'javascript', 'react', 'machine', 'learning']
    matched = [word for word in jd_words if word in resume_words]
    weight = sum([3 if w in core_skills else 1 for w in matched])
    total = sum([3 if w in core_skills else 1 for w in jd_words])
    score = (weight / total) * 100 if total else 0
    score += min(years, 10)
    missing = [w for w in core_skills if w in jd_words and w not in resume_words]
    return round(score, 2), matched, missing
def final_score(tfidf, spacy_sim, skill):
    return round(tfidf * 0.3 + spacy_sim * 0.3 + skill * 0.4, 2)

def get_tag(score, exp, cgpa, linkedin, github):
    if linkedin == "❌" or github == "❌":
        return "⚠️ Missing Profile"
    if score >= 90 and exp >= 3 and cgpa >= 8:
        return "🔥 Top Talent"
    elif score >= 75:
        return "✅ Good Fit"
    else:
        return "❌ Needs Review"

# =========================
# 🔍 Resume Screening Logic
# =========================
if jd_text and resume_files:
    st.subheader("🔎 Screening Resumes...")

    results = []
    all_keywords = []

    for file in resume_files:
        raw_text = extract_text_from_pdf(file)
        if not raw_text.strip():
            st.error(f"❌ Could not read {file.name}. Skipping.")
            continue

        experience = extract_experience(raw_text)
        email = extract_email(raw_text)
        linkedin, github = extract_links(raw_text)
        cgpa = extract_scores(raw_text)
        tfidf = tfidf_score(jd_text, raw_text)
        spacy_sim = spacy_score(jd_text, raw_text)
        skill, matched, missing = skill_score(jd_text, raw_text, experience)
        total_score = final_score(tfidf, spacy_sim, skill)
        tag = get_tag(total_score, experience, cgpa, linkedin, github)

        all_keywords.extend(matched)

        results.append({
            "File Name": file.name,
            "Score (%)": total_score,
            "TF-IDF": tfidf,
            "SpaCy": spacy_sim,
            "Skill Match": skill,
            "Experience (yrs)": experience,
            "CGPA": cgpa,
            "Email": email,
            "LinkedIn": linkedin,
            "GitHub": github,
            "Tag": tag,
            "Missing Skills": ", ".join(missing)
        })

    # --- Display Table ---
    df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False)
    st.success(f"✅ Screening Complete! {len(df)} resumes processed.")
    st.dataframe(df, use_container_width=True)

    # Download Option
    st.download_button("📥 Download Results CSV", data=df.to_csv(index=False), file_name="screening_results.csv")

    # Warnings
    if any(df["LinkedIn"] == "❌") or any(df["GitHub"] == "❌"):
        st.warning("⚠️ Some profiles are missing LinkedIn or GitHub links.")
    # ============================
    # 📊 AI Insights & Analytics
    # ============================
    st.subheader("📊 Resume Insights & Visuals")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Avg. Score", f"{df['Score (%)'].mean():.2f}%")
    col2.metric("💼 Avg. Experience", f"{df['Experience (yrs)'].mean():.1f} yrs")
    col3.metric("🎓 Avg. CGPA", f"{df['CGPA'].mean():.2f}")
    col4.metric("✅ Good Fit Candidates", f"{(df['Tag'] == '✅ Good Fit').sum()}")

    # ==========================
    # 📌 Tag Distribution Chart
    # ==========================
    st.markdown("#### 🧮 Candidate Tag Distribution")
    tag_counts = df['Tag'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.pie(tag_counts.values, labels=tag_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # ==========================
    # 📊 Experience Range Chart
    # ==========================
    st.markdown("#### 💼 Experience Groups")
    bins = [0, 2, 5, 10, 30]
    labels = ['0-2 yrs', '3-5 yrs', '6-10 yrs', '10+ yrs']
    df['Exp Range'] = pd.cut(df['Experience (yrs)'], bins=bins, labels=labels)
    exp_counts = df['Exp Range'].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(exp_counts.index, exp_counts.values, color="#00cec9")
    ax2.set_ylabel("Candidates")
    ax2.set_title("Experience Distribution")
    st.pyplot(fig2)

    # ==========================
    # ☁️ Word Cloud of Skills
    # ==========================
    st.markdown("#### ☁️ Matched Skill Cloud")
    text_blob = ' '.join(all_keywords)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_blob)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis('off')
    st.pyplot(fig3)

    # ==========================
    # 🏅 Leaderboard
    # ==========================
    st.markdown("#### 🏅 Top 5 Candidates Leaderboard")
    top5 = df.head(5)
    for idx, row in top5.iterrows():
        st.markdown(f"**{row['File Name']}** — 🎯 Score: {row['Score (%)']}% | 💼 Exp: {row['Experience (yrs)']} yrs | 🎓 CGPA: {row['CGPA']}")
        st.caption(f"📧 {row['Email']} | 🔗 LinkedIn: {row['LinkedIn']} | 💻 GitHub: {row['GitHub']}")
        st.progress(int(row["Score (%)"]))

    # ==========================
    # 📧 Email Manual Sender
    # ==========================
    st.markdown("### 📤 Send Email to Candidate")
    selected = st.selectbox("Choose Candidate to Email", df["File Name"])
    selected_row = df[df["File Name"] == selected].iloc[0]
    subject = st.text_input("Subject", value="🎉 You have been shortlisted!")
    body = st.text_area("Email Body", value=f"""
Dear {selected.replace('.pdf','')},

Congratulations! You have been shortlisted for the next round.

🧠 Total Score: {selected_row['Score (%)']}%
💼 Experience: {selected_row['Experience (yrs)']} years
💬 Tag: {selected_row['Tag']}

We'll contact you soon with more details.

Regards,  
HR Team — Shree Ram Recruitments
""", height=180)

    if st.button("📨 Send Email"):
        send_email_to_candidate(
            name=selected_row["File Name"],
            score=selected_row["Score (%)"],
            feedback=selected_row["Tag"],
            recipient=selected_row["Email"],
            subject=subject,
            message=body
        )
        st.success("✅ Email Sent Successfully!")
    # ==========================
    # 📄 Resume Previews (Side-by-side)
    # ==========================
    st.markdown("### 🧾 Resume Text Viewer")
    selected_resume = st.selectbox("🔍 Select Resume to Preview", df["File Name"])
    raw_text = resume_text_map.get(selected_resume, "")
    st.text_area("🔍 Resume Text", value=raw_text, height=300, key="resume_preview")

    # ==========================
    # 📋 JD Keyword Highlights
    # ==========================
    st.markdown("### 📌 Job Description Keywords Match")
    top_keywords = list(set(jd_text.lower().split()))
    matched_set = set(df[df["File Name"] == selected_resume]["Matched Keywords"].values[0].split(", "))
    highlighted = [
        f"✅ **{word}**" if word in matched_set else word
        for word in top_keywords if len(word) > 3
    ]
    st.markdown(" ".join(highlighted[:100]))

    # ==========================
    # 💡 Smart Recommendation Panel
    # ==========================
    st.markdown("### 💡 Recommendation Engine")

    def smart_recommendation(row):
        suggestions = []
        if row["Experience (yrs)"] < min_exp:
            suggestions.append("📉 Gain more hands-on experience.")
        if row["Score (%)"] < cutoff:
            suggestions.append("🎯 Improve skill match with JD.")
        if row["CGPA"] < 7.5:
            suggestions.append("🎓 Improve academic profile.")
        if row["LinkedIn"] == "❌":
            suggestions.append("🔗 Add a LinkedIn profile.")
        if row["GitHub"] == "❌":
            suggestions.append("💻 Share your GitHub projects.")
        if not suggestions:
            suggestions.append("✅ Great profile! Consider for next round.")
        return " | ".join(suggestions)

    df["Recommendation"] = df.apply(smart_recommendation, axis=1)

    # ==========================
    # 🔎 Resume Scoring Breakdown
    # ==========================
    st.markdown("### 🔍 Score Explanation")

    def show_breakdown(row):
        st.write(f"📁 Resume: **{row['File Name']}**")
        st.write(f"✅ TF-IDF Similarity: {row['TFIDF Score']}%")
        st.write(f"🧠 SpaCy Semantic Match: {row['SpaCy Score']}%")
        st.write(f"🎯 Skill Matching Score: {row['Skill Match Score']}%")
        st.write(f"📈 Final Score: **{row['Score (%)']}%**")
        st.write(f"❌ Missing Skills: {row['Missing Skills'] or 'None'}")
        st.info(row["Recommendation"])

    if st.checkbox("🧠 Show Full Score Breakdown for All Candidates"):
        for _, row in df.iterrows():
            with st.expander(f"📎 {row['File Name']} — {row['Tag']}"):
                show_breakdown(row)
    else:
        with st.expander(f"📎 {selected_resume} — {df[df['File Name'] == selected_resume]['Tag'].values[0]}"):
            show_breakdown(df[df['File Name'] == selected_resume].iloc[0])

    # ==========================
    # 🔍 Compare Two Candidates
    # ==========================
    st.markdown("### 🤼 Compare Two Candidates")
    c1, c2 = st.columns(2)
    name1 = c1.selectbox("Candidate A", df["File Name"], key="comp1")
    name2 = c2.selectbox("Candidate B", df["File Name"], index=1 if len(df) > 1 else 0, key="comp2")

    row1 = df[df["File Name"] == name1].iloc[0]
    row2 = df[df["File Name"] == name2].iloc[0]

    comp_df = pd.DataFrame({
        "Metric": ["Score (%)", "Experience (yrs)", "CGPA", "12th Marks", "Missing Skills"],
        name1: [row1["Score (%)"], row1["Experience (yrs)"], row1["CGPA"], row1["12th Marks"], row1["Missing Skills"]],
        name2: [row2["Score (%)"], row2["Experience (yrs)"], row2["CGPA"], row2["12th Marks"], row2["Missing Skills"]],
    })

    st.table(comp_df)

    # ==========================
    # 📥 Final CSV Download
    # ==========================
    st.markdown("### 📄 Export All Results")
    full_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Full Report", full_csv, file_name="final_screening_results.csv", mime="text/csv")
