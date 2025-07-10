import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from email_sender import send_email_to_candidate
from login import login_section
import runpy
from datetime import datetime

# --- Initialize model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- UI Styling ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    transition: all 0.3s ease-in-out;
}
body {
    background: linear-gradient(135deg, #f3f4f6, #f0fdf4);
}
.main .block-container {
    padding: 2rem;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.95);
    box-shadow: 0 12px 36px rgba(0,0,0,0.12);
    animation: fadeInZoom 0.9s ease-in-out;
    border: 1px solid #e0e0e0;
}
@keyframes fadeInZoom {
    from { opacity: 0; transform: scale(0.98); }
    to { opacity: 1; transform: scale(1); }
}
[data-testid="stMetric"] > div {
    background: #ffffffdd;
    padding: 1.2rem;
    border-radius: 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.header("📂 Semantic Resume Screening (AI-Based)")

# --- Job Description Section ---
jd_text = ""
job_roles = {"Upload my own": None}
jd_dir = "data"
if os.path.exists(jd_dir):
    for fname in os.listdir(jd_dir):
        if fname.endswith(".txt"):
            job_roles[fname.replace(".txt", "").replace("_", " ").title()] = os.path.join(jd_dir, fname)

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

resume_files = st.file_uploader("🗕 Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True)
cutoff = st.slider("📈 Score Cutoff", 0, 100, 80)
min_experience = st.slider("💼 Minimum Experience Required", 0, 15, 2)

# --- Extractors ---
def extract_text_from_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return ''.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        return f"[ERROR] {str(e)}"
def extract_years_of_experience(text):
    import re
    from datetime import datetime

    text = text.lower()
    total_months = 0

    # Match ranges like 'Jan 2020 - Jul 2023' or 'May 2022 to Present'
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

    # Fallback: look for phrases like '4 years', '4+ years', 'Experience – 4 year'
    if total_months == 0:
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\+)?\s*(year|yrs|years)\b', text)
        if not match:
            match = re.search(r'experience[^\d]{0,10}(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))





def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def semantic_score(resume_text, jd_text, years_exp):
    try:
        jd_embed = model.encode([jd_text])[0]
        resume_embed = model.encode([resume_text])[0]
        score = cosine_similarity([jd_embed], [resume_embed])[0][0] * 100
        score += min(years_exp, 10)
        return round(min(score, 100), 2)
    except:
        return 0.0

def generate_summary(text, experience):
    lines = text.strip().split("\n")[:5]
    return f"{experience}+ years exp. | {lines[0]}" if lines else f"{experience}+ years exp."

def get_tag(score, exp):
    if score > 90 and exp >= 3:
        return "🔥 Top Talent"
    elif score >= 75:
        return "✅ Good Fit"
    return "⚠️ Needs Review"

def plot_wordcloud(texts):
    text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# --- Screening Logic ---
if jd_text and resume_files:
    results = []
    resume_text_map = {}
    st.info("📁 Starting semantic screening using embeddings...")

    for file in resume_files:
        text = extract_text_from_pdf(file)
        if text.startswith("[ERROR]"):
            st.error(f"Could not process {file.name}")
            continue

        exp = extract_years_of_experience(text)
        email = extract_email(text)
        score = semantic_score(text, jd_text, exp)
        summary = generate_summary(text, exp)

        results.append({
            "File Name": file.name,
            "Score (%)": score,
            "Years Experience": exp,
            "Summary": summary,
            "Email": email or "Not found"
        })
        resume_text_map[file.name] = text

    df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False)
    df['Tag'] = df.apply(lambda row: get_tag(row['Score (%)'], row['Years Experience']), axis=1)
    shortlisted = df[(df['Score (%)'] >= cutoff) & (df['Years Experience'] >= min_experience)]

    st.metric("📊 Avg. Score", f"{df['Score (%)'].mean():.2f}%")
    st.metric("💼 Shortlisted", len(shortlisted))

    st.markdown("### 🏆 Top Candidates")
    for _, row in df.head(3).iterrows():
        st.subheader(f"{row['File Name']} — {row['Score (%)']}%")
        st.text(row['Summary'])
        st.caption(f"Email: {row['Email']}")
        with st.expander("📄 Resume Preview"):
            st.code(resume_text_map.get(row['File Name'], ""))

    st.download_button("📄 Download Results CSV", df.to_csv(index=False).encode("utf-8"), file_name="results.csv")

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
