import streamlit as st
from login import login_section
from email_sender import send_email_to_candidate
import pdfplumber
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Enhanced UI Styling ---
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

st.header("\ud83d\udcc2 Resume Screening")

# --- Job Role Selection ---
job_roles = {"Upload my own": None}
jd_dir = "data"
if os.path.exists(jd_dir):
    for fname in os.listdir(jd_dir):
        if fname.endswith(".txt"):
            job_roles[fname.replace(".txt", "").replace("_", " ").title()] = os.path.join(jd_dir, fname)

jd_option = st.selectbox("\ud83d\udccc Select Job Role or Upload Your Own JD", list(job_roles.keys()))
jd_text = ""
if jd_option == "Upload my own":
    jd_file = st.file_uploader("Upload Job Description (TXT)", type="txt")
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
else:
    jd_path = job_roles[jd_option]
    if jd_path and os.path.exists(jd_path):
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

resume_files = st.file_uploader("\ud83d\uddd5\ufe0f Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True)
cutoff = st.slider("\ud83d\udcc8 Score Cutoff", 0, 100, 80)
min_experience = st.slider("\ud83d\udcbc Minimum Experience Required", 0, 15, 2)

# --- Utility Functions ---
def extract_text_from_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return ''.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        return f"[ERROR] {str(e)}"

def extract_years_of_experience(text):
    text = text.lower()
    patterns = [r'(\d{1,2})\s*\+?\s*(years?|yrs?|year)', r'experience\s*[-:]?\s*(\d{1,2})\s*(years?|yrs?|year)']
    years = []
    for pattern in patterns:
        found = re.findall(pattern, text)
        for match in found:
            years.append(int(match[0]))
    return max(years) if years else 0

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def tfidf_score(resume_text, jd_text, years_exp):
    docs = [jd_text.lower(), resume_text.lower()]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(docs)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    score += min(years_exp, 10)
    return round(min(score, 100), 2)

def smart_score(resume_text, jd_text, years_exp):
    score = tfidf_score(resume_text, jd_text, years_exp)
    feedback = "\u2705 Excellent match!" if score >= 80 else "\u26a0\ufe0f Resume may need more targeted phrasing for this role."
    return score, "Semantic Score", feedback, []


def generate_summary(text, experience, skills):
    return f"{experience}+ years experience in {skills}."[:160]

def get_tag(score, exp):
    if score > 90 and exp >= 3:
        return "🔥 Top Talent"
    elif score >= 75:
        return "✅ Good Fit"
    return "⚠️ Needs Review"

def plot_wordcloud(all_keywords):
    text = ' '.join(all_keywords)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# --- Screening Logic ---
if jd_text and resume_files:
    results = []
    resume_text_map = {}

    upload_folder = "uploaded_resumes"
    os.makedirs(upload_folder, exist_ok=True)

    with st.spinner("🔍 Screening resumes..."):
        for file in resume_files:
            # ✅ Save the uploaded resume
            with open(os.path.join(upload_folder, file.name), "wb") as out_file:
                out_file.write(file.getbuffer())

            resume_text = extract_text_from_pdf(file)
            if resume_text.startswith("[ERROR]"):
                st.error(f"❌ Could not read {file.name}. Skipping.")
                continue
            experience = extract_years_of_experience(resume_text)
            score, matched_keywords, feedback, missing = smart_score(resume_text, jd_text, experience)
            summary = generate_summary(resume_text, experience, matched_keywords)
            email = extract_email(resume_text)
            results.append({
                "File Name": file.name,
                "Score (%)": score,
                "Years Experience": experience,
                "Matched Keywords": matched_keywords,
                "Missing Skills": ", ".join(missing),
                "Feedback": feedback,
                "Summary": summary,
                "Email": email or "Not found"
            })
            resume_text_map[file.name] = resume_text

    if results:
        df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False)
        df['Tag'] = df.apply(lambda row: get_tag(row['Score (%)'], row['Years Experience']), axis=1)
        shortlisted = df[(df['Score (%)'] >= cutoff) & (df['Years Experience'] >= min_experience)]

        # --- Insights Section ---
        st.success("🎉 Screening completed. Review the results below.")
        core_skills = ['python', 'java', 'sql', 'html', 'css', 'javascript', 'react', 'machine', 'learning']
        total_covered_skills = sum(1 for skill in core_skills if any(skill in mk for mk in df['Matched Keywords']))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📈 Avg. Score", f"{df['Score (%)'].mean():.2f}%")
        col2.metric("🧠 Core Skills Matched", f"{total_covered_skills}/9")
        col3.metric("📄 Total Resumes", len(df))
        col4.metric("✅ Shortlisted", len(shortlisted))

        # --- Wordcloud ---
        st.markdown("### ☁️ Word Cloud of Matched Skills")
        all_keywords = []
        for kw in df['Matched Keywords']:
            all_keywords.extend(kw.split(', '))
        plot_wordcloud(all_keywords)

        # --- Top Candidates ---
        st.markdown("### 🏅 Top Candidates")
        top3 = df.head(3)
        for _, row in top3.iterrows():
            st.markdown(f"**{row['File Name']}** — Score: {row['Score (%)']}% | Exp: {row['Years Experience']} yrs")
            st.markdown(f"📝 Summary: {row['Summary']}")
            st.caption(f"Matched Skills: {row['Matched Keywords']}")
            st.info(row['Feedback'])
            with st.expander("📄 Resume Preview"):
                st.code(resume_text_map.get(row['File Name'], ""))
            st.divider()

        # --- CSV Download ---
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download Results CSV", data=csv, file_name="results.csv", mime="text/csv")

        # --- Emailing Section ---
        st.markdown("### ✉️ Send Emails to Shortlisted Candidates")
        st.dataframe(shortlisted, use_container_width=True)

        email_ready = shortlisted[shortlisted["Email"].str.contains("@", na=False)]
        if not email_ready.empty:
            subject = st.text_input("Subject", value="🎉 You've been shortlisted!")
            body_template = st.text_area("Email Body", height=220, value="""
Dear {{name}},

Congratulations! You have been shortlisted after a thorough resume screening.

🧠 Score: {{score}}%
💬 Feedback: {{feedback}}

We’ll contact you with further steps.

Regards,  
HR Team — Shree Ram Recruitments
""")

            for _, row in email_ready.iterrows():
                preview = body_template.replace("{{name}}", row["File Name"].replace(".pdf", "")).replace("{{score}}", str(row["Score (%)"])).replace("{{feedback}}", row["Feedback"])
                with st.expander(f"📧 Preview Email for {row['File Name']} ({row['Email']})"):
                    st.markdown(f"<div style='background:#f9f9ff;padding:15px;border-left:4px solid #00cec9;border-radius:10px;'>{preview}</div>", unsafe_allow_html=True)

            if st.button("📤 Send Emails Now", type="primary"):
                for _, row in email_ready.iterrows():
                    msg = body_template.replace("{{name}}", row["File Name"].replace(".pdf", "")).replace("{{score}}", str(row["Score (%)"])).replace("{{feedback}}", row["Feedback"])
                    send_email_to_candidate(
                        name=row["File Name"],
                        score=row["Score (%)"],
                        feedback=row["Feedback"],
                        recipient=row["Email"],
                        subject=subject,
                        message=msg
                    )
                st.success("✅ Emails sent to all shortlisted candidates!")
        else:
            st.warning("⚠️ No valid emails found for shortlisted candidates.")
    else:
        st.warning("⚠️ No resumes were successfully processed.")
