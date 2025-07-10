import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
import sklearn

import joblib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer

# --- External Dependencies (Ensure these files exist in your environment) ---
# from email_sender import send_email_to_candidate
# from login import login_section

# Placeholder for send_email_to_candidate if the file is not available
def send_email_to_candidate(name, score, feedback, recipient, subject, message):
    st.info(f"Simulating email send to {recipient} (Name: {name}, Score: {score}%, Feedback: {feedback})")
    st.info(f"Subject: {subject}\nMessage: {message}")
    # In a real application, you would integrate your email sending logic here
    pass

# Placeholder for login_section if the file is not available
def login_section():
    # In a real application, you would have your login logic here
    st.sidebar.success("Login section placeholder.")
    return True # Assume logged in for demonstration


# --- Load Embedding + ML Model ---
model = SentenceTransformer("all-MiniLM-L6-v2")
try:
    ml_model = joblib.load("ml_screening_model.pkl")
    st.success("✅ ML model loaded successfully")
except Exception as e:
    st.error(f"❌ Could not load model: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory.")
    ml_model = None

st.info(f"📦 Loaded model: {type(ml_model).__name__} | sklearn: {sklearn.__version__}")

# --- Helpers ---
def clean_text(text):
    """Cleans text by removing newlines, extra spaces, and non-ASCII characters."""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip().lower()

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return ''.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        return f"[ERROR] {str(e)}"

def extract_years_of_experience(text):
    """Extracts years of experience from a given text by parsing date ranges or keywords."""
    text = text.lower()
    total_months = 0
    # Regex to find date ranges like "Jan 2020 - Present" or "January 2018 to Dec 2022"
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
        # Fallback to direct experience keywords if date ranges are not found
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\+)?\s*(year|yrs|years)\b', text)
        if not match:
            match = re.search(r'experience[^\d]{0,10}(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

    return round(total_months / 12, 1)

def extract_email(text):
    """Extracts an email address from the given text."""
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def smart_score(resume_text, jd_text, years_exp):
    """
    Calculates a 'smart score' based on keyword overlap and experience.
    Also identifies matched and missing keywords, and provides simple feedback.
    """
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    jd_words = set(re.findall(r'\b\w+\b', jd_text.lower()))

    # Calculate overlap
    overlap = resume_words & jd_words
    matched_keywords = ", ".join(sorted(list(overlap))) # Sort for consistency

    # Identify missing skills from JD that are not in resume
    missing_skills_set = jd_words - resume_words
    missing_skills = ", ".join(sorted(list(missing_skills_set))) # Sort for consistency

    # Score calculation logic
    base_score = min(len(overlap), 25) * 3 # Max 75 points from keywords
    experience_score = min(years_exp, 10) # Max 10 points from experience
    score = base_score + experience_score
    score = round(min(score, 100), 2) # Cap score at 100

    # Simple feedback generation
    feedback = "Good keyword match and experience."
    if score < 50:
        feedback = "Low keyword match. Consider reviewing the resume for relevance."
    elif years_exp < 2:
        feedback = "Good keyword match but experience is on the lower side."
    elif not matched_keywords:
        feedback = "Very few common keywords found. Significant mismatch."

    return score, matched_keywords, missing_skills, feedback

def semantic_score(resume_text, jd_text, years_exp):
    """
    Calculates a semantic score using an ML model and provides additional details.
    Falls back to smart_score if the ML model is not loaded or prediction fails.
    """
    st.warning("⚙️ semantic_score() function triggered")

    # Clean text using the global clean_text function
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    # Initialize return values
    score = 0.0
    matched_keywords = ""
    missing_skills = ""
    feedback = "Initial assessment."

    # If ML model is not loaded, fall back to smart_score
    if ml_model is None:
        st.error("❌ ML model not loaded. Falling back to smart_score for all metrics.")
        score, matched_keywords, missing_skills, feedback = smart_score(resume_text, jd_text, years_exp)
        return score, matched_keywords, missing_skills, feedback

    try:
        # Generate embeddings for JD and resume
        jd_embed = model.encode(jd_clean)
        resume_embed = model.encode(resume_clean)

        # Extract numerical features for the ML model
        resume_words = set(re.findall(r'\b\w+\b', resume_clean))
        jd_words = set(re.findall(r'\b\w+\b', jd_clean))
        keyword_overlap_count = len(resume_words & jd_words)
        resume_len = len(resume_clean.split())

        core_skills = ['sql', 'excel', 'python', 'tableau', 'powerbi', 'r', 'aws']
        matched_core_skills_count = sum(1 for skill in core_skills if skill in resume_clean)

        extra_feats = np.array([keyword_overlap_count, resume_len, matched_core_skills_count])

        # Concatenate all features for the ML model
        features = np.concatenate([jd_embed, resume_embed, extra_feats])

        st.info(f"🔍 Feature shape: {features.shape}")

        # Predict score using the loaded ML model
        predicted_score = ml_model.predict([features])[0]
        st.success(f"🧠 Predicted score: {predicted_score:.2f}")

        score = float(np.clip(predicted_score, 0, 100)) # Ensure score is between 0 and 100

        # Calculate matched and missing keywords for display, even if ML model is used
        overlap_words_set = resume_words & jd_words
        matched_keywords = ", ".join(sorted(list(overlap_words_set)))
        missing_skills_set = jd_words - resume_words
        missing_skills = ", ".join(sorted(list(missing_skills_set)))

        # Generate feedback based on ML score
        if score > 90:
            feedback = "Excellent fit based on semantic analysis and ML prediction."
        elif score >= 75:
            feedback = "Good fit, strong semantic match and ML prediction."
        elif score >= 50:
            feedback = "Moderate fit. Some areas for improvement based on ML prediction."
        else:
            feedback = "Lower semantic match. Consider reviewing for relevance."

        # Original fallback logic: if ML score is too low, use smart_score to be more robust
        if score < 10:
            st.warning("⚠️ ML score is very low. Using fallback smart_score for a more reliable assessment.")
            score, matched_keywords, missing_skills, feedback = smart_score(resume_text, jd_text, years_exp)

        return round(score, 2), matched_keywords, missing_skills, feedback

    except Exception as e:
        st.error(f"❌ semantic_score failed during prediction: {e}. Falling back to smart_score.")
        # Fallback to smart_score if any error occurs during ML prediction
        score, matched_keywords, missing_skills, feedback = smart_score(resume_text, jd_text, years_exp)
        return score, matched_keywords, missing_skills, feedback

# --- Streamlit UI ---
st.title("🧠 ScreenerPro – AI Resume Screener")

# Login section (if enabled)
# if not login_section():
#     st.stop() # Stop execution if not logged in

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

df = pd.DataFrame() # Initialize DataFrame outside the if block

if jd_text and resume_files:
    results = []
    resume_text_map = {}
    for file in resume_files:
        text = extract_text_from_pdf(file)
        if text.startswith("[ERROR]"):
            st.error(f"Could not process {file.name}: {text}")
            continue

        exp = extract_years_of_experience(text)
        email = extract_email(text)
        # Call semantic_score and unpack all returned values
        score, matched_keywords, missing_skills, feedback = semantic_score(text, jd_text, exp)
        summary = f"{exp}+ years exp. | {text.strip().splitlines()[0]}" if text else f"{exp}+ years exp."

        results.append({
            "File Name": file.name,
            "Score (%)": score,
            "Years Experience": exp,
            "Summary": summary,
            "Email": email or "Not found",
            "Matched Keywords": matched_keywords, # New field
            "Missing Skills": missing_skills,     # New field
            "Feedback": feedback                 # New field
        })
        resume_text_map[file.name] = text

    df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False)

    # === 📊 Global Resume Screening Insights ===
    st.markdown("## 📊 Screening Insights & Summary")

    avg_score = df['Score (%)'].mean()
    avg_exp = df['Years Experience'].mean()

    # Aggregate top matched skills across all resumes
    all_matched_skills = []
    for keywords in df['Matched Keywords'].dropna():
        all_matched_skills.extend([skill.strip().lower() for skill in keywords.split(',') if skill.strip()])
    top_matched_skills = pd.Series(all_matched_skills).value_counts().head(5)

    # Aggregate top missing skills across all resumes
    all_missing_skills = []
    for skills in df['Missing Skills'].dropna():
        all_missing_skills.extend([skill.strip().lower() for skill in skills.split(',') if skill.strip()])
    top_missing_skills = pd.Series(all_missing_skills).value_counts().head(5)


    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Average Score", f"{avg_score:.2f}%")
        st.metric("🧠 Avg. Experience", f"{avg_exp:.1f} yrs")

    with col2:
        st.markdown("### 🔝 Top 5 Matched Skills")
        if not top_matched_skills.empty:
            for skill, count in top_matched_skills.items():
                st.markdown(f"- ✅ {skill} ({count})")
        else:
            st.info("No common matched skills found.")

        st.markdown("### ❌ Top 5 Missing Skills")
        if not top_missing_skills.empty:
            for skill, count in top_missing_skills.items():
                st.markdown(f"- ⚠️ {skill} ({count})")
        else:
            st.info("No common missing skills found.")

    st.divider()

    # === 🏆 Show Top 3 Candidates ===
    st.markdown("## 🏆 Top Candidates")
    # Ensure this uses the full sorted DataFrame for top candidates
    top_candidates_display = df.head(3)

    if not top_candidates_display.empty:
        for _, row in top_candidates_display.iterrows():
            st.subheader(f"{row['File Name']} — {row['Score (%)']}%")
            st.write(f"🧠 Experience: {row['Years Experience']} years")
            st.caption(f"Matched Skills: {row['Matched Keywords']}")
            st.caption(f"Missing: {row['Missing Skills']}")
            st.info(f"📋 Feedback: {row['Feedback']}")
            with st.expander("📄 Resume Preview"):
                st.code(resume_text_map.get(row['File Name'], ''))
    else:
        st.info("No candidates to display yet.")

    st.divider()

    # Add a 'Tag' column for quick categorization
    df['Tag'] = df.apply(lambda row: "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3 else (
        "✅ Good Fit" if row['Score (%)'] >= 75 else "⚠️ Needs Review"), axis=1)

    shortlisted = df[(df['Score (%)'] >= cutoff) & (df['Years Experience'] >= min_experience)]

    st.metric("✅ Shortlisted Candidates", len(shortlisted))

    st.markdown("### 📋 All Candidate Results")
    st.dataframe(df) # Display the full DataFrame

    st.download_button(
        "📥 Download Results CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="screener_results.csv",
        mime="text/csv"
    )

    st.markdown("### ✉️ Send Emails to Shortlisted Candidates")
    email_ready = shortlisted[shortlisted['Email'].str.contains("@", na=False)]

    if email_ready.empty:
        st.info("No shortlisted candidates with valid emails to send emails to.")
    else:
        st.write(f"Found {len(email_ready)} shortlisted candidates with valid emails.")
        subject = st.text_input("Email Subject", "You're Shortlisted - Next Steps for [Job Role]")
        body = st.text_area("Email Body", """
Dear {{name}},

We are pleased to inform you that you've been shortlisted for the next round for the [Job Role] position.
Your profile scored {{score}}% in our AI-powered screening system.

We will be in touch shortly with the next steps.

Regards,
Recruitment Team
""")

        if st.button("📧 Send Emails"):
            for _, row in email_ready.iterrows():
                # Replace placeholders in the email body
                msg = body.replace("{{name}}", row['File Name'].replace(".pdf", "").split('_')[0].title())\
                            .replace("{{score}}", str(round(row['Score (%)'], 2)))
                # Assuming `jd_option` holds the job role name
                msg = msg.replace("[Job Role]", jd_option if jd_option != "Upload my own" else "the specified role")

                send_email_to_candidate(
                    name=row["File Name"].replace(".pdf", ""),
                    score=row["Score (%)"],
                    feedback=row["Tag"],
                    recipient=row["Email"],
                    subject=subject,
                    message=msg
                )
            st.success("✅ Emails sent to shortlisted candidates!")
else:
    if not jd_text:
        st.warning("Please upload a Job Description or select a predefined job role.")
    if not resume_files:
        st.warning("Please upload one or more resumes.")
