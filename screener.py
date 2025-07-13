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
import collections
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse # For encoding mailto links
import nltk # For stopwords
import seaborn as sns # For visualizations
import plotly.express as px # For interactive plots

# Download NLTK stopwords data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Import skills data from a separate file ---
# Ensure 'skills_data.py' is in the same directory as this script
from skills_data import ALL_SKILLS_MASTER_SET, SORTED_MASTER_SKILLS, CUSTOM_STOP_WORDS

# --- Stop Words List (Using NLTK) ---
NLTK_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
STOP_WORDS = NLTK_STOP_WORDS.union(CUSTOM_STOP_WORDS)

# Pre-sort MASTER_SKILLS by length descending for efficient multi-word matching
_MASTER_SKILLS_SORTED_BY_LENGTH = sorted(list(ALL_SKILLS_MASTER_SET), key=len, reverse=True)

# Pre-compile regex patterns for MASTER_SKILLS for faster matching
_MASTER_SKILLS_REGEX_PATTERNS = {
    skill_phrase.lower(): re.compile(r'\b' + re.escape(skill_phrase.lower()) + r'\b')
    for skill_phrase in _MASTER_SKILLS_SORTED_BY_LENGTH
}


# --- Load Embedding + ML Model ---
@st.cache_resource
def load_ml_model():
    """Loads the SentenceTransformer model for embeddings and a pre-trained ML screening model."""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Ensure ml_screening_model.pkl is trained with predict_proba capability (e.g., RandomForestClassifier, XGBClassifier)
        # NOTE: The training script saves as 'resume_screening_model.pkl'.
        # You must rename 'resume_screening_model.pkl' to 'ml_screening_model.pkl' in your directory.
        ml_model = joblib.load("ml_screening_model.pkl")
        
        # --- IMPORTANT CHECK FOR predict_proba ---
        if not hasattr(ml_model, 'predict_proba'):
            st.error(f"❌ Loaded ML model ({type(ml_model)}) does not have 'predict_proba' method. Please ensure 'ml_screening_model.pkl' is a classifier trained to output probabilities (e.g., RandomForestClassifier, XGBClassifier).")
            return None, None
        # --- End IMPORTANT CHECK ---

        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory and network is available for SentenceTransformer.")
        return None, None

# Load all models at the start of the app
model, ml_model = load_ml_model()


# --- Helper Functions (Ensuring all are defined in this file) ---

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None
    return text

def clean_text(text):
    """Cleans text by removing special characters, extra spaces, and converting to lowercase."""
    if text is None:
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space
    return text.lower() # Convert to lowercase

def extract_skills(text, job_description_skills=None):
    """
    Extracts skills from text using a master list and job description skills for prioritization.
    Prioritizes multi-word skills and uses a custom stop word list.
    """
    if text is None:
        return set()

    extracted = set()
    text_lower = text.lower()

    # Create a combined set of skills from master list and JD for efficient lookup
    # Prioritize skills from JD if provided, by adding them to the front of the sorted list
    search_skills = SORTED_MASTER_SKILLS
    if job_description_skills:
        jd_skills_lower = {s.lower() for s in job_description_skills}
        # Add JD skills to the search list, ensuring they are also in the master set
        # and sort them by length to prioritize multi-word matches
        jd_specific_sorted_skills = sorted([s for s in jd_skills_lower if s in ALL_SKILLS_MASTER_SET], key=len, reverse=True)
        # Combine, ensuring no duplicates and maintaining priority for JD skills
        search_skills = sorted(list(set(jd_specific_sorted_skills + SORTED_MASTER_SKILLS)), key=len, reverse=True)


    # First pass: Look for exact matches of multi-word skills (longest first)
    # This helps prevent matching "java" if "javascript" is present
    for skill in search_skills:
        if len(skill.split()) > 1: # Only consider multi-word skills here
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                extracted.add(skill)

    # Second pass: Look for single-word skills, avoiding those already found as part of multi-word skills
    # and filtering out common stop words
    words = set(re.findall(r'\b\w+\b', text_lower))
    for word in words:
        if word in ALL_SKILLS_MASTER_SET and word not in CUSTOM_STOP_WORDS:
            is_part_of_multi_word = False
            for multi_word_skill in extracted:
                if len(multi_word_skill.split()) > 1 and word in multi_word_skill.split():
                    is_part_of_multi_word = True
                    break
            if not is_part_of_multi_word:
                extracted.add(word)

    return extracted


def calculate_skill_match(job_skills, resume_skills):
    """Calculates matched and missing skills."""
    matched_skills = job_skills.intersection(resume_skills)
    missing_skills = job_skills.difference(resume_skills)
    return list(matched_skills), list(missing_skills)

def extract_experience(text):
    """
    Extracts total years of experience from text using improved regex patterns.
    Looks for "X years", "X+ years", "X-Y years", "X years of experience", etc.
    """
    if text is None:
        return 0

    text_lower = text.lower()
    total_experience = 0

    # Pattern 1: "X years" or "X+ years" or "X-Y years" where X, Y are numbers
    # Prioritize patterns that explicitly mention "years of experience" or similar
    patterns = [
        r'(\d+\.?\d*)\s*(?:plus|\+)?\s*years\s+of\s+experience',
        r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*years', # e.g., "5-7 years"
        r'(\d+\.?\d*)\s*(?:to|-)\s*(\d+\.?\d*)\s*yrs', # e.g., "5 to 7 yrs"
        r'(\d+\.?\d*)\s*(?:plus|\+)?\s*years?', # e.g., "5 years", "5+ years", "5 yrs"
        r'(\d+\.?\d*)\s*yrs?'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple): # For patterns with groups like "X-Y years"
                try:
                    # Take the upper bound or average for a range
                    exp_start = float(match[0])
                    exp_end = float(match[1]) if len(match) > 1 and match[1] else exp_start
                    total_experience = max(total_experience, exp_end) # Take the higher end of the range
                except ValueError:
                    continue
            else: # For single number patterns
                try:
                    total_experience = max(total_experience, float(match))
                except ValueError:
                    continue
    
    # Heuristic: If "experience" is mentioned multiple times, and a number is near it,
    # but no clear "X years" pattern, this might catch it.
    # This is a fallback and can be less accurate.
    if total_experience == 0:
        exp_keywords = ["experience", "exp"]
        for keyword in exp_keywords:
            # Look for numbers preceding or following the keyword within a small window
            # e.g., "5 years experience", "experience of 3 years"
            match = re.search(r'(\d+\.?\d*)\s*(?:year|yr)s?\s+' + re.escape(keyword), text_lower)
            if match:
                try:
                    total_experience = max(total_experience, float(match.group(1)))
                except ValueError:
                    pass
            match = re.search(re.escape(keyword) + r'\s+(?:of\s+)?(\d+\.?\d*)\s*(?:year|yr)s?', text_lower)
            if match:
                try:
                    total_experience = max(total_experience, float(match.group(1)))
                except ValueError:
                    pass

    return total_experience

def extract_email(text):
    """Extracts an email address from the given text."""
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def extract_name(text):
    """
    Attempts to extract a name from the first few lines of the resume text.
    This is a heuristic and might not be perfect for all resume formats.
    """
    lines = text.strip().split('\n')
    if not lines:
        return None

    potential_name_lines = []
    for line in lines[:3]:
        line = line.strip()
        if not re.search(r'[@\d\.\-]', line) and len(line.split()) <= 4 and (line.isupper() or (line and line[0].isupper() and all(word[0].isupper() or not word.isalpha() for word in line.split()))):
            potential_name_lines.append(line)

    if potential_name_lines:
        name = max(potential_name_lines, key=len)
        name = re.sub(r'summary|education|experience|skills|projects|certifications', '', name, flags=re.IGNORECASE).strip()
        if name:
            return name.title()
    return None

def semantic_score(resume_text, jd_text, years_exp):
    """
    Calculates a semantic score using an ML model and provides additional details.
    Falls back to smart_score if the ML model is not loaded or prediction fails.
    Applies MASTER_SKILLS filtering for keyword analysis (internally, not for display).
    """
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    score = 0.0
    semantic_similarity = 0.0

    if ml_model is None or model is None:
        st.warning("ML models not loaded. Providing basic score.")
        resume_words = extract_skills(resume_clean) # Use extract_skills for consistency
        jd_words = extract_skills(jd_clean) # Use extract_skills for consistency
        
        overlap_count = len(resume_words.intersection(jd_words))
        total_jd_words = len(jd_words)
        
        basic_score = (overlap_count / total_jd_words) * 70 if total_jd_words > 0 else 0
        basic_score += min(years_exp * 5, 30) # Add up to 30 for experience
        score = round(min(basic_score, 100), 2)
        
        return score, 0.0 # Return 0 for semantic similarity if ML not available


    try:
        jd_embed = model.encode(jd_clean)
        resume_embed = model.encode(resume_clean)

        semantic_similarity = cosine_similarity(jd_embed.reshape(1, -1), resume_embed.reshape(1, -1))[0][0]
        semantic_similarity = float(np.clip(semantic_similarity, 0, 1))

        # Internal calculation for model, not for display
        resume_words_filtered = extract_skills(resume_clean)
        jd_words_filtered = extract_skills(jd_clean)
        keyword_overlap_count = len(resume_words_filtered.intersection(jd_words_filtered))
        
        years_exp_for_model = float(years_exp) if years_exp is not None else 0.0

        features = np.concatenate([jd_embed, resume_embed, [years_exp_for_model], [keyword_overlap_count]])

        # The ML model was trained to predict job categories.
        # We'll use its prediction probability for the most likely category as a score indicator.
        # This assumes the model's 'predict_proba' method is well-calibrated for suitability.
        predicted_proba = ml_model.predict_proba(features.reshape(1, -1))
        
        # Find the probability of the most likely class
        max_proba = np.max(predicted_proba)
        
        # Scale this probability to a 0-100 score.
        # This is a heuristic and can be adjusted.
        blended_score = (max_proba * 100 * 0.7) + (semantic_similarity * 100 * 0.3)
        
        score = float(np.clip(blended_score, 0, 100))
        
        return round(score, 2), round(semantic_similarity, 2)

    except Exception as e:
        st.warning(f"Error during semantic scoring, falling back to basic: {e}")
        resume_words = extract_skills(resume_clean)
        jd_words = extract_skills(jd_clean)
        
        overlap_count = len(resume_words.intersection(jd_words))
        total_jd_words = len(jd_words)
        
        basic_score = (overlap_count / total_jd_words) * 70 if total_jd_words > 0 else 0
        basic_score += min(years_exp * 5, 30) # Add up to 30 for experience
        score = round(min(basic_score, 100), 2)

        return score, 0.0 # Return 0 for semantic similarity on fallback

# --- Email Generation Function ---
def create_mailto_link(recipient_email, candidate_name, job_title="Job Opportunity", sender_name="Recruiting Team"):
    """
    Generates a mailto: link with pre-filled subject and body for inviting a candidate.
    """
    subject = urllib.parse.quote(f"Invitation for Interview - {job_title} - {candidate_name}")
    body = urllib.parse.quote(f"""Dear {candidate_name},

We were very impressed with your profile and would like to invite you for an interview for the {job_title} position.

Best regards,

The {sender_name}""")
    return f"mailto:{recipient_email}?subject={subject}&body={body}"

# --- Resume Screener Page ---
def resume_screener_page():
    st.title("🧠 ScreenerPro – AI-Powered Resume Screener")

    # --- Job Description and Controls Section ---
    st.markdown("## ⚙️ Define Job Requirements & Screening Criteria")
    col1, col2 = st.columns([2, 1])

    with col1:
        jd_text = ""
        job_roles = {"Upload my own": None}
        # Check if 'data' directory exists and load sample JDs
        if os.path.exists("data"):
            for fname in os.listdir("data"):
                if fname.endswith(".txt"):
                    job_roles[fname.replace(".txt", "").replace("_", " ").title()] = os.path.join("data", fname)

        jd_option = st.selectbox("📌 **Select a Pre-Loaded Job Role or Upload Your Own Job Description**", list(job_roles.keys()))
        if jd_option == "Upload my own":
            jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], help="Upload a PDF file containing the job description.")
            if jd_file:
                jd_text = extract_text_from_pdf(jd_file)
                if jd_text is None: # Handle PDF extraction errors
                    st.error("Could not extract text from Job Description PDF. Please ensure it's a readable PDF.")
                    jd_text = "" # Clear text to prevent further processing with bad data
        else:
            jd_path = job_roles[jd_option]
            if jd_path and os.path.exists(jd_path):
                with open(jd_path, "r", encoding="utf-8") as f:
                    jd_text = f.read()
        
        if jd_text:
            with st.expander("📝 View Loaded Job Description"):
                st.text_area("Job Description Content", jd_text, height=200, disabled=True, label_visibility="collapsed")

    with col2:
        cutoff = st.slider("📈 **Minimum Score Cutoff (%)**", 0, 100, 75, help="Candidates scoring below this percentage will be flagged for closer review or considered less suitable.")
        st.session_state['screening_cutoff_score'] = cutoff

        min_experience = st.slider("💼 **Minimum Experience Required (Years)**", 0, 15, 2, help="Candidates with less than this experience will be noted.")
        st.session_state['screening_min_experience'] = min_experience

        st.markdown("---")
        st.info("Once criteria are set, upload resumes below to begin screening.")

    resume_files = st.file_uploader("📄 **Upload Resumes (PDF)**", type="pdf", accept_multiple_files=True, help="Upload one or more PDF resumes for screening.")

    if jd_text and resume_files:
        # --- Job Description Keyword Cloud ---
        st.markdown("## ☁️ Job Description Keyword Cloud")
        st.caption("Visualizing the most frequent and important keywords from the Job Description.")
        
        jd_words_for_cloud_set = extract_skills(clean_text(jd_text)) # Use clean_text here

        jd_words_for_cloud = " ".join(list(jd_words_for_cloud_set))

        if jd_words_for_cloud:
            wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(jd_words_for_cloud)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No significant keywords to display for the Job Description. Please ensure your JD has sufficient content or adjust your skills_data.py file.")
        st.markdown("---")

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(resume_files):
            status_text.text(f"Processing {file.name} ({i+1}/{len(resume_files)})...")
            progress_bar.progress((i + 1) / len(resume_files))

            text = extract_text_from_pdf(file)
            if text is None: # Handle case where extract_text_from_pdf returns None
                st.error(f"Failed to process {file.name}: Could not extract text.")
                continue

            exp = extract_experience(text)
            email = extract_email(text)
            candidate_name = extract_name(text) or file.name.replace('.pdf', '').replace('_', ' ').title()

            # Calculate Matched Keywords and Missing Skills
            resume_words_set = extract_skills(clean_text(text)) # Use clean_text here
            jd_words_set = extract_skills(clean_text(jd_text)) # Use clean_text here

            matched_keywords = list(resume_words_set.intersection(jd_words_set))
            missing_skills = list(jd_words_set.difference(resume_words_set)) 

            score, semantic_similarity = semantic_score(text, jd_text, exp)
            
            # Simple AI Suggestion (no T5 model)
            ai_suggestion = f"Candidate scored {score:.2f}% with {exp:.1f} years experience. Semantic similarity: {semantic_similarity:.2f}. "
            if score >= st.session_state['screening_cutoff_score'] and exp >= st.session_state['screening_min_experience']:
                ai_suggestion += "Strong match for the role."
            elif score >= st.session_state['screening_cutoff_score'] * 0.75 or exp >= st.session_state['screening_min_experience'] * 0.75:
                ai_suggestion += "Potential candidate, requires further review."
            else:
                ai_suggestion += "Limited match, consider for other roles or if pipeline is low."


            results.append({
                "File Name": file.name,
                "Candidate Name": candidate_name,
                "Score (%)": score,
                "Years Experience": exp,
                "Email": email or "Not Found",
                "AI Suggestion": ai_suggestion,
                "Matched Keywords": ", ".join(matched_keywords),
                "Missing Skills": ", ".join(missing_skills),
                "Semantic Similarity": semantic_similarity,
                "Resume Raw Text": text
            })
        
        progress_bar.empty()
        status_text.empty()

        df_results = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False).reset_index(drop=True)

        # Store results in session state for the Analytics Dashboard
        st.session_state['screening_results_df'] = df_results
        
        # Add a download button for the results
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        # Tagging logic for display
        df_results['Tag'] = df_results.apply(lambda row: 
            "👑 Exceptional Match" if row['Score (%)'] >= 90 and row['Years Experience'] >= 5 and row['Semantic Similarity'] >= 0.85 else (
            "🔥 Strong Candidate" if row['Score (%)'] >= 80 and row['Years Experience'] >= 3 and row['Semantic Similarity'] >= 0.7 else (
            "✨ Promising Fit" if row['Score (%)'] >= 60 and row['Years Experience'] >= 1 else (
            "⚠️ Needs Review" if row['Score (%)'] >= 40 else 
            "❌ Limited Match"))), axis=1)

        st.markdown("## 📋 Comprehensive Candidate Results Table")
        st.caption("Full details for all processed resumes. **For deep dive analytics and keyword breakdowns, refer to the Analytics Dashboard.**")
        
        comprehensive_cols = [
            'Candidate Name',
            'Score (%)',
            'Years Experience',
            'Semantic Similarity',
            'Tag',
            'Email',
            'AI Suggestion',
            'Matched Keywords',
            'Missing Skills',
        ]
        
        final_display_cols = [col for col in comprehensive_cols if col in df_results.columns]

        st.dataframe(
            df_results[final_display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score (%)": st.column_config.ProgressColumn(
                    "Score (%)",
                    help="Matching score against job requirements",
                    format="%f",
                    min_value=0,
                    max_value=100,
                ),
                "Years Experience": st.column_config.NumberColumn(
                    "Years Experience",
                    help="Total years of professional experience",
                    format="%.1f years",
                ),
                "Semantic Similarity": st.column_config.NumberColumn(
                    "Semantic Similarity",
                    help="Conceptual similarity between JD and Resume (higher is better)",
                    format="%.2f",
                    min_value=0,
                    max_value=1
                ),
                "AI Suggestion": st.column_config.Column(
                    "AI Suggestion",
                    help="AI's concise overall assessment and recommendation"
                ),
                "Matched Keywords": st.column_config.Column(
                    "Matched Keywords",
                    help="Keywords found in both JD and Resume"
                ),
                "Missing Skills": st.column_config.Column(
                    "Missing Skills",
                    help="Key skills from JD not found in Resume"
                ),
            }
        )

        st.info("Remember to check the Analytics Dashboard for in-depth visualizations of skill overlaps, gaps, and other metrics!")
    else:
        st.info("Please upload a Job Description and at least one Resume to begin the screening process.")


# --- Analytics Dashboard Page ---
def analytics_dashboard_page():
    # --- Page Styling ---
    st.markdown("""
    <style>
    .analytics-box {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.96);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        animation: fadeInSlide 0.7s ease-in-out;
        margin-bottom: 2rem;
    }
    @keyframes fadeInSlide {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    h3 {
        color: #00cec9;
        font-weight: 700;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="analytics-box">', unsafe_allow_html=True)
    st.markdown("## 📊 Screening Analytics Dashboard")

    # --- Load Data ---
    if 'screening_results_df' in st.session_state and not st.session_state['screening_results_df'].empty:
        df = st.session_state['screening_results_df']
        st.info("✅ Loaded screening results from current session.")
    else:
        st.warning("⚠️ No screening data found in current session. Please run the screener first on the 'Resume Screener' page.")
        st.markdown("</div>", unsafe_allow_html=True) # Close the analytics box div
        st.stop() # Stop execution if no data is available

    # --- Essential Column Check ---
    essential_core_columns = ['Score (%)', 'Years Experience', 'File Name', 'Candidate Name']

    missing_essential_columns = [col for col in essential_core_columns if col not in df.columns]

    if missing_essential_columns:
        st.error(f"Error: The loaded data is missing essential core columns: {', '.join(missing_essential_columns)}."
                 " Please ensure your screening process generates at least these required data fields.")
        st.markdown("</div>", unsafe_allow_html=True) # Close the analytics box div
        st.stop()

    # --- Filters Section ---
    st.markdown("### 🔍 Filter Results")
    filter_cols = st.columns(3)

    with filter_cols[0]:
        min_score, max_score = float(df['Score (%)'].min()), float(df['Score (%)'].max())
        score_range = st.slider(
            "Filter by Score (%)",
            min_value=min_score,
            max_value=max_score,
            value=(min_score, max_score),
            step=1.0,
            key="analytics_score_filter" # Unique key
        )

    with filter_cols[1]:
        min_exp, max_exp = float(df['Years Experience'].min()), float(df['Years Experience'].max())
        exp_range = st.slider(
            "Filter by Years Experience",
            min_value=min_exp,
            max_value=max_exp,
            value=(min_exp, max_exp),
            step=0.5,
            key="analytics_exp_filter" # Unique key
        )

    with filter_cols[2]:
        # Use a default from session state if available, otherwise a sensible default
        default_shortlist_threshold = st.session_state.get('screening_cutoff_score', 75)
        shortlist_threshold = st.slider(
            "Set Shortlisting Cutoff Score (%)",
            min_value=0,
            max_value=100,
            value=default_shortlist_threshold,
            step=1,
            key="analytics_shortlist_filter" # Unique key
        )

    # Apply filters
    filtered_df = df[
        (df['Score (%)'] >= score_range[0]) & (df['Score (%)'] <= score_range[1]) &
        (df['Years Experience'] >= exp_range[0]) & (df['Years Experience'] <= exp_range[1])
    ].copy()

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your criteria.")
        st.markdown("</div>", unsafe_allow_html=True) # Close the analytics box div
        st.stop()

    # Add Shortlisted/Not Shortlisted column to filtered_df for plotting
    filtered_df['Shortlisted'] = filtered_df['Score (%)'].apply(lambda x: f"Yes (Score >= {shortlist_threshold}%)" if x >= shortlist_threshold else "No")

    # --- Metrics ---
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg. Score", f"{filtered_df['Score (%)'].mean():.2f}%")
    col2.metric("Avg. Experience", f"{filtered_df['Years Experience'].mean():.1f} yrs")
    col3.metric("Total Candidates", f"{len(filtered_df)}")
    shortlisted_count_filtered = (filtered_df['Score (%)'] >= shortlist_threshold).sum()
    col4.metric("Shortlisted", f"{shortlisted_count_filtered}")

    st.divider()

    # --- Detailed Candidate Table ---
    st.markdown("### 📋 Filtered Candidates List")
    display_cols_for_table = ['File Name', 'Candidate Name', 'Score (%)', 'Years Experience', 'Shortlisted']

    if 'Matched Keywords' in filtered_df.columns:
        display_cols_for_table.append('Matched Keywords')
    if 'Missing Skills' in filtered_df.columns:
        display_cols_for_table.append('Missing Skills')
    if 'AI Suggestion' in filtered_df.columns:
        display_cols_for_table.append('AI Suggestion')

    st.dataframe(
        filtered_df[display_cols_for_table].sort_values(by="Score (%)", ascending=False),
        use_container_width=True
    )

    # --- Download Filtered Data ---
    @st.cache_data
    def convert_df_to_csv(df_to_convert):
        return df_to_convert.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(filtered_df)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_screening_results.csv",
        mime="text/csv",
        help="Download the data currently displayed in the table above."
    )

    st.divider()

    # --- Visualizations ---
    st.markdown("### 📊 Visualizations")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Score Distribution", "Experience Distribution", "Shortlist Breakdown", "Score vs. Experience", "Skill Clouds"])

    with tab1:
        st.markdown("#### Score Distribution")
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df['Score (%)'], bins=10, kde=True, color="#00cec9", ax=ax_hist)
        ax_hist.set_xlabel("Score (%)")
        ax_hist.set_ylabel("Number of Candidates")
        st.pyplot(fig_hist)
        plt.close(fig_hist) # Close the figure to free up memory

    with tab2:
        st.markdown("#### Experience Distribution")
        fig_exp, ax_exp = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df['Years Experience'], bins=5, kde=True, color="#fab1a0", ax=ax_exp)
        ax_exp.set_xlabel("Years of Experience")
        ax_exp.set_ylabel("Number of Candidates")
        st.pyplot(fig_exp)
        plt.close(fig_exp) # Close the figure to free up memory

    with tab3:
        st.markdown("#### Shortlist Breakdown")
        shortlist_counts = filtered_df['Shortlisted'].value_counts()
        if not shortlist_counts.empty:
            fig_pie = px.pie(
                names=shortlist_counts.index,
                values=shortlist_counts.values,
                title=f"Candidates Shortlisted vs. Not Shortlisted (Cutoff: {shortlist_threshold}%)",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Not enough data to generate Shortlist Breakdown.")

    with tab4:
        st.markdown("#### Score vs. Years Experience")
        fig_scatter = px.scatter(
            filtered_df,
            x="Years Experience",
            y="Score (%)",
            hover_name="Candidate Name",
            color="Shortlisted",
            title="Candidate Score vs. Years Experience",
            labels={"Years Experience": "Years of Experience", "Score (%)": "Matching Score (%)"},
            trendline="ols",
            color_discrete_map={f"Yes (Score >= {shortlist_threshold}%)": "green", "No": "red"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


    with tab5:
        col_wc1, col_wc2 = st.columns(2)
        with col_wc1:
            st.markdown("#### ☁️ Common Skills WordCloud")
            if 'Matched Keywords' in filtered_df.columns and not filtered_df['Matched Keywords'].empty:
                all_keywords = [
                    kw.strip() for kws in filtered_df['Matched Keywords'].dropna()
                    for kw in str(kws).split(',') if kw.strip()
                ]
                if all_keywords:
                    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_keywords))
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                    plt.close(fig_wc) # Close the figure
                else:
                    st.info("No common skills to display in the WordCloud for filtered data.")
            else:
                st.info("No 'Matched Keywords' data available or column not found for WordCloud.")
            
        with col_wc2:
            st.markdown("#### ❌ Top Missing Skills")
            if 'Missing Skills' in filtered_df.columns and not filtered_df['Missing Skills'].empty:
                all_missing = pd.Series([
                    s.strip() for row in filtered_df['Missing Skills'].dropna()
                    for s in str(row).split(',') if s.strip()
                ])
                if not all_missing.empty:
                    sns.set_style("whitegrid") # Apply style before creating figure
                    fig_ms, ax_ms = plt.subplots(figsize=(8, 4))
                    top_missing = all_missing.value_counts().head(10)
                    sns.barplot(x=top_missing.values, y=top_missing.index, ax=ax_ms, palette="coolwarm")
                    ax_ms.set_xlabel("Count")
                    ax_ms.set_ylabel("Missing Skill")
                    st.pyplot(fig_ms)
                    plt.close(fig_ms) # Close the figure
                else:
                    st.info("No top missing skills to display for filtered data.")
            else:
                st.info("No 'Missing Skills' data available or column not found.")

    st.markdown("</div>", unsafe_allow_html=True)


# --- Main App Logic (Multi-Page Setup) ---
st.set_page_config(layout="wide", page_title="ScreenerPro")

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'screener' # Default page

# Sidebar for navigation
st.sidebar.title("Navigation")
if st.sidebar.button("📄 Resume Screener"):
    st.session_state['page'] = 'screener'
if st.sidebar.button("📊 Analytics Dashboard"):
    st.session_state['page'] = 'analytics'

st.sidebar.markdown("---")
st.sidebar.title("About ScreenerPro")
st.sidebar.info(
    "ScreenerPro is an AI-powered application designed to streamline the resume screening "
    "process. It leverages a custom-trained Machine Learning model and a Sentence Transformer for "
    "semantic understanding.\n\n"
    "Upload job descriptions and resumes, and let AI assist you in identifying the best-fit candidates!"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Developed by [Manav Nagpal](https://www.linkedin.com/in/manav-nagpal-b03a743b/)"
)

# Display the selected page
if st.session_state['page'] == 'screener':
    resume_screener_page()
elif st.session_state['page'] == 'analytics':
    analytics_dashboard_page()
