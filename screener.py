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
import nltk
import collections
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    # st.success("✅ NLTK stopwords downloaded successfully!") # Removed debug

# --- External Dependencies (Placeholders for demonstration) ---
def send_email_to_candidate(name, score, feedback, recipient, subject, message):
    # st.info(f"Simulating email send to {recipient} (Name: {name}, Score: {score}%, Feedback: {feedback})") # Removed debug
    # st.info(f"Subject: {subject}\nMessage: {message}") # Removed debug
    pass

def login_section():
    # st.sidebar.success("Login section placeholder.") # Removed debug
    return True # Assume logged in for demonstration

# --- Load Embedding + ML Model ---
@st.cache_resource # Cache the model loading for better performance
def load_ml_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ml_model = joblib.load("ml_screening_model.pkl")
        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory.")
        return None, None

model, ml_model = load_ml_model()

# st.info(f"📦 Loaded model: {type(ml_model).__name__} | sklearn: {sklearn.__version__}") # Removed debug

# --- Stop Words List (Using NLTK) ---
NLTK_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
CUSTOM_STOP_WORDS = set([
    "work", "experience", "years", "year", "months", "month", "day", "days", "project", "projects",
    "team", "teams", "developed", "managed", "led", "created", "implemented", "designed",
    "responsible", "proficient", "knowledge", "ability", "strong", "proven", "demonstrated",
    "solution", "solutions", "system", "systems", "platform", "platforms", "framework", "frameworks",
    "database", "databases", "server", "servers", "cloud", "computing", "machine", "learning",
    "artificial", "intelligence", "api", "apis", "rest", "graphql", "agile", "scrum", "kanban",
    "devops", "ci", "cd", "testing", "qa",
    "security", "network", "networking", "virtualization",
    "containerization", "docker", "kubernetes", "git", "github", "gitlab", "bitbucket", "jira",
    "confluence", "slack", "microsoft", "google", "amazon", "azure", "oracle", "sap", "crm", "erp",
    "salesforce", "servicenow", "tableau", "powerbi", "qlikview", "excel", "word", "powerpoint",
    "outlook", "visio", "html", "css", "js", "web", "data", "science", "analytics", "engineer",
    "software", "developer", "analyst", "business", "management", "reporting", "analysis", "tools",
    "python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "swift", "kotlin", "r",
    "sql", "nosql", "linux", "unix", "windows", "macos", "ios", "android", "mobile", "desktop",
    "application", "applications", "frontend", "backend", "fullstack", "ui", "ux", "design",
    "architecture", "architect", "engineering", "scientist", "specialist", "consultant",
    "associate", "senior", "junior", "lead", "principal", "director", "manager", "head", "chief",
    "officer", "president", "vice", "executive", "ceo", "cto", "cfo", "coo", "hr", "human",
    "resources", "recruitment", "talent", "acquisition", "onboarding", "training", "development",
    "performance", "compensation", "benefits", "payroll", "compliance", "legal", "finance",
    "accounting", "auditing", "tax", "budgeting", "forecasting", "investments", "marketing",
    "sales", "customer", "service", "support", "operations", "supply", "chain", "logistics",
    "procurement", "manufacturing", "production", "quality", "assurance", "control", "research",
    "innovation", "product", "program", "portfolio", "governance", "risk", "communication",
    "presentation", "negotiation", "problem", "solving", "critical", "thinking", "analytical",
    "creativity", "adaptability", "flexibility", "teamwork", "collaboration", "interpersonal",
    "organizational", "time", "multitasking", "detail", "oriented", "independent", "proactive",
    "self", "starter", "results", "driven", "client", "facing", "stakeholder", "engagement",
    "vendor", "budget", "cost", "reduction", "process", "improvement", "standardization",
    "optimization", "automation", "digital", "transformation", "change", "methodologies",
    "industry", "regulations", "regulatory", "documentation", "technical", "writing",
    "dashboards", "visualizations", "workshops", "feedback", "reviews", "appraisals",
    "offboarding", "employee", "relations", "diversity", "inclusion", "equity", "belonging",
    "corporate", "social", "responsibility", "csr", "sustainability", "environmental", "esg",
    "ethics", "integrity", "professionalism", "confidentiality", "discretion", "accuracy",
    "precision", "efficiency", "effectiveness", "scalability", "robustness", "reliability",
    "vulnerability", "assessment", "penetration", "incident", "response", "disaster",
    "recovery", "continuity", "bcp", "drp", "gdpr", "hipaa", "soc2", "iso", "nist", "pci",
    "dss", "ccpa", "privacy", "protection", "grc", "cybersecurity", "information", "infosec",
    "threat", "intelligence", "soc", "event", "siem", "identity", "access", "iam", "privileged",
    "pam", "multi", "factor", "authentication", "mfa", "single", "sign", "on", "sso",
    "encryption", "decryption", "firewall", "ids", "ips", "vpn", "endpoint", "antivirus",
    "malware", "detection", "forensics", "handling", "assessments", "policies", "procedures",
    "guidelines", "mitre", "att&ck", "modeling", "secure", "lifecycle", "sdlc", "awareness",
    "phishing", "vishing", "smishing", "ransomware", "spyware", "adware", "rootkits",
    "botnets", "trojans", "viruses", "worms", "zero", "day", "exploits", "patches", "patching",
    "updates", "upgrades", "configuration", "ticketing", "crm", "erp", "scm", "hcm", "financial",
    "accounting", "bi", "warehousing", "etl", "extract", "transform", "load", "lineage",
    "master", "mdm", "lakes", "marts", "big", "hadoop", "spark", "kafka", "flink", "mongodb",
    "cassandra", "redis", "elasticsearch", "relational", "mysql", "postgresql", "db2",
    "teradata", "snowflake", "redshift", "synapse", "bigquery", "aurora", "dynamodb",
    "documentdb", "cosmosdb", "graph", "neo4j", "graphdb", "timeseries", "influxdb",
    "timescaledb", "columnar", "vertica", "clickhouse", "vector", "pinecone", "weaviate",
    "milvus", "qdrant", "chroma", "faiss", "annoy", "hnswlib", "scikit", "learn", "tensorflow",
    "pytorch", "keras", "xgboost", "lightgbm", "catboost", "statsmodels", "numpy", "pandas",
    "matplotlib", "seaborn", "plotly", "bokeh", "dash", "flask", "django", "fastapi", "spring",
    "boot", ".net", "core", "node.js", "express.js", "react", "angular", "vue.js", "svelte",
    "jquery", "bootstrap", "tailwind", "sass", "less", "webpack", "babel", "npm", "yarn",
    "ansible", "terraform", "jenkins", "gitlab", "github", "actions", "codebuild", "codepipeline",
    "codedeploy", "build", "deploy", "run", "lambda", "functions", "serverless", "microservices",
    "gateway", "mesh", "istio", "linkerd", "grpc", "restful", "soap", "message", "queues",
    "rabbitmq", "activemq", "bus", "sqs", "sns", "pubsub", "version", "control", "svn",
    "mercurial", "trello", "asana", "monday.com", "smartsheet", "project", "primavera",
    "zendesk", "freshdesk", "itil", "cobit", "prince2", "pmp", "master", "owner", "lean",
    "six", "sigma", "black", "belt", "green", "yellow", "qms", "9001", "27001", "14001",
    "ohsas", "18001", "sa", "8000", "cmii", "cmi", "cism", "cissp", "ceh", "comptia",
    "security+", "network+", "a+", "linux+", "ccna", "ccnp", "ccie", "certified", "solutions",
    "architect", "developer", "sysops", "administrator", "specialty", "professional", "azure",
    "az-900", "az-104", "az-204", "az-303", "az-304", "az-400", "az-500", "az-700", "az-800",
    "az-801", "dp-900", "dp-100", "dp-203", "ai-900", "ai-102", "da-100", "pl-900", "pl-100",
    "pl-200", "pl-300", "pl-400", "pl-500", "ms-900", "ms-100", "ms-101", "ms-203", "ms-500",
    "ms-700", "ms-720", "ms-740", "ms-600", "sc-900", "sc-200", "sc-300", "sc-400", "md-100",
    "md-101", "mb-200", "mb-210", "mb-220", "mb-230", "mb-240", "mb-260", "mb-300", "mb-310",
    "mb-320", "mb-330", "mb-340", "mb-400", "mb-500", "mb-600", "mb-700", "mb-800", "mb-910",
    "mb-920", "gcp-ace", "gcp-pca", "gcp-pde", "gcp-pse", "gcp-pml", "gcp-psa", "gcp-pcd",
    "gcp-pcn", "gcp-psd", "gcp-pda", "gcp-pci", "gcp-pws", "gcp-pwa", "gcp-pme", "gcp-pms",
    "gcp-pmd", "gcp-pma", "gcp-pmc", "gcp-pmg", "cisco", "juniper", "red", "hat", "rhcsa",
    "rhce", "vmware", "vcpa", "vcpd", "vcpi", "vcpe", "vcpx", "citrix", "cc-v", "cc-p",
    "cc-e", "cc-m", "cc-s", "cc-x", "palo", "alto", "pcnsa", "pcnse", "fortinet", "fcsa",
    "fcsp", "fcc", "fcnsp", "fct", "fcp", "fcs", "fce", "fcn", "fcnp", "fcnse"
])
STOP_WORDS = NLTK_STOP_WORDS.union(CUSTOM_STOP_WORDS)

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

def get_top_keywords(text, num_keywords=15):
    """Extracts and returns the top N most frequent keywords from text, excluding stop words."""
    cleaned_text = clean_text(text)
    words = [word for word in re.findall(r'\b\w+\b', cleaned_text) if word not in STOP_WORDS]
    word_counts = collections.Counter(words)
    return [word for word, count in word_counts.most_common(num_keywords)]

def smart_score(resume_text, jd_text, years_exp):
    """
    Calculates a 'smart score' based on keyword overlap and experience.
    Also identifies matched and missing keywords, and provides simple feedback.
    Applies STOP_WORDS filtering for keyword analysis.
    """
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    resume_words = {word for word in re.findall(r'\b\w+\b', resume_clean) if word not in STOP_WORDS}
    jd_words = {word for word in re.findall(r'\b\w+\b', jd_clean) if word not in STOP_WORDS}

    overlap = resume_words & jd_words
    matched_keywords = ", ".join(sorted(list(overlap)))

    missing_skills_set = jd_words - resume_words
    missing_skills = ", ".join(sorted(list(missing_skills_set)))

    keyword_overlap_count = len(overlap)
    if len(jd_words) > 0:
        jd_coverage_percentage = (keyword_overlap_count / len(jd_words)) * 100
    else:
        jd_coverage_percentage = 0.0

    base_score = min(len(overlap), 25) * 3
    experience_score = min(years_exp, 10)
    score = base_score + experience_score
    score = round(min(score, 100), 2)

    feedback = "Good keyword match and experience."
    if score < 50:
        feedback = "The resume has limited keyword alignment with the job description. Review closely for transferable skills."
    elif years_exp < 2:
        feedback = "Strong keyword match, though the candidate's experience level is at the lower end of expectations."
    elif not matched_keywords:
        feedback = "Very few common keywords found. The profile seems to be a significant mismatch."

    return score, matched_keywords, missing_skills, feedback, 0.0, round(jd_coverage_percentage, 2)


def semantic_score(resume_text, jd_text, years_exp):
    """
    Calculates a semantic score using an ML model and provides additional details.
    Falls back to smart_score if the ML model is not loaded or prediction fails.
    Applies STOP_WORDS filtering for keyword analysis before display.
    """
    # st.warning("⚙️ semantic_score() function triggered") # Removed debug

    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    score = 0.0
    matched_keywords = ""
    missing_skills = ""
    feedback = "Initial assessment."
    semantic_similarity = 0.0
    jd_coverage_percentage = 0.0

    if ml_model is None or model is None:
        # st.error("❌ ML model not loaded. Falling back to smart_score for all metrics.") # Removed debug
        score, matched_keywords, missing_skills, feedback, semantic_similarity, jd_coverage_percentage = smart_score(resume_text, jd_text, years_exp)
        return score, matched_keywords, missing_skills, feedback, semantic_similarity, jd_coverage_percentage

    try:
        jd_embed = model.encode(jd_clean)
        resume_embed = model.encode(resume_clean)

        semantic_similarity = cosine_similarity(jd_embed.reshape(1, -1), resume_embed.reshape(1, -1))[0][0]
        semantic_similarity = float(np.clip(semantic_similarity, 0, 1))

        resume_words_all = set(re.findall(r'\b\w+\b', resume_clean))
        jd_words_all = set(re.findall(r'\b\w+\b', jd_clean))

        resume_words_filtered = {word for word in resume_words_all if word not in STOP_WORDS}
        jd_words_filtered = {word for word in jd_words_all if word not in STOP_WORDS}

        keyword_overlap_count = len(resume_words_filtered & jd_words_filtered)
        
        years_exp_for_model = float(years_exp) if years_exp is not None else 0.0

        features = np.concatenate([jd_embed, resume_embed, [years_exp_for_model], [keyword_overlap_count]])

        # st.info(f"🔍 Feature shape: {features.shape}") # Removed debug

        predicted_score = ml_model.predict([features])[0]
        # st.success(f"🧠 Predicted score (ML base): {predicted_score:.2f}") # Removed debug

        if len(jd_words_filtered) > 0:
            jd_coverage_percentage = (keyword_overlap_count / len(jd_words_filtered)) * 100
        else:
            jd_coverage_percentage = 0.0

        blended_score = (predicted_score * 0.6) + \
                        (jd_coverage_percentage * 0.1) + \
                        (semantic_similarity * 100 * 0.3)

        if semantic_similarity > 0.7 and years_exp >= 3:
            blended_score += 5

        score = float(np.clip(blended_score, 0, 100))

        overlap_words_set_display = resume_words_filtered & jd_words_filtered
        matched_keywords = ", ".join(sorted(list(overlap_words_set_display)))
        missing_skills_set_display = jd_words_filtered - resume_words_filtered
        missing_skills = ", ".join(sorted(list(missing_skills_set_display)))

        if score > 90:
            feedback = "Excellent fit: Outstanding alignment with job requirements, high keyword coverage, and strong relevant experience."
        elif score >= 75:
            feedback = "Good fit: Solid alignment with the role, good keyword coverage, and relevant experience demonstrated."
        elif score >= 60:
            feedback = "Moderate fit: Decent potential, but areas for improvement in specific keyword alignment or deeper experience matching."
        else:
            # Enhanced feedback for lower fit
            feedback = "Initial review suggests this profile may require closer manual review. There are some gaps in direct keyword relevance or semantic alignment, and the overall fit score is lower. Consider if transferable skills or other experiences not immediately captured by the automated system might still make this candidate suitable."

        if score < 10:
            # st.warning("⚠️ Blended score is very low. Using fallback smart_score for a more reliable assessment.") # Removed debug
            score, matched_keywords, missing_skills, feedback, semantic_similarity, jd_coverage_percentage = smart_score(resume_text, jd_text, years_exp)

        return round(score, 2), matched_keywords, missing_skills, feedback, round(semantic_similarity, 2), round(jd_coverage_percentage, 2)

    except Exception as e:
        # st.error(f"❌ semantic_score failed during prediction: {e}. Falling back to smart_score.") # Removed debug
        score, matched_keywords, missing_skills, feedback, semantic_similarity, jd_coverage_percentage = smart_score(resume_text, jd_text, years_exp)
        return score, matched_keywords, missing_skills, feedback, semantic_similarity, jd_coverage_percentage


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="ScreenerPro - AI Resume Screener", page_icon="🧠")
st.title("🧠 ScreenerPro – AI Resume Screener")

# Login section (if enabled)
# if not login_section():
#    st.stop()

# --- Job Description and Controls Section ---
st.markdown("## ⚙️ Setup Job Description & Screening Criteria")
col1, col2 = st.columns([2, 1])

with col1:
    jd_text = ""
    job_roles = {"Upload my own": None}
    if os.path.exists("data"):
        for fname in os.listdir("data"):
            if fname.endswith(".txt"):
                job_roles[fname.replace(".txt", "").replace("_", " ").title()] = os.path.join("data", fname)

    jd_option = st.selectbox("📌 **Select Job Role or Upload Your Own JD**", list(job_roles.keys()))
    if jd_option == "Upload my own":
        jd_file = st.file_uploader("Upload Job Description (TXT)", type="txt")
        if jd_file:
            jd_text = jd_file.read().decode("utf-8")
    else:
        jd_path = job_roles[jd_option]
        if jd_path and os.path.exists(jd_path):
            with open(jd_path, "r", encoding="utf-8") as f:
                jd_text = f.read()
    
    if jd_text:
        with st.expander("📝 View Loaded Job Description"):
            st.text_area("Job Description Content", jd_text, height=200, disabled=True)

with col2:
    cutoff = st.slider("📈 **Minimum Score Cutoff (%)**", 0, 100, 75) # Adjusted default for better filtering
    min_experience = st.slider("💼 **Minimum Experience Required (Years)**", 0, 15, 2)
    st.markdown("---")
    st.info("Upload resumes to start the screening process.")

resume_files = st.file_uploader("📄 **Upload Resumes (PDF)**", type="pdf", accept_multiple_files=True)

df = pd.DataFrame()

if jd_text and resume_files:
    # --- Job Description Keyword Cloud ---
    st.markdown("---")
    st.markdown("## ☁️ Job Description Keyword Cloud")
    jd_words_for_cloud = " ".join([word for word in re.findall(r'\b\w+\b', clean_text(jd_text)) if word not in STOP_WORDS])
    if jd_words_for_cloud:
        wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(jd_words_for_cloud)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("No significant keywords to display for the Job Description after filtering common words. Please ensure your JD has sufficient content.")
    st.markdown("---")

    results = []
    resume_text_map = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(resume_files):
        status_text.text(f"Processing {file.name} ({i+1}/{len(resume_files)})...")
        progress_bar.progress((i + 1) / len(resume_files))

        text = extract_text_from_pdf(file)
        if text.startswith("[ERROR]"):
            st.error(f"Failed to process {file.name}: {text.replace('[ERROR] ', '')}")
            continue

        exp = extract_years_of_experience(text)
        email = extract_email(text)
        candidate_name = extract_name(text) or file.name.replace('.pdf', '').replace('_', ' ').title()

        score, matched_keywords, missing_skills, feedback, semantic_similarity, jd_coverage_percentage = semantic_score(text, jd_text, exp)

        results.append({
            "File Name": file.name,
            "Candidate Name": candidate_name,
            "Score (%)": score,
            "Years Experience": exp,
            "Email": email or "Not Found",
            "Matched Keywords": matched_keywords,
            "Missing Skills": missing_skills,
            "Feedback": feedback,
            "Semantic Similarity": semantic_similarity,
            "JD Keyword Coverage (%)": jd_coverage_percentage,
            "Resume Raw Text": text
        })
        resume_text_map[file.name] = text
    
    progress_bar.empty() # Clear the progress bar
    status_text.empty() # Clear the status text


    df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False).reset_index(drop=True)

    st.session_state['screening_results'] = results

    # --- Overall Candidate Comparison Chart ---
    st.markdown("## 📊 Candidate Score Comparison")
    if not df.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(df['Candidate Name'], df['Score (%)'], color=['#4CAF50' if s >= cutoff else '#FFC107' if s >= (cutoff * 0.75) else '#F44336' for s in df['Score (%)']])
        ax.set_xlabel("Candidate", fontsize=14)
        ax.set_ylabel("Score (%)", fontsize=14)
        ax.set_title("Resume Screening Scores Across Candidates", fontsize=16, fontweight='bold')
        ax.set_ylim(0, 100)
        plt.xticks(rotation=60, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Upload resumes to see a comparison chart.")

    st.markdown("---")

    # === Detailed Individual Candidate Analysis ===
    st.markdown("## 🔍 Detailed Candidate Insights")

    if not df.empty:
        jd_top_skills_list = get_top_keywords(jd_text, num_keywords=20)
        
        for idx, row in df.iterrows():
            candidate_display_name = row['Candidate Name']
            
            with st.container(border=True): # Use a container to group each candidate's analysis
                st.subheader(f"{idx+1}. {candidate_display_name} - Score: {row['Score (%)']:.2f}%")
                
                col_info, col_exp_match = st.columns([3, 1])

                with col_info:
                    st.markdown(f"**Overall Assessment:** {row['Feedback']}")
                    st.write(f"**Years of Experience:** {row['Years Experience']:.1f} years")
                    st.write(f"**Contact Email:** {row['Email']}")
                    st.write(f"**Semantic Similarity (JD vs. Resume):** **{row['Semantic Similarity']:.2f}** (A measure of conceptual alignment, higher is better.)")
                    st.write(f"**JD Keyword Coverage:** **{row['JD Keyword Coverage (%)']:.2f}%** (Percentage of job description keywords found in the resume.)")

                with col_exp_match:
                    st.markdown("### Experience Match")
                    exp_ratio = min(row['Years Experience'] / min_experience, 1.0) if min_experience > 0 else 1.0
                    st.progress(exp_ratio)
                    if row['Years Experience'] >= min_experience:
                        st.success(f"Meets/Exceeds required {min_experience} years.")
                    else:
                        st.warning(f"Below required {min_experience} years.")

                st.markdown("---")
                st.markdown("### 🎯 Skill Alignment Breakdown (Top JD Skills)")
                resume_words_for_matching = {word for word in re.findall(r'\b\w+\b', clean_text(row['Resume Raw Text'])) if word not in STOP_WORDS}

                matched_jd_skills = []
                missing_jd_skills = []
                for skill in jd_top_skills_list:
                    if skill in resume_words_for_matching:
                        matched_jd_skills.append(skill)
                    else:
                        missing_jd_skills.append(skill)

                if matched_jd_skills:
                    st.markdown(f"**✅ Matched Skills:** {', '.join(sorted(matched_jd_skills))}")
                else:
                    st.info("No significant top JD skills were directly matched in this resume.")

                if missing_jd_skills:
                    st.markdown(f"**❌ Missing Skills:** {', '.join(sorted(missing_jd_skills))}")
                else:
                    st.success("All top JD skills found in this resume!")
                
                # --- Resume Keyword Cloud ---
                st.markdown("### ☁️ Candidate Keyword Cloud")
                resume_words_for_cloud = " ".join([word for word in re.findall(r'\b\w+\b', clean_text(row['Resume Raw Text'])) if word not in STOP_WORDS])
                if resume_words_for_cloud:
                    wordcloud_res = WordCloud(width=600, height=300, background_color='white', collocations=False).generate(resume_words_for_cloud)
                    fig_res, ax_res = plt.subplots(figsize=(8, 4))
                    ax_res.imshow(wordcloud_res, interpolation='bilinear')
                    ax_res.axis('off')
                    st.pyplot(fig_res)
                else:
                    st.info("No significant keywords to display for this resume after filtering common words.")


                with st.expander("📄 View Full Resume Text"):
                    st.code(resume_text_map.get(row['File Name'], ''), height=300)
            st.markdown("---") # Separator between candidate analyses
    else:
        st.info("No candidates to display detailed analysis for yet.")

    st.markdown("---")

    # === "Who is Better" Statement ===
    if not df.empty:
        top_candidate = df.iloc[0]
        st.markdown("## 🏆 Top Recommended Candidate")
        st.success(
            f"Based on our AI-powered screening, **{top_candidate['Candidate Name']}** "
            f"stands out as the top candidate with an impressive score of **{top_candidate['Score (%)']:.2f}%** "
            f"and **{top_candidate['Years Experience']:.1f} years of relevant experience**. "
            f"Their profile demonstrates an **{top_candidate['Feedback'].lower()}**."
        )
    else:
        st.info("Upload resumes to identify the top candidate.")

    st.markdown("---")

    # Add a 'Tag' column for quick categorization
    df['Tag'] = df.apply(lambda row: "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3 else (
        "✅ Good Fit" if row['Score (%)'] >= 75 else "⚠️ Needs Review"), axis=1)

    shortlisted = df[(df['Score (%)'] >= cutoff) & (df['Years Experience'] >= min_experience)]

    st.markdown("## 🚀 Screening Summary")
    col_metrics = st.columns(3)
    with col_metrics[0]:
        st.metric("Total Candidates Processed", len(df))
    with col_metrics[1]:
        st.metric("Shortlisted Candidates", len(shortlisted))
    with col_metrics[2]:
        st.metric("Cutoff Score Applied", f"{cutoff}%")

    st.markdown("### 📋 All Candidate Results Table")
    # Display only relevant columns for the main table overview
    display_df = df[['Candidate Name', 'Score (%)', 'Years Experience', 'JD Keyword Coverage (%)', 'Semantic Similarity', 'Feedback', 'Tag', 'Email']]
    st.dataframe(display_df, use_container_width=True)

    # Add download button for results
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Full Results (CSV)",
        data=csv_data,
        file_name="candidate_screening_results.csv",
        mime="text/csv",
        help="Download a CSV file containing all screening results, including detailed metrics."
    )
    st.markdown("---")
else:
    st.info("Please upload a Job Description and at least one Resume to begin the screening process.")
