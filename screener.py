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
import nltk # Import NLTK

# Download NLTK stopwords data if not already downloaded
# This line will only run once when the app starts or when this part of the code is executed.
# It's good practice to put this outside functions if it's a one-time setup.
try:
    nltk.data.find('corpora/stopwords')
except LookupError: # Changed from nltk.downloader.DownloadError to LookupError
    nltk.download('stopwords')
    st.success("✅ NLTK stopwords downloaded successfully!")


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

# --- Stop Words List (Using NLTK) ---
# Get the English stop words from NLTK and convert to a set for efficient lookups
NLTK_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))

# You can add custom words to this set if NLTK's list doesn't cover everything you consider a stop word
# For example, common resume/JD specific words that are not skills:
CUSTOM_STOP_WORDS = set([
    "work", "experience", "years", "year", "months", "month", "day", "days", "project", "projects",
    "team", "teams", "developed", "managed", "led", "created", "implemented", "designed",
    "responsible", "proficient", "knowledge", "ability", "strong", "proven", "demonstrated",
    "solution", "solutions", "system", "systems", "platform", "platforms", "framework", "frameworks",
    "database", "databases", "server", "servers", "cloud", "computing", "machine", "learning",
    "artificial", "intelligence", "api", "apis", "rest", "graphql", "agile", "scrum", "kanban",
    "devops", "ci", "cd", "testing", "qa", "security", "network", "networking", "virtualization",
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

# Combine NLTK stopwords with your custom stopwords
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
    Applies STOP_WORDS filtering for keyword analysis.
    """
    # Filter out stop words from resume and JD text before finding overlaps
    resume_words = {word for word in re.findall(r'\b\w+\b', resume_text.lower()) if word not in STOP_WORDS}
    jd_words = {word for word in re.findall(r'\b\w+\b', jd_text.lower()) if word not in STOP_WORDS}

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
    Applies STOP_WORDS filtering for keyword analysis before display.
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

        # Extract numerical features for the ML model - these are NOT filtered by STOP_WORDS for embedding input
        # but keyword_overlap_count is filtered to be consistent with training.
        resume_words_all = set(re.findall(r'\b\w+\b', resume_clean))
        jd_words_all = set(re.findall(r'\b\w+\b', jd_clean))

        # Filter words for keyword overlap count using STOP_WORDS
        resume_words_filtered = {word for word in resume_words_all if word not in STOP_WORDS}
        jd_words_filtered = {word for word in jd_words_all if word not in STOP_WORDS}

        keyword_overlap_count = len(resume_words_filtered & jd_words_filtered)
        resume_len = len(resume_clean.split())

        core_skills = ['sql', 'excel', 'python', 'tableau', 'powerbi', 'r', 'aws'] # These are specific, not general stop words
        matched_core_skills_count = sum(1 for skill in core_skills if skill in resume_clean)

        extra_feats = np.array([keyword_overlap_count, resume_len, matched_core_skills_count])

        # Concatenate all features for the ML model
        features = np.concatenate([jd_embed, resume_embed, extra_feats])

        st.info(f"🔍 Feature shape: {features.shape}")

        # Predict score using the loaded ML model
        predicted_score = ml_model.predict([features])[0]
        st.success(f"🧠 Predicted score: {predicted_score:.2f}")

        score = float(np.clip(predicted_score, 0, 100)) # Ensure score is between 0 and 100

        # Calculate matched and missing keywords for display, applying STOP_WORDS filter
        overlap_words_set_display = resume_words_filtered & jd_words_filtered
        matched_keywords = ", ".join(sorted(list(overlap_words_set_display)))
        missing_skills_set_display = jd_words_filtered - resume_words_filtered
        missing_skills = ", ".join(sorted(list(missing_skills_set_display)))

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
            # Re-run smart_score to ensure consistent logic for fallback, which also uses STOP_WORDS
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
            "Matched Keywords": matched_keywords, # Correct
            "Missing Skills": missing_skills,     # Corrected: was matched_keywords
            "Feedback": feedback                 # Correct
        })
        resume_text_map[file.name] = text

    df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False)

    # === 📊 Global Resume Screening Insights ===
    st.markdown("## 📊 Screening Insights & Summary")

    avg_score = df['Score (%)'].mean()
    avg_exp = df['Years Experience'].mean()

    # Aggregate top matched skills across all resumes
    all_matched_skills = []
    for keywords_str in df['Matched Keywords'].dropna():
        # Split by comma and strip whitespace, then add to list
        all_matched_skills.extend([skill.strip() for skill in keywords_str.split(',') if skill.strip()])
    top_matched_skills = pd.Series(all_matched_skills).value_counts().head(5)

    # Aggregate top missing skills across all resumes
    all_missing_skills = []
    for skills_str in df['Missing Skills'].dropna():
        # Split by comma and strip whitespace, then add to list
        all_missing_skills.extend([skill.strip() for skill in skills_str.split(',') if skill.strip()])
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
            st.info("No common matched skills found after filtering.")

        st.markdown("### ❌ Top 5 Missing Skills")
        if not top_missing_skills.empty:
            for skill, count in top_missing_skills.items():
                st.markdown(f"- ⚠️ {skill} ({count})")
        else:
            st.info("No common missing skills found after filtering.")

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
