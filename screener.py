import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
import sklearn
import joblib
import numpy as np
from datetime import datetime # Needed for extract_years_of_experience
import matplotlib.pyplot as plt
from wordcloud import WordCloud # Import WordCloud
from sentence_transformers import SentenceTransformer
import nltk # Import NLTK
import collections # For counting word frequencies
from sklearn.metrics.pairwise import cosine_similarity # For semantic similarity

# Download NLTK stopwords data if not already downloaded
# This line will only run once when the app starts or when this part of the code is executed.
# It's good practice to put this outside functions if it's a one-time setup.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
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
# IMPORTANT: Ensure this SentenceTransformer model matches the one used in train_model.py
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

def extract_name(text):
    """
    Attempts to extract a name from the first few lines of the resume text.
    This is a heuristic and might not be perfect for all resume formats.
    """
    lines = text.strip().split('\n')
    if not lines:
        return None

    # Heuristic: Assume name is in the first 1-3 lines and is usually capitalized
    # Filter out lines that look like contact info (email, phone, linkedin, github)
    potential_name_lines = []
    for line in lines[:3]: # Check first 3 lines
        line = line.strip()
        if not re.search(r'[@\d\.\-]', line) and len(line.split()) <= 4 and line.isupper() or (line and line[0].isupper() and all(word[0].isupper() or not word.isalpha() for word in line.split())):
            potential_name_lines.append(line)

    if potential_name_lines:
        # Take the longest potential name line as the name
        name = max(potential_name_lines, key=len)
        # Remove common resume headers if they somehow get picked up as names
        name = re.sub(r'summary|education|experience|skills|projects|certifications', '', name, flags=re.IGNORECASE).strip()
        if name:
            return name.title() # Return in title case
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
    resume_clean = clean_text(resume_text) # Clean text here for consistency
    jd_clean = clean_text(jd_text) # Clean text here for consistency

    # Filter out stop words from resume and JD text before finding overlaps
    resume_words = {word for word in re.findall(r'\b\w+\b', resume_clean) if word not in STOP_WORDS}
    jd_words = {word for word in re.findall(r'\b\w+\b', jd_clean) if word not in STOP_WORDS}

    # Calculate overlap
    overlap = resume_words & jd_words
    matched_keywords = ", ".join(sorted(list(overlap))) # Sort for consistency

    # Identify missing skills from JD that are not in resume
    missing_skills_set = jd_words - resume_words
    missing_skills = ", ".join(sorted(list(missing_skills_set))) # Sort for consistency

    # Calculate JD Keyword Coverage Percentage for smart_score as well
    keyword_overlap_count = len(overlap) # Use the calculated overlap
    if len(jd_words) > 0:
        jd_coverage_percentage = (keyword_overlap_count / len(jd_words)) * 100
    else:
        jd_coverage_percentage = 0.0

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

    # semantic_similarity is not calculated in smart_score, so return 0.0
    return score, matched_keywords, missing_skills, feedback, 0.0, round(jd_coverage_percentage, 2)


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
    semantic_similarity = 0.0 # Initialize semantic_similarity
    jd_coverage_percentage = 0.0 # Initialize jd_coverage_percentage

    # If ML model is not loaded, fall back to smart_score
    if ml_model is None:
        st.error("❌ ML model not loaded. Falling back to smart_score for all metrics.")
        score, matched_keywords, missing_skills, feedback, semantic_similarity, jd_coverage_percentage = smart_score(resume_text, jd_text, years_exp)
        return score, matched_keywords, missing_skills, feedback, semantic_similarity, jd_coverage_percentage

    try:
        # Generate embeddings for JD and resume
        jd_embed = model.encode(jd_clean)
        resume_embed = model.encode(resume_clean)

        # Calculate semantic similarity (cosine similarity)
        semantic_similarity = cosine_similarity(jd_embed.reshape(1, -1), resume_embed.reshape(1, -1))[0][0]
        semantic_similarity = float(np.clip(semantic_similarity, 0, 1)) # Ensure between 0 and 1

        # Extract numerical features for the ML model - these are NOT filtered by STOP_WORDS for embedding input
        # but keyword_overlap_count is filtered to be consistent with training.
        resume_words_all = set(re.findall(r'\b\w+\b', resume_clean))
        jd_words_all = set(re.findall(r'\b\w+\b', jd_clean))

        # Filter words for keyword overlap count using STOP_WORDS
        resume_words_filtered = {word for word in resume_words_all if word not in STOP_WORDS}
        jd_words_filtered = {word for word in jd_words_all if word not in STOP_WORDS}

        keyword_overlap_count = len(resume_words_filtered & jd_words_filtered)
        
        # Ensure years_exp is a float, default to 0.0 if None
        years_exp_for_model = float(years_exp) if years_exp is not None else 0.0

        # Concatenate all features for the ML model (384 + 384 + 1 + 1 = 770 features)
        # Removed resume_len and matched_core_skills_count to match 770 features
        features = np.concatenate([jd_embed, resume_embed, [years_exp_for_model], [keyword_overlap_count]])

        st.info(f"🔍 Feature shape: {features.shape}")

        # Predict score using the loaded ML model
        predicted_score = ml_model.predict([features])[0]
        st.success(f"🧠 Predicted score (ML base): {predicted_score:.2f}")

        # Calculate JD Keyword Coverage Percentage for display purposes
        if len(jd_words_filtered) > 0:
            jd_coverage_percentage = (keyword_overlap_count / len(jd_words_filtered)) * 100
        else:
            jd_coverage_percentage = 0.0

        # Blend ML predicted score with JD keyword coverage and semantic similarity for stronger differentiation
        # Adjusted weights: More emphasis on ML predicted score and semantic similarity.
        blended_score = (predicted_score * 0.6) + \
                        (jd_coverage_percentage * 0.1) + \
                        (semantic_similarity * 100 * 0.3) # Scale semantic_similarity to 0-100 range

        # Introduce a bonus for high semantic match AND good experience
        if semantic_similarity > 0.7 and years_exp >= 3: # Thresholds can be adjusted
            blended_score += 5 # Add a bonus of 5 points

        score = float(np.clip(blended_score, 0, 100)) # Ensure score is between 0 and 100


        # Calculate matched and missing keywords for display, applying STOP_WORDS filter
        overlap_words_set_display = resume_words_filtered & jd_words_filtered
        matched_keywords = ", ".join(sorted(list(overlap_words_set_display)))
        missing_skills_set_display = jd_words_filtered - resume_words_filtered
        missing_skills = ", ".join(sorted(list(missing_skills_set_display)))

        # Generate feedback based on ML score
        if score > 90:
            feedback = "Excellent fit: Outstanding semantic match, high keyword coverage, and strong experience."
        elif score >= 75:
            feedback = "Good fit: Solid semantic match, good keyword coverage, and relevant experience."
        elif score >= 60: # Adjusted threshold for "Moderate fit" to reflect higher potential scores
            feedback = "Moderate fit: Decent alignment, but some areas for improvement in specific keywords or experience."
        else:
            feedback = "Lower fit: Significant gaps in semantic match and keyword coverage. Further review recommended."

        # Original fallback logic: if ML score is too low, use smart_score to be more robust
        # Note: The fallback will now also return jd_coverage_percentage
        if score < 10:
            st.warning("⚠️ Blended score is very low. Using fallback smart_score for a more reliable assessment.")
            score, matched_keywords, missing_skills, feedback, semantic_similarity, jd_coverage_percentage = smart_score(resume_text, jd_text, years_exp)

        return round(score, 2), matched_keywords, missing_skills, feedback, round(semantic_similarity, 2), round(jd_coverage_percentage, 2)

    except Exception as e:
        st.error(f"❌ semantic_score failed during prediction: {e}. Falling back to smart_score.")
        # Fallback to smart_score if any error occurs during ML prediction
        score, matched_keywords, missing_skills, feedback, semantic_similarity, jd_coverage_percentage = smart_score(resume_text, jd_text, years_exp)
        return score, matched_keywords, missing_skills, feedback, semantic_similarity, jd_coverage_percentage


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
    # --- Job Description Keyword Cloud ---
    st.markdown("## ☁️ Job Description Keyword Cloud")
    jd_words_for_cloud = " ".join([word for word in re.findall(r'\b\w+\b', clean_text(jd_text)) if word not in STOP_WORDS])
    if jd_words_for_cloud:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(jd_words_for_cloud)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("No significant keywords to display for the Job Description after filtering common words.")
    st.divider()


    results = []
    resume_text_map = {}
    for file in resume_files:
        text = extract_text_from_pdf(file)
        if text.startswith("[ERROR]"):
            st.error(f"Could not process {file.name}: {text}")
            continue

        exp = extract_years_of_experience(text)
        email = extract_email(text)
        candidate_name = extract_name(text) or file.name.replace('.pdf', '').replace('_', ' ').title() # Use extracted name or cleaned file name
        # Call semantic_score and unpack all returned values, including semantic_similarity and jd_coverage_percentage
        score, matched_keywords, missing_skills, feedback, semantic_similarity, jd_coverage_percentage = semantic_score(text, jd_text, exp)
        summary = f"{exp}+ years exp. | {text.strip().splitlines()[0]}" if text else f"{exp}+ years exp."

        results.append({
            "File Name": file.name,
            "Candidate Name": candidate_name, # Store extracted name
            "Score (%)": score,
            "Years Experience": exp,
            "Summary": summary,
            "Email": email or "Not found",
            "Matched Keywords": matched_keywords,
            "Missing Skills": missing_skills,
            "Feedback": feedback,
            "Semantic Similarity": semantic_similarity, # Add semantic similarity to results
            "JD Keyword Coverage (%)": jd_coverage_percentage, # Add JD keyword coverage
            "Resume Raw Text": text # Store raw text for individual word cloud
        })
        resume_text_map[file.name] = text

    df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False)

    # --- Save results to session state for main.py to access ---
    st.session_state['screening_results'] = results

    # --- Overall Candidate Comparison Chart (Improved Matplotlib Bar Chart) ---
    st.markdown("## 📊 Candidate Score Comparison")
    if not df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use the extracted candidate name for the x-axis labels
        bars = ax.bar(df['Candidate Name'], df['Score (%)'], color='skyblue')
        ax.set_xlabel("Candidate", fontsize=12)
        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_title("Resume Screening Scores", fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100) # Ensure y-axis goes from 0 to 100
        plt.xticks(rotation=45, ha='right') # Rotate labels for readability
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 2), ha='center', va='bottom') # Add score labels
        st.pyplot(fig)
    else:
        st.info("Upload resumes to see a comparison chart.")

    st.divider()

    # === Detailed Individual Candidate Analysis ===
    st.markdown("## 📝 Detailed Candidate Analysis")

    if not df.empty:
        # Get top JD skills once for all candidates
        jd_top_skills_list = get_top_keywords(jd_text, num_keywords=20)
        jd_top_skills_set = set(jd_top_skills_list)

        for _, row in df.iterrows():
            candidate_display_name = row['Candidate Name'] # Use the extracted name
            st.subheader(f"Analysis for {candidate_display_name}")
            individual_analysis_paragraph = (
                f"**{candidate_display_name}** scored **{row['Score (%)']:.2f}%** "
                f"with **{row['Years Experience']:.1f} years of experience**. "
                f"This candidate's profile is assessed as: **{row['Feedback']}**. "
            )

            st.markdown(individual_analysis_paragraph)

            # --- New Feature: Semantic Similarity Score ---
            st.markdown(f"**Semantic Similarity (JD vs. Resume):** {row['Semantic Similarity']:.2f} (Higher is better)")

            # --- New Feature: JD Keyword Coverage Percentage ---
            st.markdown(f"**JD Keyword Coverage:** {row['JD Keyword Coverage (%)']:.2f}% of job description keywords found in resume.")

            # --- Enhanced Skill Matching Breakdown ---
            st.markdown("### 📊 Skill Alignment with Job Description")
            resume_words_for_matching = {word for word in re.findall(r'\b\w+\b', clean_text(row['Resume Raw Text'])) if word not in STOP_WORDS}

            matched_jd_skills = []
            missing_jd_skills = []
            for skill in jd_top_skills_list:
                if skill in resume_words_for_matching:
                    matched_jd_skills.append(skill)
                else:
                    missing_jd_skills.append(skill)

            if matched_jd_skills:
                st.markdown(f"**✅ Matched Job Description Skills:** {', '.join(sorted(matched_jd_skills))}")
            else:
                st.info("No significant top JD skills matched in this resume.")

            if missing_jd_skills:
                st.markdown(f"**❌ Missing Job Description Skills:** {', '.join(sorted(missing_jd_skills))}")
            else:
                st.success("All top JD skills found in this resume!")

            # --- New Feature: Experience Match Visual ---
            st.markdown("### ⏳ Experience Match")
            exp_ratio = min(row['Years Experience'] / min_experience, 1.0) if min_experience > 0 else 1.0
            st.progress(exp_ratio)
            if row['Years Experience'] >= min_experience:
                st.success(f"Candidate has {row['Years Experience']:.1f} years of experience, meeting or exceeding the required {min_experience} years.")
            else:
                st.warning(f"Candidate has {row['Years Experience']:.1f} years of experience, less than the required {min_experience} years.")

            with st.expander("📄 Resume Preview"):
                st.code(resume_text_map.get(row['File Name'], ''))
            st.markdown("---") # Separator for individual analyses
    else:
        st.info("No candidates to display yet for detailed analysis.")

    st.divider()

    # === "Who is Better" Statement ===
    if not df.empty:
        top_candidate = df.iloc[0]
        st.markdown("## 🏆 Top Candidate Recommendation")
        st.success(
            f"Based on the screening, **{top_candidate['Candidate Name']}** "
            f"is the top-ranked candidate with a score of **{top_candidate['Score (%)']:.2f}%** and "
            f"**{top_candidate['Years Experience']:.1f} years of experience**. "
            f"Their profile shows a **{top_candidate['Feedback'].lower()}**."
        )
    else:
        st.info("Upload resumes to get a top candidate recommendation.")

    st.divider()

    # Add a 'Tag' column for quick categorization
    df['Tag'] = df.apply(lambda row: "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3 else (
        "✅ Good Fit" if row['Score (%)'] >= 75 else "⚠️ Needs Review"), axis=1)

    shortlisted = df[(df['Score (%)'] >= cutoff) & (df['Years Experience'] >= min_experience)]

    st.metric("✅ Shortlisted Candidates", len(shortlisted))

    st.markdown("### 📋 All Candidate Results Table")
    st.dataframe(df) # Display the DataFrame

    # Add download button for results
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_data,
        file_name="candidate_screening_results.csv",
        mime="text/csv",
    )

    # Add download button for detailed report (PDF)
    # This would require a library like FPDF or ReportLab, which is outside the scope of this interaction.
    # st.download_button(
    #     label="⬇️ Download Detailed PDF Report",
    #     data=b'', # Placeholder
    #     file_name="detailed_report.pdf",
    #     mime="application/pdf",
    #     disabled=True, # Disable for now as functionality is not implemented
    #     help="Detailed PDF report generation is not yet implemented."
    # )
