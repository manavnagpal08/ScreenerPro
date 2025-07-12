import streamlit as st
import pandas as pd
import docx2txt
import pdfplumber
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import json # For structured responses from LLM
from collections import Counter # To count skill occurrences

# Import skills data
from skills_data import ALL_SKILLS_MASTER_SET, SORTED_MASTER_SKILLS

# Load English tokenizer, tagger, parser, NER and word vectors
# Ensure you have downloaded the model: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm' in your terminal.")
    st.stop()


# --- Configuration ---
# Define a comprehensive list of stop words to filter out common non-skill words
STOP_WORDS = set([
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "for", "with", "at", "by", "from", "into", "during", "to", "of", "in",
    "on", "up", "down", "out", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y",
    "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    # Add common resume/job description specific words that are not skills
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


# --- Helper Functions ---

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

def extract_text_from_docx(docx_file):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        text = docx2txt.process(docx_file)
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
    return text

def extract_contact_info(text):
    """Extracts name, email, and phone number using regex."""
    name = re.search(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})", text)
    email = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    phone = re.search(r"(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", text)
    return (name.group(0) if name else "N/A",
            email.group(0) if email else "N/A",
            phone.group(0) if phone else "N/A")

def extract_years_of_experience(text):
    """Extracts years of experience from text."""
    # Look for patterns like "X years", "X+ years", "X-Y years"
    matches = re.findall(r'(\d+\+?|\d+\s*-\s*\d+)\s*(?:year|yr)s? of experience', text, re.IGNORECASE)
    if matches:
        # Take the first match and try to convert it to a number
        exp_str = matches[0].replace('+', '').strip()
        if '-' in exp_str:
            # If it's a range, take the upper bound or average
            try:
                start, end = map(int, exp_str.split('-'))
                return (start + end) / 2
            except ValueError:
                return 0
        else:
            try:
                return int(exp_str)
            except ValueError:
                return 0
    # Also look for "X years" or "X yr" followed by other words, but not necessarily "of experience"
    matches = re.findall(r'(\d+)\s*(?:year|yr)s?\b', text, re.IGNORECASE)
    if matches:
        # Filter out common years like "2020" if they are not clearly experience
        # This is a heuristic and might need refinement
        potential_years = [int(m) for m in matches if int(m) < 30] # Assuming max 30 years experience
        if potential_years:
            return max(potential_years) # Take the highest number as a proxy
    return 0

def clean_text(text):
    """Cleans text by lowercasing, removing punctuation, and stop words."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)

def calculate_cosine_similarity(text1, text2):
    """Calculates cosine similarity between two texts."""
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))[0][0]

def calculate_jaccard_similarity(text1, text2):
    """Calculates Jaccard similarity between two texts."""
    words1 = set(clean_text(text1).split())
    words2 = set(clean_text(text2).split())
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union != 0 else 0

def generate_ai_suggestion(jd_text, resume_text, matching_score, semantic_score, candidate_name):
    """Generates an AI-powered suggestion for a candidate using the Gemini API."""
    prompt = f"""
    You are an AI-powered resume screening assistant.
    Given the following Job Description and Candidate Resume, along with their matching scores, provide a concise hiring suggestion.
    Focus on key strengths, potential gaps, and a clear recommendation (e.g., "Strong Fit", "Good Match", "Consider for other roles", "Not a fit").

    Job Description:
    {jd_text[:1500]} # Truncate JD for prompt length
    
    Candidate Resume (Snippet):
    {resume_text[:1500]} # Truncate Resume for prompt length

    Matching Score (Keyword-based): {matching_score:.2f}%
    Semantic Similarity Score: {semantic_score:.2f}%
    Candidate Name: {candidate_name}

    Please provide a suggestion in the following JSON format:
    {{
        "candidate_name": "...",
        "overall_recommendation": "...",
        "strengths": ["...", "..."],
        "gaps": ["...", "..."],
        "next_steps": "..."
    }}
    """
    
    # Gemini API call
    chatHistory = []
    chatHistory.push({ role: "user", parts: [{ text: prompt }] });
    const payload = {
        contents: chatHistory,
        generationConfig: {
            responseMimeType: "application/json",
            responseSchema: {
                type: "OBJECT",
                properties: {
                    "candidate_name": { "type": "STRING" },
                    "overall_recommendation": { "type": "STRING" },
                    "strengths": { "type": "ARRAY", "items": { "type": "STRING" } },
                    "gaps": { "type": "ARRAY", "items": { "type": "STRING" } },
                    "next_steps": { "type": "STRING" }
                },
                "propertyOrdering": ["candidate_name", "overall_recommendation", "strengths", "gaps", "next_steps"]
            }
        }
    };
    const apiKey = ""
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

    try:
        response = await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        result = await response.json();

        if result.candidates and result.candidates[0].content and result.candidates[0].content.parts:
            # The API returns a string that needs to be parsed as JSON
            json_string = result.candidates[0].content.parts[0].text
            suggestion = json.loads(json_string)
            return suggestion
        else:
            st.warning("AI suggestion could not be generated. Unexpected API response structure.")
            return None
    except Exception as e:
        st.error(f"Error generating AI suggestion: {e}")
        return None

def generate_wordcloud(text, font_path):
    """
    Generates a word cloud from identified skills in the provided text,
    using the ALL_SKILLS_MASTER_SET from skills_data.py.
    """
    text_lower = text.lower()
    
    # Use a Counter to store skill frequencies
    found_skills_counts = Counter()

    # Iterate through SORTED_MASTER_SKILLS (longest first for multi-word matching)
    # This helps ensure that "Machine Learning" is matched before "Learning"
    temp_text = text_lower # Use a temporary copy to mark found skills
    
    for skill in SORTED_MASTER_SKILLS:
        # Use regex to find whole word matches for the skill
        # re.escape handles special characters in skill names
        # \b ensures whole word boundary
        pattern = r'\b' + re.escape(skill) + r'\b'
        
        # Find all occurrences of the skill
        matches = list(re.finditer(pattern, temp_text))
        
        for match in matches:
            found_skills_counts[skill] += 1
            # Replace the found skill with spaces to prevent re-matching parts of it
            # This is crucial for accurate counting, especially with overlapping skills
            temp_text = temp_text[:match.start()] + ' ' * len(match.group(0)) + temp_text[match.end():]

    # If no skills are found, inform the user
    if not found_skills_counts:
        st.info("No relevant skills from the master list were found in the Job Description to generate a word cloud.")
        return None

    # Create a single string where each skill is repeated according to its count
    # This allows WordCloud to correctly size words based on frequency
    wordcloud_input_string = " ".join([skill for skill, count in found_skills_counts.items() for _ in range(count)])

    # Generate word cloud
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        min_font_size=10,
        font_path=font_path,
        collocations=False # Set to False to avoid combining words that appear together often but aren't specific skills
    ).generate(wordcloud_input_string)

    return wordcloud


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="AI Resume Screener")

st.title("AI-Powered Resume Screener 🤖")
st.markdown("""
    Upload a Job Description and multiple resumes to get instant matching scores,
    extract key information, and receive AI-powered suggestions.
""")

# --- Job Description Upload ---
st.header("1. Upload Job Description")
jd_file = st.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="jd_uploader")
jd_text = ""

if jd_file:
    if jd_file.type == "application/pdf":
        jd_text = extract_text_from_pdf(jd_file)
    elif jd_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        jd_text = extract_text_from_docx(jd_file)
    else: # text/plain
        jd_text = jd_file.read().decode("utf-8")
    st.success("Job Description uploaded successfully!")
    st.subheader("Job Description Preview:")
    st.expander("Click to view JD", expanded=False).write(jd_text)

    # Generate and display JD word cloud (now skills-based)
    st.subheader("Job Description Skills Word Cloud")
    # Using a common font that should be available on most systems or Streamlit's environment
    # You might need to provide a specific .ttf path if running locally and seeing errors
    # For web deployment, 'DejaVuSans' is often a safe bet if available, or 'sans-serif' default
    # If font issues persist, consider bundling a .ttf file and referencing its path.
    # For now, let's assume a generic sans-serif is handled by WordCloud default or system.
    # If you have 'arial.ttf' or 'DejaVuSans.ttf' available, you can specify the path:
    # font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # Example for Linux
    # font_path = "C:/Windows/Fonts/arial.ttf" # Example for Windows
    # For a robust solution, you'd need to ensure the font file is accessible.
    # For this example, we'll try to rely on WordCloud's default font path handling or a common one.
    
    # A more robust way to handle font path for WordCloud in a potentially unknown environment:
    # Try to find a common font or use a default if not found.
    # For demonstration, we'll assume a basic font is available or WordCloud's default is acceptable.
    # If this causes issues, a .ttf file needs to be provided and its path correctly referenced.
    
    # Let's try to use a common font name that WordCloud might resolve or fall back gracefully
    # If you have a specific font file (e.g., 'arial.ttf') in the same directory as screenr.py:
    # font_path = "arial.ttf" 
    # Otherwise, rely on WordCloud's internal font discovery or default.
    
    # For simplicity and broad compatibility, let's use the default font handling of WordCloud
    # and only specify font_path if a specific .ttf is guaranteed to be present and accessible.
    # For this demonstration, we'll pass None and let WordCloud use its default.
    # If you want a specific font, you'd need to ensure it's deployed with your app.
    
    # Let's use a common font name that might be resolved, or let WordCloud default if not found.
    # This is a common challenge in containerized/cloud environments.
    # A safer bet is to bundle a font file. For now, let's assume 'sans-serif' or similar.
    
    # For now, let's pass a placeholder or let WordCloud handle default.
    # If running locally and you have a font file, specify its path.
    # Example: font_path = "path/to/your/font.ttf"
    
    # A common approach for Streamlit is to put font files in a 'fonts' directory
    # and reference them relative to the script.
    # E.g., if 'arial.ttf' is in a 'fonts' subfolder: font_path = "fonts/arial.ttf"
    
    # For this example, we'll try to use a generic font name or let WordCloud default.
    # If you encounter errors, please ensure a font file is accessible.
    
    # A robust way would be to check for common font paths or ship one.
    # For now, let's use a generic approach that might work or require local adjustment.
    
    # Let's provide a common font name that WordCloud might find or default.
    # If issues, user needs to provide a path to a .ttf file.
    # For example, on many Linux systems, 'DejaVuSans.ttf' is common.
    # font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    
    # For the purpose of this example, we will assume a font is available or WordCloud defaults.
    # If running locally and you get errors, download a .ttf font (e.g., Arial.ttf)
    # and place it in the same directory as your script, then set font_path = "Arial.ttf"
    
    font_path_for_wordcloud = None # Let WordCloud use its default or try to find one
    
    if jd_text:
        jd_wordcloud = generate_wordcloud(jd_text, font_path_for_wordcloud)
        if jd_wordcloud:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(jd_wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig) # Close the figure to prevent display issues
        else:
            st.info("Could not generate a skills word cloud for the Job Description. Ensure it contains relevant skills from the master list.")
    else:
        st.info("Upload a Job Description to see its skills word cloud.")


# --- Resumes Upload ---
st.header("2. Upload Resumes")
resume_files = st.file_uploader("Upload Resumes (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="resume_uploader")

if jd_text and resume_files:
    st.header("3. Candidate Analysis Results")
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, resume_file in enumerate(resume_files):
        status_text.text(f"Processing resume {i+1}/{len(resume_files)}: {resume_file.name}")
        
        resume_text = ""
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(resume_file)
        else: # text/plain
            resume_text = resume_file.read().decode("utf-8")

        if resume_text:
            # Extract basic info
            name, email, phone = extract_contact_info(resume_text)
            years_exp = extract_years_of_experience(resume_text)

            # Clean texts for similarity calculations
            cleaned_jd_text = clean_text(jd_text)
            cleaned_resume_text = clean_text(resume_text)

            # Calculate matching scores
            matching_score = calculate_jaccard_similarity(cleaned_jd_text, cleaned_resume_text) * 100
            semantic_score = calculate_cosine_similarity(jd_text, resume_text) * 100 # Use raw text for semantic

            # Generate AI suggestion
            with st.spinner(f"Generating AI suggestion for {name}..."):
                ai_suggestion = generate_ai_suggestion(jd_text, resume_text, matching_score, semantic_score, name)

            results.append({
                "Resume Name": resume_file.name,
                "Candidate Name": name,
                "Email": email,
                "Phone": phone,
                "Years Experience": years_exp,
                "Matching Score (%)": f"{matching_score:.2f}",
                "Semantic Similarity (%)": f"{semantic_score:.2f}",
                "AI Suggestion": ai_suggestion,
                "Full Resume Text": resume_text # Store for detailed view
            })
        progress_bar.progress((i + 1) / len(resume_files))
    status_text.text("All resumes processed!")

    if results:
        df = pd.DataFrame(results)
        
        # Display main results table
        st.subheader("Summary of Candidates")
        st.dataframe(df[['Resume Name', 'Candidate Name', 'Years Experience', 'Matching Score (%)', 'Semantic Similarity (%)']])

        # Detailed view for each candidate
        st.subheader("Detailed Candidate Insights")
        for i, row in df.iterrows():
            with st.expander(f"Details for {row['Candidate Name']} ({row['Resume Name']})"):
                st.write(f"**Email:** {row['Email']}")
                st.write(f"**Phone:** {row['Phone']}")
                st.write(f"**Years Experience:** {row['Years Experience']}")
                st.write(f"**Keyword Matching Score:** {row['Matching Score (%)']}%")
                st.write(f"**Semantic Similarity Score:** {row['Semantic Similarity (%)']}%")

                st.subheader("AI-Powered Suggestion:")
                if row['AI Suggestion']:
                    suggestion = row['AI Suggestion']
                    st.write(f"**Overall Recommendation:** {suggestion.get('overall_recommendation', 'N/A')}")
                    st.write("**Strengths:**")
                    for strength in suggestion.get('strengths', []):
                        st.markdown(f"- {strength}")
                    st.write("**Gaps:**")
                    for gap in suggestion.get('gaps', []):
                        st.markdown(f"- {gap}")
                    st.write(f"**Next Steps:** {suggestion.get('next_steps', 'N/A')}")
                else:
                    st.write("AI suggestion not available.")

                st.subheader("Full Resume Text:")
                st.text_area(f"Resume Text for {row['Candidate Name']}", row['Full Resume Text'], height=300)

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="resume_screening_results.csv",
            mime="text/csv",
        )

        # Plotting scores
        st.subheader("Candidate Score Visualization")
        scores_df = df.copy()
        scores_df['Matching Score (%)'] = scores_df['Matching Score (%)'].astype(float)
        scores_df['Semantic Similarity (%)'] = scores_df['Semantic Similarity (%)'].astype(float)

        fig, ax = plt.subplots(figsize=(10, 6))
        scores_df.plot(x='Candidate Name', y=['Matching Score (%)', 'Semantic Similarity (%)'], kind='bar', ax=ax)
        ax.set_ylabel("Score (%)")
        ax.set_title("Matching and Semantic Similarity Scores per Candidate")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

elif resume_files and not jd_text:
    st.warning("Please upload a Job Description first to start the screening process.")
elif not resume_files and jd_text:
    st.info("Upload resumes to start screening against the Job Description.")
else:
    st.info("Upload a Job Description and Resumes to begin the AI Resume Screening process.")

