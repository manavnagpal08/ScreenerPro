import pandas as pd
import numpy as np
import re
from datetime import datetime # Needed for extract_years_of_experience
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer
import joblib
import os
import nltk # Import NLTK

# Download NLTK stopwords data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    print("✅ NLTK stopwords downloaded successfully for training script!")


# --- Configuration ---
MODEL_SAVE_PATH = "ml_screening_model.pkl"
# DATASET_PATH = "resume_jd_labeled_data_scaled.csv" # Commented out as we are using synthetic data


# Load embedding model
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading SentenceTransformer model: {e}")
    print("Please ensure you have an active internet connection or the model is cached locally.")
    exit() # Exit if the embedding model cannot be loaded

# --- Stop Words List (Using NLTK for consistency with app) ---
NLTK_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
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


# --- Helper Functions (Copied/Adapted from your Streamlit app) ---
def clean_text(text):
    """Cleans text by removing newlines, extra spaces, and non-ASCII characters."""
    text = re.sub(r'\n', ' ', text) # Remove newlines
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Remove non-ASCII characters
    return text.strip().lower()

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


# --- Synthetic Data Generation (Larger and More Diverse) ---
print("Generating a larger and more diverse synthetic dataset for demonstration purposes...")
synthetic_data = [
    # High Match Examples (Score 80-100) - Data Analyst Focus
    {
        "resume_text": "Data Analyst with 5 years experience in SQL, Python (Pandas), Tableau, and Power BI. Developed interactive dashboards and reports for business insights.",
        "jd_text": "Senior Data Analyst (4+ years) with expertise in SQL, Python for data manipulation, and visualization tools like Tableau/Power BI. Strong analytical and reporting skills required.",
        "score": 95
    },
    {
        "resume_text": "Experienced Business Intelligence Analyst, 6 years. Proficient in SQL, Excel, Power BI, and data warehousing. Led data-driven projects.",
        "jd_text": "BI Analyst (5+ years) needed. Strong in SQL, Excel, Power BI, and data modeling. Experience with large datasets.",
        "score": 90
    },
    {
        "resume_text": "Junior Data Analyst, 2 years experience. Skills in SQL, basic Python, and Excel. Assisted with data cleaning and report generation.",
        "jd_text": "Entry-level Data Analyst (1-3 years) with SQL and Excel skills. Python knowledge a plus. Support data reporting.",
        "score": 85
    },
    {
        "resume_text": "Data Scientist, 7 years experience. Specializes in ML with Python (Scikit-learn, TensorFlow), NLP, and big data (Spark).",
        "jd_text": "Lead Data Scientist (6+ years) expert in Python, ML, NLP, and distributed computing (Spark). PhD preferred.",
        "score": 98
    },
    {
        "resume_text": "Supply Chain Analyst, 4 years experience. Optimized logistics using data analysis, Excel, and some SQL.",
        "jd_text": "Supply Chain Data Analyst (3+ years) with strong analytical skills, Excel, and SQL for logistics optimization.",
        "score": 88
    },
    {
        "resume_text": "Financial Data Analyst, 5 years. Built financial models in Excel, created dashboards in Power BI. SQL for data extraction.",
        "jd_text": "Financial Analyst (4+ years) proficient in financial modeling, Excel, Power BI, and SQL for reporting.",
        "score": 92
    },
    {
        "resume_text": "Marketing Data Analyst, 3 years. Analyzed campaign performance using Google Analytics, SQL, and Excel.",
        "jd_text": "Marketing Analyst (2+ years) with strong analytical skills, Google Analytics, SQL, and Excel.",
        "score": 87
    },
    {
        "resume_text": "Healthcare Data Analyst, 6 years. Worked with patient data, SQL, and R for statistical analysis. Tableau for visualization.",
        "jd_text": "Healthcare Data Analyst (5+ years) with SQL, R, and Tableau experience. Strong understanding of healthcare data.",
        "score": 93
    },
    {
        "resume_text": "Web Analytics Specialist, 4 years. Expertise in Google Analytics, A/B testing, and data interpretation. Some Python.",
        "jd_text": "Web Analyst (3+ years) skilled in Google Analytics, A/B testing, and Python for data analysis.",
        "score": 89
    },
    {
        "resume_text": "Product Data Analyst, 5 years. Analyzed product usage, user behavior. SQL, Python (Pandas), and Mixpanel.",
        "jd_text": "Product Analyst (4+ years) to analyze user data. Proficient in SQL, Python, and product analytics tools.",
        "score": 91
    },

    # Moderate Match Examples (Score 50-79)
    {
        "resume_text": "Software Engineer with 3 years experience in Java and Spring Boot. Basic SQL knowledge.",
        "jd_text": "Data Analyst required with 5+ years experience. Must have strong SQL, Python, and Tableau for advanced analytics.",
        "score": 60 # Some overlap but not primary focus
    },
    {
        "resume_text": "Recent graduate, Computer Science. Projects in C++ and data structures. No professional experience.",
        "jd_text": "Junior Data Analyst (1-3 years) with SQL and Excel skills. Python knowledge a plus. Support data reporting.",
        "score": 55 # Good foundational skills, but limited direct experience/tools
    },
    {
        "resume_text": "Project Coordinator, 4 years experience. Managed small projects, organized meetings. Some Excel reporting.",
        "jd_text": "Business Intelligence Analyst (5+ years) with strong SQL, Power BI, and data warehousing experience.",
        "score": 50 # General project skills, limited data skills
    },
    {
        "resume_text": "Marketing Manager, 7 years experience. Developed marketing strategies, managed teams.",
        "jd_text": "Marketing Data Analyst (3+ years) with strong analytical skills, Google Analytics, SQL, and Excel.",
        "score": 65 # Relevant domain, but less hands-on data skills
    },
    {
        "resume_text": "IT Support Specialist, 5 years experience. Troubleshooted network issues and provided user support.",
        "jd_text": "Data Engineer (4+ years) with expertise in ETL, big data (Spark, Hadoop), and cloud platforms (AWS). Python essential.",
        "score": 52 # Some tech background, but not data engineering specific
    },
    {
        "resume_text": "Financial Advisor, 8 years experience. Advised clients on investments. Used financial software.",
        "jd_text": "Financial Data Analyst, 5 years. Built financial models in Excel, created dashboards in Power BI. SQL for data extraction.",
        "score": 68 # Financial domain match, but less data analysis tools
    },
    {
        "resume_text": "QA Tester, 3 years. Manual and automated testing using Selenium. Some Python scripting.",
        "jd_text": "Data Analyst (4+ years) with expertise in SQL, Python for data manipulation, and visualization tools like Tableau/Power BI.",
        "score": 58 # Python skill, but different domain focus
    },
    {
        "resume_text": "HR Generalist, 6 years. Managed recruitment, employee relations. Used HRIS systems.",
        "jd_text": "People Analytics Specialist (5+ years) with strong data analysis skills, SQL, and HR data experience.",
        "score": 62 # Data related to HR, but not general data analyst tools
    },
    {
        "resume_text": "Customer Success Manager, 4 years. Managed client relationships, resolved issues. Used CRM software.",
        "jd_text": "Product Data Analyst, 5 years. Analyzed product usage, user behavior. SQL, Python (Pandas), and Mixpanel.",
        "score": 50 # General business alignment, but limited data skills
    },
    {
        "resume_text": "Network Administrator, 7 years. Managed network infrastructure, configured routers/switches.",
        "jd_text": "Cybersecurity Analyst, 3 years in SIEM, incident response, network security. Certified Ethical Hacker.",
        "score": 70 # Strong network, but specific security tools missing
    },

    # Low Match Examples (Score 0-49) - Non-relevant roles
    {
        "resume_text": "Chef with 10 years experience running a restaurant kitchen. Managed inventory and staff.",
        "jd_text": "Senior Data Analyst (4+ years) with expertise in SQL, Python for data manipulation, and visualization tools like Tableau/Power BI. Strong analytical and reporting skills required.",
        "score": 10
    },
    {
        "resume_text": "Elementary School Teacher, 8 years experience. Developed lesson plans and managed classrooms.",
        "jd_text": "Data Scientist, 7 years experience. Specializes in ML with Python (Scikit-learn, TensorFlow), NLP, and big data (Spark).",
        "score": 5
    },
    {
        "resume_text": "Construction Worker, 12 years experience. Operated heavy machinery and supervised construction sites.",
        "jd_text": "DevOps Engineer with 6 years experience in AWS, Docker, Kubernetes, and CI/CD pipelines (Jenkins). Automation and scripting expertise.",
        "score": 8
    },
    {
        "resume_text": "Artist, 15 years experience. Created paintings and sculptures. Exhibited in galleries.",
        "jd_text": "Full Stack Developer (4+ years) skilled in React, Node.js, and NoSQL databases. Experience building RESTful APIs.",
        "score": 7
    },
    {
        "resume_text": "Professional Athlete, 10 years experience. Competed in international sports events.",
        "jd_text": "Business Analyst, 8 years experience. Expertise in requirements gathering, process mapping, and Agile methodologies. SQL proficient.",
        "score": 12
    },
    {
        "resume_text": "Hair Stylist, 7 years experience. Provided hair cutting and styling services to clients.",
        "jd_text": "Cybersecurity Analyst, 3 years in SIEM, incident response, network security. Certified Ethical Hacker.",
        "score": 9
    },
    {
        "resume_text": "Musician, 20 years experience performing live concerts and composing music.",
        "jd_text": "Project Manager (PMP certified) with 10 years experience leading software development projects using Scrum.",
        "score": 6
    },
    {
        "resume_text": "Flight Attendant, 8 years experience. Ensured passenger safety and comfort on flights.",
        "jd_text": "UX/UI Designer (3+ years) skilled in Figma, Sketch, user research, and prototyping. Strong portfolio.",
        "score": 10
    },
    {
        "resume_text": "Bartender, 5 years experience. Prepared drinks and managed bar inventory.",
        "jd_text": "Java Backend Developer, 7 years experience. Spring Boot, Microservices, REST APIs, Kafka. Worked on high-traffic systems.",
        "score": 11
    },
    {
        "resume_text": "Gardener, 3 years experience. Maintained gardens and landscapes.",
        "jd_text": "Data Engineer, 5 years experience. ETL pipelines, Spark, Hadoop, AWS Glue, Python scripting.",
        "score": 13
    },
    # Adding more diverse examples to increase dataset size and variety
    {
        "resume_text": "Recent grad with a degree in English Literature. Strong writing skills. No tech experience.",
        "jd_text": "Data Analyst with 3+ years experience in Python, SQL, and data visualization.",
        "score": 20
    },
    {
        "resume_text": "Customer Support Lead, 5 years. Managed a team of 10, resolved complex customer issues.",
        "jd_text": "Senior Business Analyst (7+ years) for Agile projects. Strong in requirements analysis, process optimization, and SQL.",
        "score": 35
    },
    {
        "resume_text": "Freelance Photographer, 6 years. Specializes in portrait photography and photo editing.",
        "jd_text": "UX/UI Designer (3+ years) skilled in Figma, Sketch, and user-centered design. Portfolio required.",
        "score": 40
    },
    {
        "resume_text": "Mechanical Engineer, 8 years. Designed mechanical systems and performed simulations.",
        "jd_text": "Data Scientist with 5+ years experience in Python, R, and statistical modeling.",
        "score": 30
    },
    {
        "resume_text": "Research Assistant, 2 years. Conducted lab experiments and analyzed results using basic statistics.",
        "jd_text": "Data Analyst with 3+ years experience in SQL, Python, and Tableau for business intelligence.",
        "score": 45
    },
    {
        "resume_text": "Administrative Assistant, 10 years. Managed office operations, scheduling, and record keeping.",
        "jd_text": "Project Manager (5+ years) with PMP certification and experience in software development.",
        "score": 25
    },
    {
        "resume_text": "Logistics Coordinator, 3 years. Managed shipping schedules and inventory tracking.",
        "jd_text": "Supply Chain Data Analyst (3+ years) with strong analytical skills, Excel, and SQL for logistics optimization.",
        "score": 75 # Moderate match, relevant domain
    },
    {
        "resume_text": "Web Designer, 4 years. Created responsive websites using HTML, CSS, and JavaScript. Familiar with Figma.",
        "jd_text": "Full Stack Developer (4+ years) skilled in React, Node.js, and NoSQL databases. Experience building RESTful APIs.",
        "score": 70 # Good overlap, but not full stack
    },
    {
        "resume_text": "Database Administrator, 7 years. Managed SQL Server databases, backups, and performance tuning.",
        "jd_text": "Data Engineer (4+ years) with expertise in ETL, big data (Spark, Hadoop), and cloud platforms (AWS). Python essential.",
        "score": 78 # Strong database, but less big data/cloud
    },
    {
        "resume_text": "Technical Support Engineer, 5 years. Provided remote technical support for enterprise software.",
        "jd_text": "Cybersecurity Analyst, 3 years in SIEM, incident response, network security. Certified Ethical Hacker.",
        "score": 60 # General tech support, but not security specific
    },
    {
        "resume_text": "Data Entry Clerk, 2 years. Entered data into spreadsheets and databases. High accuracy.",
        "jd_text": "Data Analyst required with 3+ years experience. Must have strong SQL, Python, and Tableau for advanced analytics.",
        "score": 30
    },
    {
        "resume_text": "Sales Representative, 4 years. Generated leads and closed sales. Used CRM system.",
        "jd_text": "Marketing Data Analyst, 3 years. Analyzed campaign performance using Google Analytics, SQL, and Excel.",
        "score": 28
    },
    {
        "resume_text": "Academic Researcher, 5 years. Published papers, conducted statistical analysis using R.",
        "jd_text": "Data Scientist with 5+ years experience in Python, R, and statistical modeling.",
        "score": 80 # Strong academic fit
    },
    {
        "resume_text": "Network Engineer, 6 years. Designed and implemented network solutions. Cisco certified.",
        "jd_text": "DevOps Engineer with 6 years experience in AWS, Docker, Kubernetes, and CI/CD pipelines (Jenkins). Automation and scripting expertise.",
        "score": 65 # Some overlap in infrastructure
    },
    {
        "resume_text": "UI Developer, 3 years. Built user interfaces with HTML, CSS, JavaScript, and React.",
        "jd_text": "UX/UI Designer (3+ years) skilled in Figma, Sketch, and user-centered design. Portfolio required.",
        "score": 72 # Strong UI, but less UX design
    },
    {
        "resume_text": "Backend Developer, 4 years. Developed APIs using Node.js and Express. Familiar with SQL.",
        "jd_text": "Java Backend Developer, 7 years experience. Spring Boot, Microservices, REST APIs, Kafka. Worked on high-traffic systems.",
        "score": 60 # Backend match, but different language/framework
    },
    {
        "resume_text": "Data Analyst with 1 year experience. Basic SQL and Excel. Learning Python.",
        "jd_text": "Data Analyst required with 3+ years experience. Must have strong SQL, Python, and Tableau for advanced analytics.",
        "score": 60 # Entry level, but direct relevant skills
    },
    {
        "resume_text": "Senior Data Analyst, 8 years. Led teams, developed complex reports in Tableau, Power BI. Expert in SQL and Python.",
        "jd_text": "Data Analyst (4+ years) with expertise in SQL, Python for data manipulation, and visualization tools like Tableau/Power BI. Strong analytical and reporting skills required.",
        "score": 98 # Very strong match
    },
    {
        "resume_text": "Software Tester, 5 years. Created test plans, executed manual and automated tests.",
        "jd_text": "QA Automation Engineer (3+ years) with Selenium, Python, and CI/CD integration.",
        "score": 70 # Good testing, but needs more automation focus
    },
    {
        "resume_text": "Business Consultant, 10 years. Advised clients on strategy and operations. Some data analysis.",
        "jd_text": "Business Analyst, 8 years experience. Expertise in requirements gathering, process mapping, and Agile methodologies. SQL proficient.",
        "score": 75 # Good strategic/business fit, but less hands-on BA
    },
    {
        "resume_text": "Database Developer, 6 years. Designed and implemented databases. SQL and stored procedures.",
        "jd_text": "Data Engineer, 5 years experience. ETL pipelines, Spark, Hadoop, AWS Glue, Python scripting.",
        "score": 70 # Strong database, but less big data/cloud engineering
    },
    {
        "resume_text": "Machine Learning Engineer, 3 years. Developed ML models in Python, deployed to production.",
        "jd_text": "Data Scientist, 7 years experience. Specializes in ML with Python (Scikit-learn, TensorFlow), NLP, and big data (Spark).",
        "score": 85 # Good ML, but less experience than JD asks
    },
    {
        "resume_text": "Data Visualization Specialist, 4 years. Created interactive dashboards using D3.js and Tableau.",
        "jd_text": "Data Analyst with 3+ years experience in Python, SQL, and data visualization.",
        "score": 88 # Strong visualization, good match
    },
    {
        "resume_text": "IT Project Manager, 7 years. Managed IT infrastructure projects. Agile and Waterfall.",
        "jd_text": "Project Manager (8+ years) with PMP and Scrum Master certifications. Lead complex software projects.",
        "score": 80 # Good project management, but slightly less experience than JD
    },
    {
        "resume_text": "Cloud Engineer, 5 years. Managed AWS infrastructure, deployed applications with Docker.",
        "jd_text": "DevOps Engineer with 6 years experience in AWS, Docker, Kubernetes, and CI/CD pipelines (Jenkins). Automation and scripting expertise.",
        "score": 85 # Good cloud/docker, but less CI/CD/Kubernetes focus
    },
    {
        "resume_text": "Data Science Intern, 6 months. Assisted with data cleaning and model evaluation.",
        "jd_text": "Data Scientist, 7 years experience. Specializes in ML with Python (Scikit-learn, TensorFlow), NLP, and big data (Spark).",
        "score": 40 # Very low experience for senior role
    },
    {
        "resume_text": "Business Analyst with 2 years experience. Gathered requirements, created process flows.",
        "jd_text": "Senior Business Analyst (7+ years) for Agile projects. Strong in requirements analysis, process optimization, and SQL.",
        "score": 50 # Relevant, but too junior for senior role
    },
    {
        "resume_text": "Cybersecurity Intern, 3 months. Assisted with security audits and vulnerability scans.",
        "jd_text": "Cybersecurity Analyst, 3 years in SIEM, incident response, network security. Certified Ethical Hacker.",
        "score": 30 # Very low experience for analyst role
    },
    {
        "resume_text": "Python Developer, 2 years. Developed scripts for automation. Familiar with REST APIs.",
        "jd_text": "Full Stack Developer (4+ years) skilled in React, Node.js, and NoSQL databases. Experience building RESTful APIs.",
        "score": 55 # Good Python, but not full stack or Node.js
    },
    {
        "resume_text": "Data Analyst with 3 years experience. Proficient in SQL, Excel. Some exposure to Python.",
        "jd_text": "Data Analyst required with 3+ years experience. Must have strong SQL, Python, and Tableau for advanced analytics.",
        "score": 75 # Good match, but might lack advanced Python/Tableau
    },
    {
        "resume_text": "Senior Data Analyst with 6 years experience. Led projects, mentored juniors. Expert in Tableau, Power BI, SQL, Python.",
        "jd_text": "Data Analyst required with 3+ years experience. Must have strong SQL, Python, and Tableau for advanced analytics.",
        "score": 98 # Excellent match
    },
    {
        "resume_text": "Entry-level Data Scientist. Master's degree in Statistics. Projects in R and Python.",
        "jd_text": "Data Scientist, 7 years experience. Specializes in ML with Python (Scikit-learn, TensorFlow), NLP, and big data (Spark).",
        "score": 60 # Good academic, but no professional experience for senior role
    },
    {
        "resume_text": "DevOps Intern, 6 months. Assisted with CI/CD pipeline setup.",
        "jd_text": "DevOps Engineer with 6 years experience in AWS, Docker, Kubernetes, and CI/CD pipelines (Jenkins). Automation and scripting expertise.",
        "score": 35 # Very low experience for senior role
    },
    {
        "resume_text": "Full Stack Developer, 2 years. Built small web apps with Angular and Node.js. Used MySQL.",
        "jd_text": "Full Stack Developer (4+ years) skilled in React, Node.js, and NoSQL databases. Experience building RESTful APIs.",
        "score": 65 # Good start, but less experience and different frontend framework
    },
    {
        "resume_text": "Business Analyst, 3 years. Gathered requirements, created documentation. Some exposure to Agile.",
        "jd_text": "Senior Business Analyst (7+ years) for Agile projects. Strong in requirements analysis, process optimization, and SQL.",
        "score": 55 # Relevant, but too junior for senior role
    },
    {
        "resume_text": "Cybersecurity Analyst, 1 year. Monitored security alerts. Basic knowledge of SIEM.",
        "jd_text": "Cybersecurity Analyst, 3 years in SIEM, incident response, network security. Certified Ethical Hacker.",
        "score": 60 # Relevant, but less experience and certifications
    },
    {
        "resume_text": "Project Coordinator, 2 years. Assisted project managers. Organized schedules.",
        "jd_text": "Project Manager (8+ years) with PMP and Scrum Master certifications. Lead complex software projects.",
        "score": 40 # Too junior for senior project manager
    },
    {
        "resume_text": "UX Designer, 2 years. Created wireframes and prototypes in Sketch. Conducted user interviews.",
        "jd_text": "UI/UX Designer (3+ years) skilled in Figma, Sketch, and user-centered design. Portfolio required.",
        "score": 70 # Good skills, but slightly less experience than JD
    },
    {
        "resume_text": "Backend Developer, 3 years. Developed microservices in Python (Flask). Used PostgreSQL.",
        "jd_text": "Java Backend Developer, 7 years experience. Spring Boot, Microservices, REST APIs, Kafka. Worked on high-traffic systems.",
        "score": 60 # Relevant, but different language/framework and less experience
    },
    {
        "resume_text": "Data Engineer Intern, 3 months. Assisted with data loading scripts.",
        "jd_text": "Data Engineer, 5 years experience. ETL pipelines, Spark, Hadoop, AWS Glue, Python scripting.",
        "score": 30 # Very low experience for engineer role
    },
    {
        "resume_text": "Data Analyst with 4 years experience. Strong in SQL, Excel, and Power BI. Experience with large datasets.",
        "jd_text": "Data Analyst required with 3+ years experience. Must have strong SQL, Excel, and Power BI skills for business intelligence and reporting.",
        "score": 90 # Excellent match
    },
    {
        "resume_text": "Data Scientist with 2 years experience. Built predictive models using Python and Scikit-learn.",
        "jd_text": "Data Scientist, 7 years experience. Specializes in ML with Python (Scikit-learn, TensorFlow), NLP, and big data (Spark).",
        "score": 70 # Good skills, but less experience for senior role
    },
    {
        "resume_text": "DevOps Engineer, 3 years. Managed AWS resources. Familiar with Docker.",
        "jd_text": "DevOps Engineer with 6 years experience in AWS, Docker, Kubernetes, and CI/CD pipelines (Jenkins). Automation and scripting expertise.",
        "score": 65 # Relevant, but less experience and missing some tools
    },
    {
        "resume_text": "Full Stack Developer, 3 years. Built web applications with Vue.js and Express. Used MongoDB.",
        "jd_text": "Full Stack Developer (4+ years) skilled in React, Node.js, and NoSQL databases. Experience building RESTful APIs.",
        "score": 70 # Good skills, but less experience and different frontend framework
    },
    {
        "resume_text": "Business Analyst, 5 years. Requirements gathering, stakeholder management. Used Jira.",
        "jd_text": "Senior Business Analyst (7+ years) for Agile projects. Strong in requirements analysis, process optimization, and SQL.",
        "score": 75 # Good match, but slightly less experience for senior role
    },
    {
        "resume_text": "Cybersecurity Analyst, 2 years. Monitored security systems. Conducted vulnerability scans.",
        "jd_text": "Cybersecurity Analyst, 3 years in SIEM, incident response, network security. Certified Ethical Hacker.",
        "score": 80 # Good match, slightly less experience/certs
    },
    {
        "resume_text": "Project Manager, 5 years. Managed cross-functional teams. PMP certified.",
        "jd_text": "Project Manager (8+ years) with PMP and Scrum Master certifications. Lead complex software projects.",
        "score": 85 # Good match, but less experience for senior role
    },
    {
        "resume_text": "UX/UI Designer, 2 years. Created user flows and prototypes. Familiar with Adobe XD.",
        "jd_text": "UI/UX Designer (3+ years) skilled in Figma, Sketch, and user-centered design. Portfolio required.",
        "score": 65 # Relevant, but less experience and different tools
    },
    {
        "resume_text": "Backend Developer, 5 years. Developed APIs in C#. Used SQL Server. Experience with Azure.",
        "jd_text": "Java Backend Developer, 7 years experience. Spring Boot, Microservices, REST APIs, Kafka. Worked on high-traffic systems.",
        "score": 60 # Relevant, but different language/framework
    },
    {
        "resume_text": "Data Engineer, 3 years. Built data pipelines using Python and Airflow. Some AWS experience.",
        "jd_text": "Data Engineer, 5 years experience. ETL pipelines, Spark, Hadoop, AWS Glue, Python scripting.",
        "score": 75 # Good match, but less experience and missing some tools
    },
    {
        "resume_text": "Data Analyst with 2 years experience. Proficient in SQL, Excel. Some exposure to Python.",
        "jd_text": "Data Analyst required with 3+ years experience. Must have strong SQL, Python, and Tableau for advanced analytics.",
        "score": 60 # Entry level, but direct relevant skills
    },
    {
        "resume_text": "Senior Data Analyst, 6 years experience. Led projects, mentored juniors. Expert in Tableau, Power BI, SQL, Python.",
        "jd_text": "Data Analyst required with 3+ years experience. Must have strong SQL, Python, and Tableau for advanced analytics.",
        "score": 98 # Excellent match
    },
    {
        "resume_text": "Entry-level Data Scientist. Master's degree in Statistics. Projects in R and Python.",
        "jd_text": "Data Scientist, 7 years experience. Specializes in ML with Python (Scikit-learn, TensorFlow), NLP, and big data (Spark).",
        "score": 60 # Good academic, but no professional experience for senior role
    },
    {
        "resume_text": "DevOps Intern, 6 months. Assisted with CI/CD pipeline setup.",
        "jd_text": "DevOps Engineer with 6 years experience in AWS, Docker, Kubernetes, and CI/CD pipelines (Jenkins). Automation and scripting expertise.",
        "score": 35 # Very low experience for senior role
    },
    {
        "resume_text": "Full Stack Developer, 2 years. Built small web apps with Angular and Node.js. Used MySQL.",
        "jd_text": "Full Stack Developer (4+ years) skilled in React, Node.js, and NoSQL databases. Experience building RESTful APIs.",
        "score": 65 # Good start, but less experience and different frontend framework
    },
    {
        "resume_text": "Business Analyst, 3 years. Gathered requirements, created documentation. Some exposure to Agile.",
        "jd_text": "Senior Business Analyst (7+ years) for Agile projects. Strong in requirements analysis, process optimization, and SQL.",
        "score": 55 # Relevant, but too junior for senior role
    },
    {
        "resume_text": "Cybersecurity Analyst, 1 year. Monitored security alerts. Basic knowledge of SIEM.",
        "jd_text": "Cybersecurity Analyst, 3 years in SIEM, incident response, network security. Certified Ethical Hacker.",
        "score": 60 # Relevant, but less experience and certifications
    },
    {
        "resume_text": "Project Coordinator, 2 years. Assisted project managers. Organized schedules.",
        "jd_text": "Project Manager (8+ years) with PMP and Scrum Master certifications. Lead complex software projects.",
        "score": 40 # Too junior for senior project manager
    },
    {
        "resume_text": "UX Designer, 2 years. Created wireframes and prototypes in Sketch. Conducted user interviews.",
        "jd_text": "UI/UX Designer (3+ years) skilled in Figma, Sketch, and user-centered design. Portfolio required.",
        "score": 70 # Good skills, but slightly less experience than JD
    },
    {
        "resume_text": "Backend Developer, 3 years. Developed microservices in Python (Flask). Used PostgreSQL.",
        "jd_text": "Java Backend Developer, 7 years experience. Spring Boot, Microservices, REST APIs, Kafka. Worked on high-traffic systems.",
        "score": 60 # Relevant, but different language/framework and less experience
    },
    {
        "resume_text": "Data Engineer Intern, 3 months. Assisted with data loading scripts.",
        "jd_text": "Data Engineer, 5 years experience. ETL pipelines, Spark, Hadoop, AWS Glue, Python scripting.",
        "score": 30 # Very low experience for engineer role
    },
    {
        "resume_text": "Data Analyst with 4 years experience. Strong in SQL, Excel, and Power BI. Experience with large datasets.",
        "jd_text": "Data Analyst required with 3+ years experience. Must have strong SQL, Excel, and Power BI skills for business intelligence and reporting.",
        "score": 90 # Excellent match
    },
    {
        "resume_text": "Data Scientist with 2 years experience. Built predictive models using Python and Scikit-learn.",
        "jd_text": "Data Scientist, 7 years experience. Specializes in ML with Python (Scikit-learn, TensorFlow), NLP, and big data (Spark).",
        "score": 70 # Good skills, but less experience for senior role
    },
    {
        "resume_text": "DevOps Engineer, 3 years. Managed AWS resources. Familiar with Docker.",
        "jd_text": "DevOps Engineer with 6 years experience in AWS, Docker, Kubernetes, and CI/CD pipelines (Jenkins). Automation and scripting expertise.",
        "score": 65 # Relevant, but less experience and missing some tools
    },
    {
        "resume_text": "Full Stack Developer, 3 years. Built web applications with Vue.js and Express. Used MongoDB.",
        "jd_text": "Full Stack Developer (4+ years) skilled in React, Node.js, and NoSQL databases. Experience building RESTful APIs.",
        "score": 70 # Good skills, but less experience and different frontend framework
    },
    {
        "resume_text": "Business Analyst, 5 years. Requirements gathering, stakeholder management. Used Jira.",
        "jd_text": "Senior Business Analyst (7+ years) for Agile projects. Strong in requirements analysis, process optimization, and SQL.",
        "score": 75 # Good match, but slightly less experience for senior role
    },
    {
        "resume_text": "Cybersecurity Analyst, 2 years. Monitored security systems. Conducted vulnerability scans.",
        "jd_text": "Cybersecurity Analyst, 3 years in SIEM, incident response, network security. Certified Ethical Hacker.",
        "score": 80 # Good match, slightly less experience/certs
    },
    {
        "resume_text": "Project Manager, 5 years. Managed cross-functional teams. PMP certified.",
        "jd_text": "Project Manager (8+ years) with PMP and Scrum Master certifications. Lead complex software projects.",
        "score": 85 # Good match, but less experience for senior role
    },
    {
        "resume_text": "UX/UI Designer, 2 years. Created user flows and prototypes. Familiar with Adobe XD.",
        "jd_text": "UI/UX Designer (3+ years) skilled in Figma, Sketch, and user-centered design. Portfolio required.",
        "score": 65 # Relevant, but less experience and different tools
    },
    {
        "resume_text": "Backend Developer, 5 years. Developed APIs in C#. Used SQL Server. Experience with Azure.",
        "jd_text": "Java Backend Developer, 7 years experience. Spring Boot, Microservices, REST APIs, Kafka. Worked on high-traffic systems.",
        "score": 60 # Relevant, but different language/framework
    },
    {
        "resume_text": "Data Engineer, 3 years. Built data pipelines using Python and Airflow. Some AWS experience.",
        "jd_text": "Data Engineer, 5 years experience. ETL pipelines, Spark, Hadoop, AWS Glue, Python scripting.",
        "score": 75 # Good match, but less experience and missing some tools
    },
    # More examples to reach ~100 samples
    {
        "resume_text": "Data Analyst with 3 years experience in SQL, Python, and basic Power BI. Worked on customer churn analysis.",
        "jd_text": "Data Analyst (3+ years) with strong SQL, Python, and Power BI for customer analytics. Experience with churn prediction a plus.",
        "score": 88
    },
    {
        "resume_text": "Data Scientist, 4 years experience. Built recommendation systems using collaborative filtering and Python.",
        "jd_text": "Data Scientist (5+ years) specializing in recommendation engines, Python (PyTorch/TensorFlow), and large-scale data.",
        "score": 85
    },
    {
        "resume_text": "DevOps Engineer, 4 years. Managed cloud infrastructure on Azure. Implemented CI/CD with Azure DevOps.",
        "jd_text": "DevOps Engineer (5+ years) with Azure experience, strong in infrastructure as code and CI/CD automation.",
        "score": 89
    },
    {
        "resume_text": "Full Stack Developer, 6 years. Expert in React, Node.js, and PostgreSQL. Led a team of junior developers.",
        "jd_text": "Lead Full Stack Developer (5+ years) with React, Node.js, and relational database expertise. Leadership experience required.",
        "score": 95
    },
    {
        "resume_text": "Business Analyst, 6 years. Focused on financial systems. Gathered requirements for ERP implementation.",
        "jd_text": "Business Analyst (5+ years) with experience in financial systems and ERP implementations. Strong analytical skills.",
        "score": 90
    },
    {
        "resume_text": "Cybersecurity Engineer, 4 years. Implemented security solutions, conducted penetration testing.",
        "jd_text": "Cybersecurity Engineer (3+ years) with experience in security architecture, penetration testing, and vulnerability management.",
        "score": 92
    },
    {
        "resume_text": "Project Manager, 7 years. Managed Agile software projects. Certified Scrum Master.",
        "jd_text": "Project Manager (6+ years) with Agile and Scrum Master certifications. Experience managing complex software development projects.",
        "score": 93
    },
    {
        "resume_text": "UX Researcher, 3 years. Conducted user interviews, usability testing, and created personas.",
        "jd_text": "UX/UI Designer (3+ years) skilled in Figma, Sketch, and user-centered design. Portfolio required.",
        "score": 80 # Strong UX research, but less UI design
    },
    {
        "resume_text": "Backend Developer, 6 years. Built high-performance APIs in Go. Used Redis for caching.",
        "jd_text": "Backend Developer (5+ years) with expertise in Go, microservices, and caching technologies (Redis).",
        "score": 91
    },
    {
        "resume_text": "Data Engineer, 4 years. Built real-time data pipelines with Kafka and Spark Streaming.",
        "jd_text": "Data Engineer (4+ years) with experience in real-time data processing (Kafka, Spark Streaming) and cloud platforms.",
        "score": 94
    },
    {
        "resume_text": "Data Analyst with 1 year experience. Basic SQL and Excel. Learning Python.",
        "jd_text": "Senior Data Analyst (4+ years) with expertise in SQL, Python for data manipulation, and visualization tools like Tableau/Power BI. Strong analytical and reporting skills required.",
        "score": 50 # Relevant skills, but too junior for senior role
    },
    {
        "resume_text": "Data Scientist, 2 years experience. Built predictive models using Python and Scikit-learn.",
        "jd_text": "Lead Data Scientist (6+ years) expert in Python, ML, NLP, and distributed computing (Spark). PhD preferred.",
        "score": 65 # Relevant skills, but too junior for lead role
    },
    {
        "resume_text": "DevOps Engineer, 3 years. Managed AWS resources. Familiar with Docker.",
        "jd_text": "Experienced DevOps Engineer (5+ years) with strong AWS, Docker, Kubernetes, and CI/CD (Jenkins) skills. Focus on automation.",
        "score": 70 # Relevant skills, but less experience and missing some tools
    },
    {
        "resume_text": "Full Stack Developer, 3 years. Built web applications with Vue.js and Express. Used MongoDB.",
        "jd_text": "Full Stack Developer (4+ years) skilled in React, Node.js, and NoSQL databases. Experience building RESTful APIs.",
        "score": 75 # Relevant skills, but less experience and different frontend framework
    },
    {
        "resume_text": "Business Analyst, 5 years. Requirements gathering, stakeholder management. Used Jira.",
        "jd_text": "Senior Business Analyst (7+ years) for Agile projects. Strong in requirements analysis, process optimization, and SQL.",
        "score": 80 # Relevant skills, but slightly less experience for senior role
    },
    {
        "resume_text": "Cybersecurity Analyst, 2 years. Monitored security systems. Conducted vulnerability scans.",
        "jd_text": "Cybersecurity Analyst (3+ years) with SIEM, incident response, and network security experience. Certifications a plus.",
        "score": 85 # Relevant skills, but less experience and certifications
    },
    {
        "resume_text": "Project Manager, 5 years. Managed cross-functional teams. PMP certified.",
        "jd_text": "Project Manager (8+ years) with PMP and Scrum Master certifications. Lead complex software projects.",
        "score": 88 # Relevant skills, but less experience for senior role
    },
    {
        "resume_text": "UX/UI Designer, 2 years. Created user flows and prototypes. Familiar with Adobe XD.",
        "jd_text": "UI/UX Designer (3+ years) skilled in Figma, Sketch, and user-centered design. Portfolio required.",
        "score": 70 # Relevant skills, but less experience and different tools
    },
    {
        "resume_text": "Backend Developer, 5 years. Developed APIs in C#. Used SQL Server. Experience with Azure.",
        "jd_text": "Backend Developer (6+ years) with Java, Spring Boot, Microservices, and message queue (Kafka) experience.",
        "score": 65 # Relevant skills, but different language/framework
    },
    {
        "resume_text": "Data Engineer, 3 years. Built data pipelines using Python and Airflow. Some AWS experience.",
        "jd_text": "Data Engineer (4+ years) with expertise in ETL, big data (Spark, Hadoop), and cloud platforms (AWS). Python essential.",
        "score": 80 # Relevant skills, but less experience and missing some tools
    },
    {
        "resume_text": "Data Analyst with 3 years experience. Strong in SQL, Excel, and Power BI. Worked on customer churn analysis.",
        "jd_text": "Data Analyst (3+ years) with strong SQL, Python, and Power BI for customer analytics. Experience with churn prediction a plus.",
        "score": 90
    },
    {
        "resume_text": "Data Scientist, 4 years experience. Built recommendation systems using collaborative filtering and Python.",
        "jd_text": "Data Scientist (5+ years) specializing in recommendation engines, Python (PyTorch/TensorFlow), and large-scale data.",
        "score": 88
    },
    {
        "resume_text": "DevOps Engineer, 4 years. Managed cloud infrastructure on Azure. Implemented CI/CD with Azure DevOps.",
        "jd_text": "DevOps Engineer (5+ years) with Azure experience, strong in infrastructure as code and CI/CD automation.",
        "score": 90
    },
    {
        "resume_text": "Full Stack Developer, 6 years. Expert in React, Node.js, and PostgreSQL. Led a team of junior developers.",
        "jd_text": "Lead Full Stack Developer (5+ years) with React, Node.js, and relational database expertise. Leadership experience required.",
        "score": 96
    },
    {
        "resume_text": "Business Analyst, 6 years. Focused on financial systems. Gathered requirements for ERP implementation.",
        "jd_text": "Business Analyst (5+ years) with experience in financial systems and ERP implementations. Strong analytical skills.",
        "score": 92
    },
    {
        "resume_text": "Cybersecurity Engineer, 4 years. Implemented security solutions, conducted penetration testing.",
        "jd_text": "Cybersecurity Engineer (3+ years) with experience in security architecture, penetration testing, and vulnerability management.",
        "score": 93
    },
    {
        "resume_text": "Project Manager, 7 years. Managed Agile software projects. Certified Scrum Master.",
        "jd_text": "Project Manager (6+ years) with Agile and Scrum Master certifications. Experience managing complex software development projects.",
        "score": 95
    },
    {
        "resume_text": "UX Researcher, 3 years. Conducted user interviews, usability testing, and created personas.",
        "jd_text": "UX/UI Designer (3+ years) skilled in Figma, Sketch, and user-centered design. Portfolio required.",
        "score": 82
    },
    {
        "resume_text": "Backend Developer, 6 years. Built high-performance APIs in Go. Used Redis for caching.",
        "jd_text": "Backend Developer (5+ years) with expertise in Go, microservices, and caching technologies (Redis).",
        "score": 93
    },
    {
        "resume_text": "Data Engineer, 4 years. Built real-time data pipelines with Kafka and Spark Streaming.",
        "jd_text": "Data Engineer (4+ years) with experience in real-time data processing (Kafka, Spark Streaming) and cloud platforms.",
        "score": 96
    },
    {
        "resume_text": "Data Analyst with 1 year experience. Basic SQL and Excel. Learning Python.",
        "jd_text": "Senior Data Analyst (4+ years) with expertise in SQL, Python for data manipulation, and visualization tools like Tableau/Power BI. Strong analytical and reporting skills required.",
        "score": 55 # Relevant skills, but too junior for senior role
    },
    {
        "resume_text": "Data Scientist, 2 years experience. Built predictive models using Python and Scikit-learn.",
        "jd_text": "Lead Data Scientist (6+ years) expert in Python, ML, NLP, and distributed computing (Spark). PhD preferred.",
        "score": 68 # Relevant skills, but too junior for lead role
    },
    {
        "resume_text": "DevOps Engineer, 3 years. Managed AWS resources. Familiar with Docker.",
        "jd_text": "Experienced DevOps Engineer (5+ years) with strong AWS, Docker, Kubernetes, and CI/CD (Jenkins) skills. Focus on automation.",
        "score": 72 # Relevant skills, but less experience and missing some tools
    },
    {
        "resume_text": "Full Stack Developer, 3 years. Built web applications with Vue.js and Express. Used MongoDB.",
        "jd_text": "Full Stack Developer (4+ years) skilled in React, Node.js, and NoSQL databases. Experience building RESTful APIs.",
        "score": 78 # Relevant skills, but less experience and different frontend framework
    },
    {
        "resume_text": "Business Analyst, 5 years. Requirements gathering, stakeholder management. Used Jira.",
        "jd_text": "Senior Business Analyst (7+ years) for Agile projects. Strong in requirements analysis, process optimization, and SQL.",
        "score": 82 # Relevant skills, but slightly less experience for senior role
    },
    {
        "resume_text": "Cybersecurity Analyst, 2 years. Monitored security systems. Conducted vulnerability scans.",
        "jd_text": "Cybersecurity Analyst (3+ years) with SIEM, incident response, network security. Certified Ethical Hacker.",
        "score": 88 # Relevant skills, but less experience and certifications
    },
    {
        "resume_text": "Project Manager, 5 years. Managed cross-functional teams. PMP certified.",
        "jd_text": "Project Manager (8+ years) with PMP and Scrum Master certifications. Lead complex software projects.",
        "score": 90 # Relevant skills, but less experience for senior role
    },
    {
        "resume_text": "UX/UI Designer, 2 years. Created user flows and prototypes. Familiar with Adobe XD.",
        "jd_text": "UI/UX Designer (3+ years) skilled in Figma, Sketch, and user-centered design. Portfolio required.",
        "score": 75 # Relevant skills, but less experience and different tools
    },
    {
        "resume_text": "Backend Developer, 5 years. Developed APIs in C#. Used SQL Server. Experience with Azure.",
        "jd_text": "Java Backend Developer, 7 years experience. Spring Boot, Microservices, REST APIs, Kafka. Worked on high-traffic systems.",
        "score": 68 # Relevant skills, but different language/framework
    },
    {
        "resume_text": "Data Engineer, 3 years. Built data pipelines using Python and Airflow. Some AWS experience.",
        "jd_text": "Data Engineer, 5 years experience. ETL pipelines, Spark, Hadoop, AWS Glue, Python scripting.",
        "score": 82 # Relevant skills, but less experience and missing some tools
    },
    {
        "resume_text": "Data Analyst with 3 years experience. Strong in SQL, Excel, and Power BI. Worked on customer churn analysis.",
        "jd_text": "Data Analyst (3+ years) with strong SQL, Python, and Power BI for customer analytics. Experience with churn prediction a plus.",
        "score": 90
    },
    {
        "resume_text": "Data Scientist, 4 years experience. Built recommendation systems using collaborative filtering and Python.",
        "jd_text": "Data Scientist (5+ years) specializing in recommendation engines, Python (PyTorch/TensorFlow), and large-scale data.",
        "score": 88
    },
    {
        "resume_text": "DevOps Engineer, 4 years. Managed cloud infrastructure on Azure. Implemented CI/CD with Azure DevOps.",
        "jd_text": "DevOps Engineer (5+ years) with Azure experience, strong in infrastructure as code and CI/CD automation.",
        "score": 90
    },
    {
        "resume_text": "Full Stack Developer, 6 years. Expert in React, Node.js, and PostgreSQL. Led a team of junior developers.",
        "jd_text": "Lead Full Stack Developer (5+ years) with React, Node.js, and relational database expertise. Leadership experience required.",
        "score": 96
    },
    {
        "resume_text": "Business Analyst, 6 years. Focused on financial systems. Gathered requirements for ERP implementation.",
        "jd_text": "Business Analyst (5+ years) with experience in financial systems and ERP implementations. Strong analytical skills.",
        "score": 92
    },
    {
        "resume_text": "Cybersecurity Engineer, 4 years. Implemented security solutions, conducted penetration testing.",
        "jd_text": "Cybersecurity Engineer (3+ years) with experience in security architecture, penetration testing, and vulnerability management.",
        "score": 93
    },
    {
        "resume_text": "Project Manager, 7 years. Managed Agile software projects. Certified Scrum Master.",
        "jd_text": "Project Manager (6+ years) with Agile and Scrum Master certifications. Experience managing complex software development projects.",
        "score": 95
    },
    {
        "resume_text": "UX Researcher, 3 years. Conducted user interviews, usability testing, and created personas.",
        "jd_text": "UX/UI Designer (3+ years) skilled in Figma, Sketch, and user-centered design. Portfolio required.",
        "score": 82
    },
    {
        "resume_text": "Backend Developer, 6 years. Built high-performance APIs in Go. Used Redis for caching.",
        "jd_text": "Backend Developer (5+ years) with expertise in Go, microservices, and caching technologies (Redis).",
        "score": 93
    },
    {
        "resume_text": "Data Engineer, 4 years. Built real-time data pipelines with Kafka and Spark Streaming.",
        "jd_text": "Data Engineer (4+ years) with experience in real-time data processing (Kafka, Spark Streaming) and cloud platforms.",
        "score": 96
    },
]

df_train = pd.DataFrame(synthetic_data)
print("✅ Synthetic dataset created.")
print(f"Synthetic dataset head:\n{df_train.head()}")


# --- Feature Extraction ---
def extract_features_for_training(jd_text, resume_text):
    """
    Extracts features from job description and resume text for model training.
    Features include sentence embeddings, keyword overlap, resume length,
    matched core skills, and YEARS OF EXPERIENCE.
    """
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    # Generate embeddings
    jd_embed = embedding_model.encode(jd_clean)
    resume_embed = embedding_model.encode(resume_clean)

    # Extract numerical features
    resume_words_all = set(re.findall(r'\b\w+\b', resume_clean))
    jd_words_all = set(re.findall(r'\b\w+\b', jd_clean))

    # Filter words for keyword overlap count using STOP_WORDS (consistent with app)
    resume_words_filtered = {word for word in resume_words_all if word not in STOP_WORDS}
    jd_words_filtered = {word for word in jd_words_all if word not in STOP_WORDS}

    keyword_overlap_count = len(resume_words_filtered & jd_words_filtered)
    resume_len = len(resume_clean.split())

    # Define core skills (can be expanded based on common job requirements)
    core_skills = ['sql', 'excel', 'python', 'tableau', 'powerbi', 'r', 'aws', 'java', 'spring', 'docker', 'kubernetes', 'figma', 'jira']
    matched_core_skills_count = sum(1 for skill in core_skills if skill in resume_clean)

    # *** IMPORTANT CHANGE: Include years of experience as a feature for training ***
    years_exp = extract_years_of_experience(resume_text)
    # Ensure years_exp is a float, default to 0.0 if None (though function should return float)
    years_exp = float(years_exp) if years_exp is not None else 0.0

    # Concatenate all features into a single vector
    # The order of features MUST be consistent between training and prediction
    extra_feats = np.array([keyword_overlap_count, resume_len, matched_core_skills_count, years_exp])
    return np.concatenate([jd_embed, resume_embed, extra_feats])

# Generate features for the entire dataset
print("Extracting features from the dataset...")
X = np.array([
    extract_features_for_training(row["jd_text"], row["resume_text"])
    for index, row in df_train.iterrows()
])
y = df_train["score"].values # Assuming 'score' is the target column in your CSV

print(f"Total samples: {len(X)}")
print(f"Feature vector shape: {X.shape}")

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --- Model Training ---
print("Training the Gradient Boosting Regressor model...")
# Increased n_estimators for potentially better performance with more data
reg = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
reg.fit(X_train, y_train)
print("✅ Model training complete.")

# --- Model Evaluation ---
print("\nEvaluating the model on the test set...")
y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")
print(f"Predicted score range: {np.min(y_pred):.2f} to {np.max(y_pred):.2f}")

# --- Save the Trained Model ---
try:
    joblib.dump(reg, MODEL_SAVE_PATH)
    print(f"\n✅ ML model retrained with extra features and saved as {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"❌ Error saving model: {e}")

print("\nTraining script finished. You can now use 'ml_screening_model.pkl' in your Streamlit app.")
