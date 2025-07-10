import streamlit as st
from login import login_section
from email_sender import send_email_to_candidate
import pdfplumber
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dateutil import parser
from datetime import datetime
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")

# --- Page Setup ---
st.set_page_config(page_title="Resume Screener Pro", layout="wide")
st.title("📂 Resume Screener Pro")
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stButton > button {
    background-color: #00cec9;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.6rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# --- Load JD ---
job_roles = {"Upload my own": None}
jd_dir = "data"
if os.path.exists(jd_dir):
    for fname in os.listdir(jd_dir):
        if fname.endswith(".txt"):
            job_roles[fname.replace(".txt", "").replace("_", " ").title()] = os.path.join(jd_dir, fname)

jd_option = st.selectbox("📌 Select Job Role or Upload Your Own JD", list(job_roles.keys()))
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

resume_files = st.file_uploader("📄 Upload Resumes", type="pdf", accept_multiple_files=True)
cutoff = st.slider("📈 Score Cutoff", 0, 100, 80)
min_exp = st.slider("💼 Minimum Years of Experience", 0, 15, 2)
cgpa_cutoff = st.slider("🎓 Min CGPA", 0.0, 10.0, 7.5, step=0.1)

# ========== UTILS ==========
def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            return ''.join([page.extract_text() or '' for page in pdf.pages])
    except:
        return ""

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else "Not Found"

def extract_links(text):
    linkedin = re.search(r'(https?://)?(www\.)?linkedin\.com/in/\S+', text, re.IGNORECASE)
    github = re.search(r'(https?://)?(www\.)?github\.com/\S+', text, re.IGNORECASE)
    return linkedin.group(0) if linkedin else None, github.group(0) if github else None

def extract_hyperlinks(file):
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file.read(), filetype="pdf")
        links = []
        for page in doc:
            links.extend([l["uri"] for l in page.get_links() if "uri" in l])
        linkedin = next((l for l in links if "linkedin.com/in" in l), None)
        github = next((l for l in links if "github.com" in l), None)
        return linkedin, github
    except:
        return None, None
def extract_scores(text):
    cgpa = re.search(r'([\d.]+)\s*/\s*10\s*(CGPA|C.G.P.A)?', text, re.IGNORECASE)
    try:
        return float(cgpa.group(1)) if cgpa else 0
    except:
        return 0

def extract_experience(text):
    patterns = [
        r'([A-Za-z]{3,9}\s\d{4})\s*[-–—to]+\s*(Present|[A-Za-z]{3,9}\s\d{4})',
        r'(\d+)\s*(?:\+)?\s*(years?|yrs?)'
    ]
    total_months = 0
    for start, end in re.findall(patterns[0], text, re.IGNORECASE):
        try:
            s = parser.parse(start)
            e = datetime.today() if 'present' in end.lower() else parser.parse(end)
            total_months += (e.year - s.year) * 12 + (e.month - s.month)
        except:
            continue

    for match in re.findall(patterns[1], text.lower()):
        try:
            total_months += int(match[0]) * 12
        except:
            continue
    return round(total_months / 12, 1)

def tfidf_score(jd, resume):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([jd, resume])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

def spacy_score(jd, resume):
    try:
        return round(nlp(jd).similarity(nlp(resume)) * 100, 2)
    except:
        return 0

def skill_match_score(jd, resume, years):
    jd_words = set(re.findall(r'\b\w+\b', jd.lower()))
    resume_words = set(re.findall(r'\b\w+\b', resume.lower()))
    core_skills = ['python', 'java', 'sql', 'html', 'css', 'javascript', 'react', 'machine', 'learning']
    matched = [word for word in jd_words if word in resume_words]
    total_weight = sum([3 if word in core_skills else 1 for word in matched])
    possible = sum([3 if word in core_skills else 1 for word in jd_words])
    score = (total_weight / possible) * 100 if possible else 0
    score += min(years, 10)
    return round(score, 2), matched, [w for w in core_skills if w in jd_words and w not in resume_words]

def final_score(tfidf, spacy, skill):
    return round(tfidf * 0.3 + spacy * 0.3 + skill * 0.4, 2)

def get_tag(score, years, cgpa):
    if score >= 90 and years >= 3 and cgpa > 8:
        return "🔥 Top Talent"
    elif score >= 75:
        return "✅ Good Fit"
    elif score < 60:
        return "❌ Not Shortlisted"
    return "⚠️ Needs Review"
# --- Resume Processing & Screening ---
if jd_text and resume_files:
    st.info("🔍 Screening resumes. Please wait...")
    results = []
    resume_text_map = {}

    for file in resume_files:
        resume_text = extract_text_from_pdf(file)
        resume_text_map[file.name] = resume_text
        experience = extract_experience(resume_text)
        cgpa = extract_scores(resume_text)
        email = extract_email(resume_text)
        linkedin, github = extract_links_from_text(resume_text)

        tfidf = tfidf_score(jd_text, resume_text)
        spacy_sim = spacy_score(jd_text, resume_text)
        skill_score, matched_keywords, missing = skill_match_score(jd_text, resume_text, experience)
        total_score = final_score(tfidf, spacy_sim, skill_score)
        tag = get_tag(total_score, experience, cgpa)

        results.append({
            "Name": file.name,
            "Score": total_score,
            "Years": experience,
            "CGPA": cgpa,
            "Matched Keywords": ", ".join(matched_keywords),
            "Missing Keywords": ", ".join(missing),
            "Email": email,
            "LinkedIn": linkedin or "❌",
            "GitHub": github or "❌",
            "Tag": tag
        })

    df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

    st.success("✅ Screening Complete!")
    st.dataframe(df, use_container_width=True)

    st.download_button("📥 Download Results CSV", data=df.to_csv(index=False), file_name="screening_results.csv")

    # --- Insights ---
    st.markdown("### 📊 Smart Insights")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Avg Score", f"{df['Score'].mean():.2f}%")
    col2.metric("💼 Avg Exp", f"{df['Years'].mean():.1f} yrs")
    col3.metric("🎓 Avg CGPA", f"{df['CGPA'].mean():.2f}")
    col4.metric("✅ Shortlisted", df[df["Tag"] != "❌ Not Shortlisted"].shape[0])

    st.markdown("### ☁️ Word Cloud of Matched Keywords")

    if "Matched Keywords" in df.columns and not df["Matched Keywords"].isnull().all():
    all_words = []
        for kw in df["Matched Keywords"].dropna():
            if isinstance(kw, str):
                all_words.extend(kw.split(", "))
    
        if all_words:
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_words))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No keywords found to generate a word cloud.")
    else:
        st.warning("⚠️ No matched keywords available to display word cloud.")


    # --- Top Candidate Highlights ---
    st.markdown("### 🏆 Top Candidates (Preview)")

    top_candidates = df.head(3)
    for _, row in top_candidates.iterrows():
        st.markdown(f"**{row['Name']}** — Score: {row['Score']}% | Exp: {row['Years']} yrs | CGPA: {row['CGPA']}")
        st.caption(f"Matched Skills: {row['Matched Keywords']}")
        st.info(f"{row['Tag']}")
        with st.expander("📄 Full Resume Text"):
            raw_text = resume_text_map.get(row['Name'], "")
            st.code(raw_text)

    # --- Email (Manual Send Only) ---
    st.markdown("### ✉️ Send Email to Selected Candidate (Manual Preview)")
    selected_resume = st.selectbox("Choose Candidate", options=df["Name"].tolist())
    raw_text = resume_text_map.get(selected_resume, "")
    selected_row = df[df["Name"] == selected_resume].iloc[0]

    default_msg = f"""
Hi {selected_resume.replace('.pdf', '')},

We're happy to inform you that you have been shortlisted for the role based on your impressive profile.

🧠 Score: {selected_row['Score']}%  
💼 Experience: {selected_row['Years']} years  
🎓 CGPA: {selected_row['CGPA']}  
🔗 LinkedIn: {selected_row['LinkedIn']}  
🔗 GitHub: {selected_row['GitHub']}

Regards,  
HR Team  
"""
    email_text = st.text_area("Preview Email", value=default_msg, height=250)
    if st.button("📤 Send Email Now"):
        send_email_to_candidate(
            name=selected_resume.replace(".pdf", ""),
            recipient=selected_row["Email"],
            subject="🎉 You've been shortlisted!",
            score=selected_row["Score"],
            feedback="Congratulations! You’ve passed the AI screening.",
            message=email_text
        )
        st.success("✅ Email sent.")
    # --- Visualizations ---
    st.markdown("### 📈 Visual Analytics")

    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.markdown("#### 🎯 Candidate Tags Distribution")
        tag_counts = df["Tag"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(tag_counts, labels=tag_counts.index, autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

    with col_bar:
        st.markdown("#### 🧠 Score Distribution")
        fig2, ax2 = plt.subplots()
        df["Score"].plot(kind="hist", bins=10, ax=ax2, color="#00b894", edgecolor="black")
        ax2.set_xlabel("Score")
        ax2.set_ylabel("Number of Candidates")
        st.pyplot(fig2)

    st.markdown("#### 💼 Experience Range Breakdown")
    bins = [0, 2, 5, 10, 15, 50]
    labels = ['0-2', '3-5', '6-10', '11-15', '15+']
    df["Exp Range"] = pd.cut(df["Years"], bins=bins, labels=labels, right=False)
    exp_range_data = df["Exp Range"].value_counts().sort_index()

    fig3, ax3 = plt.subplots()
    exp_range_data.plot(kind='bar', ax=ax3, color="#0984e3")
    ax3.set_ylabel("Number of Candidates")
    ax3.set_xlabel("Experience Range")
    st.pyplot(fig3)

    # --- Tag Filters ---
    st.markdown("### 🔍 Filter by Candidate Tag")
    tag_filter = st.selectbox("Choose Tag to View", options=["All"] + sorted(df["Tag"].unique().tolist()))

    filtered_df = df.copy() if tag_filter == "All" else df[df["Tag"] == tag_filter]
    st.dataframe(filtered_df, use_container_width=True)

    # --- Smart Resume Preview by Tag ---
    st.markdown("### 📄 Preview Resumes by Tag")

    selected_tag = st.selectbox("Select a Tag for Preview", options=sorted(df["Tag"].unique()))
    tag_resumes = df[df["Tag"] == selected_tag]

    if not tag_resumes.empty:
        preview_resume = st.selectbox("Select Candidate", tag_resumes["Name"].tolist(), key="preview_by_tag")
        preview_text = resume_text_map.get(preview_resume, "")
        st.markdown(f"#### ✨ Resume: {preview_resume}")
        st.code(preview_text)
    else:
        st.warning("No resumes found under this tag.")

    # --- Expanded Export ---
    st.markdown("### 📤 Export Filtered Candidates")
    export_csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📁 Download Filtered CSV", data=export_csv, file_name="filtered_candidates.csv", mime="text/csv")

    # --- Save JSON Report (for future automation or analysis) ---
    if st.checkbox("🧾 Save JSON Summary"):
        json_data = df.to_dict(orient="records")
        with open("screening_summary.json", "w", encoding="utf-8") as f:
            import json
            json.dump(json_data, f, indent=2)
        st.success("✅ JSON summary saved as `screening_summary.json`")
    # --- AI-Powered Skill Extraction (NLP-Based) ---
    st.markdown("### 🤖 AI Keyword Extractor")

    st.markdown("Upload any resume to auto-extract core skills using NLP.")
    single_resume = st.file_uploader("Upload Resume for Skill Extraction", type="pdf", key="ai_resume")

    if single_resume:
        raw_text = extract_text_from_pdf(single_resume)
        doc = nlp(raw_text)
        keywords = list(set([chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 2]))
        top_keywords = sorted(set(filter(lambda x: re.match(r"[a-zA-Z]+", x), keywords)))[:30]

        st.markdown("#### Top Extracted Skills:")
        st.code(", ".join(top_keywords))

        # Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(top_keywords))
        fig_kw, ax_kw = plt.subplots()
        ax_kw.imshow(wordcloud, interpolation='bilinear')
        ax_kw.axis('off')
        st.pyplot(fig_kw)

    # --- Optional Resume Upload via CSV Path (Advanced HR users) ---
    st.markdown("### 📁 Upload Resume List via CSV")
    st.caption("Upload a CSV containing columns: 'FilePath', 'CandidateName'. Resume files should exist in same folder or full path.")

    csv_upload = st.file_uploader("Upload CSV File", type="csv", key="csv_path_upload")
    if csv_upload:
        csv_df = pd.read_csv(csv_upload)
        for _, row in csv_df.iterrows():
            path = row['FilePath']
            name = row['CandidateName']
            try:
                with open(path, 'rb') as f:
                    text = extract_text_from_pdf(f)
                    st.markdown(f"**{name}**")
                    st.code(text[:600] + "...")
            except Exception as e:
                st.error(f"❌ Error loading {name}: {e}")

    # --- KMeans Clustering by Tag (Optional Analytics) ---
    from sklearn.cluster import KMeans
    import numpy as np

    st.markdown("### 🔬 Smart Clustering of Candidates")

    if st.button("🧠 Run Clustering"):
        tag_map = {'🔥 Top Talent': 2, '✅ Good Fit': 1, '⚠️ Needs Review': 0, '❌ Not Shortlisted': -1}
        df['TagCode'] = df['Tag'].map(tag_map).fillna(0)
        clustering_features = df[['Score', 'Years']].copy()
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(clustering_features)

        fig_clust, ax_clust = plt.subplots()
        scatter = ax_clust.scatter(df['Score'], df['Years'], c=df['Cluster'], cmap='Set1', s=100, alpha=0.7)
        ax_clust.set_xlabel("Score")
        ax_clust.set_ylabel("Years Experience")
        ax_clust.set_title("📊 KMeans Clustering of Candidates")
        st.pyplot(fig_clust)

    # --- Smart Filtering Box ---
    st.markdown("### 🔍 Candidate Search")
    search_query = st.text_input("Search by name, email or skill")

    if search_query:
        filtered_search = df[df.apply(lambda row: search_query.lower() in str(row).lower(), axis=1)]
        st.dataframe(filtered_search)
        st.caption(f"{len(filtered_search)} result(s) found.")
    else:
        st.caption("Use the box above to filter candidates.")

    # --- Tag Cloud for All Tags ---
    st.markdown("### ☁️ Tag Cloud & Skills Overview")

    full_keywords = []
    for keywords in df['Missing Skills']:
        full_keywords.extend(keywords.split(', '))

    tag_cloud = WordCloud(width=900, height=400, background_color="white").generate(" ".join(full_keywords))
    fig_tags, ax_tags = plt.subplots(figsize=(10, 5))
    ax_tags.imshow(tag_cloud, interpolation="bilinear")
    ax_tags.axis("off")
    st.pyplot(fig_tags)

    st.caption("Above is the word cloud of missing skills across all resumes.")
    # --- ✉️ Manual Email Preview & Send ---
    st.markdown("### ✉️ Send Emails to Shortlisted Candidates (Manual)")

    if not email_ready.empty:
        subject = st.text_input("Subject", value="🎯 You’ve Been Shortlisted!")
        body_template = st.text_area("Email Body", height=200, value="""
Dear {{name}},

We're excited to inform you that your resume has been shortlisted based on your profile, experience, and qualifications.

🧠 Final Score: {{score}}%  
💬 Feedback: {{feedback}}

Our team will contact you shortly with next steps.

Regards,  
HR Team – Shree Ram Recruitments
""")
        for _, row in email_ready.iterrows():
            preview = body_template.replace("{{name}}", row["Name"].replace(".pdf", "")).replace("{{score}}", str(row["Score"])).replace("{{feedback}}", row["Missing Skills"])
            with st.expander(f"📧 Email Preview: {row['Name']} ({row['Email']})"):
                st.markdown(preview.replace('\n', '<br>'), unsafe_allow_html=True)
                if st.button(f"📤 Send Email to {row['Name']}", key=f"send_{row['Name']}"):
                    send_email_to_candidate(
                        name=row["Name"],
                        score=row["Score"],
                        feedback=row["Missing Skills"],
                        recipient=row["Email"],
                        subject=subject,
                        message=preview
                    )
                    st.success(f"✅ Email sent to {row['Name']}")

    # --- ⭐️ Resume Rating System ---
    st.markdown("### 🌟 Resume Ratings")
    def get_stars(score):
        if score >= 90: return "⭐⭐⭐⭐⭐"
        elif score >= 80: return "⭐⭐⭐⭐"
        elif score >= 70: return "⭐⭐⭐"
        elif score >= 60: return "⭐⭐"
        else: return "⭐"

    df['Rating'] = df['Score'].apply(get_stars)
    st.dataframe(df[['Name', 'Score', 'Rating', 'Tag', 'Years', 'Email']], use_container_width=True)

    # --- 📝 Auto JD Generator from Top Profiles ---
    st.markdown("### 🧾 Auto-Generate JD from Top Candidates")

    top_profiles_text = " ".join([extract_text_from_pdf(f) for f in resume_files[:3]])
    doc = nlp(top_profiles_text)
    generated_keywords = list(set([token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and len(token.text) > 3]))
    jd_gen = "We are looking for candidates experienced in: " + ", ".join(sorted(generated_keywords[:25]))
    st.text_area("Generated JD", jd_gen, height=150)

    # --- 📊 Smart Insights Box ---
    st.markdown("### 📌 Smart Summary Insights")
    most_common_skills = pd.Series(" ".join(df["Missing Skills"]).split(", ")).value_counts().head(5)

    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.metric("🧠 Avg. Score", f"{df['Score'].mean():.2f}")
    col_i2.metric("💼 Avg. Experience", f"{df['Exp'].mean():.1f} yrs")
    col_i3.metric("✅ Shortlisted", df[df["Tag"] != "❌ Not Shortlisted"].shape[0])
    col_i4.metric("📌 Score Range", f"{df['Score'].min()} – {df['Score'].max()}")

    st.markdown("#### 🔝 Most Common Missing Skills")
    st.table(most_common_skills.reset_index().rename(columns={"index": "Skill", 0: "Frequency"}))

    # --- 🧪 Clustering Explanation (for HR insight) ---
    st.markdown("### 📚 What Do Clusters Mean?")
    st.info("""
🟥 Cluster 0 – Usually includes mid-level candidates with good fit  
🟨 Cluster 1 – May contain less experienced or average scoring candidates  
🟩 Cluster 2 – Likely top talent with high experience and matching skills  
*This helps HR visualize segmentation of applicant pool for deeper review.*
""")


