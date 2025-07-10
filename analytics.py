import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import json # Import json for firebase config

# Firebase imports (ensure these are available in your environment)
from firebase_admin import credentials, initialize_app
from firebase_admin import firestore
from firebase_admin import auth

# Initialize Firebase (only once per app run)
# Ensure this block is consistent across all files that use Firebase
if 'firebase_initialized' not in st.session_state:
    try:
        firebase_config = json.loads(os.environ.get('__firebase_config', '{}'))
        if not firebase_config:
            st.error("Firebase configuration not found. Please ensure __firebase_config is set.")
            st.stop()

        cred = credentials.Certificate(firebase_config)
        initialize_app(cred)
        st.session_state['firebase_initialized'] = True
        st.success("✅ Firebase initialized successfully!")
    except Exception as e:
        st.error(f"❌ Firebase initialization failed: {e}")
        st.stop()

db = firestore.client()

# Authenticate user (anonymously for simplicity in this demo)
if 'user_authenticated' not in st.session_state:
    try:
        initial_auth_token = os.environ.get('__initial_auth_token')
        if initial_auth_token:
            st.session_state['user_authenticated'] = True
            st.session_state['user_id'] = "authenticated_user" # Placeholder
            st.info("User authenticated via custom token (simulated).")
        else:
            st.session_state['user_authenticated'] = True
            st.session_state['user_id'] = "anonymous_user" # Placeholder
            st.warning("No custom auth token. Proceeding as anonymous user (simulated).")
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        st.session_state['user_authenticated'] = False

if not st.session_state.get('user_authenticated'):
    st.warning("Authentication required to use the app.")
    st.stop()


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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="analytics-box">', unsafe_allow_html=True)
st.markdown("## 📊 Screening Analytics Dashboard")

# --- Load Data from Firestore ---
df = pd.DataFrame() # Initialize an empty DataFrame

app_id = os.environ.get('__app_id', 'default-app-id')
public_collection_ref = db.collection('artifacts').document(app_id).collection('public').document('data').collection('screening_results')

try:
    doc_ref = public_collection_ref.document('latest_results')
    doc = doc_ref.get()
    if doc.exists:
        firestore_data = doc.to_dict()
        if 'data' in firestore_data and firestore_data['data']:
            df = pd.DataFrame(firestore_data['data'])
            st.info("✅ Loaded screening results from Firestore.")
        else:
            st.info("Firestore document 'latest_results' found, but no screening data within it.")
    else:
        st.warning("⚠️ No screening data found in Firestore. Please run the Resume Screener first.")
        st.stop() # Stop execution if no data is available
except Exception as e:
    st.error(f"Error loading results from Firestore: {e}")
    st.warning("⚠️ No screening data could be loaded. Please run the Resume Screener first.")
    st.stop() # Stop execution if data loading fails

# Check if DataFrame is still empty after loading attempts
if df.empty:
    st.info("No data available for analytics. Please screen some resumes first.")
    st.stop()

# --- Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("📈 Avg. Score", f"{df['Score (%)'].mean():.2f}%")
col2.metric("🧓 Avg. Experience", f"{df['Years Experience'].mean():.1f} yrs")
# Assuming a default cutoff of 80 for analytics dashboard if not explicitly passed
shortlisted_count = (df['Score (%)'] >= 80).sum() 
col3.metric("✅ Shortlisted", f"{shortlisted_count}")

st.divider()

# --- Top Candidates ---
st.markdown("### 🥇 Top 5 Candidates by Score")
# Dynamically select columns to avoid KeyError if 'Candidate Name' is somehow missing
display_columns = ['File Name', 'Score (%)', 'Years Experience']
if 'Candidate Name' in df.columns:
    display_columns.append('Candidate Name')
st.dataframe(df.sort_values(by="Score (%)", ascending=False).head(5)[display_columns], use_container_width=True)

# --- WordCloud ---
if 'Matched Keywords' in df.columns and not df['Matched Keywords'].empty:
    st.markdown("### ☁️ Common Skills WordCloud")
    all_keywords = [kw.strip() for kws in df['Matched Keywords'].dropna() for kw in kws.split(',') if kw.strip()]
    if all_keywords: # Ensure there are keywords before generating word cloud
        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_keywords))
        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.info("No common skills to display in the WordCloud.")
else:
    st.info("No 'Matched Keywords' data available for WordCloud.")


# --- Missing Skills ---
if 'Missing Skills' in df.columns and not df['Missing Skills'].empty:
    st.markdown("### ❌ Top Missing Skills")
    all_missing = pd.Series([s.strip() for row in df['Missing Skills'].dropna() for s in row.split(',') if s.strip()])
    if not all_missing.empty: # Ensure there are missing skills before plotting
        top_missing = all_missing.value_counts().head(10)
        sns.set_style("whitegrid")
        fig_ms, ax_ms = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top_missing.values, y=top_missing.index, ax=ax_ms, palette="coolwarm")
        ax_ms.set_xlabel("Count")
        ax_ms.set_ylabel("Missing Skill")
        st.pyplot(fig_ms)
    else:
        st.info("No top missing skills to display.")
else:
    st.info("No 'Missing Skills' column found in data or it's empty.")


# --- Score Distribution ---
st.markdown("### 📊 Score Distribution")
fig_hist, ax_hist = plt.subplots()
sns.histplot(df['Score (%)'], bins=10, kde=True, color="#00cec9", ax=ax_hist)
ax_hist.set_xlabel("Score (%)")
ax_hist.set_ylabel("Number of Candidates")
st.pyplot(fig_hist)

# --- Experience Distribution ---
st.markdown("### 💼 Experience Distribution")
fig_exp, ax_exp = plt.subplots()
sns.boxplot(x=df['Years Experience'], color="#fab1a0", ax=ax_exp)
ax_exp.set_xlabel("Years of Experience")
st.pyplot(fig_exp)

st.markdown("</div>", unsafe_allow_html=True)
