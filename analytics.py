import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import json # Still needed for general JSON operations if any

# Firebase imports (ensure these are available in your environment)
from firebase_admin import credentials, initialize_app
from firebase_admin import firestore
from firebase_admin import auth # Though auth might not be used directly here, it's common

# Initialize Firebase (only once per app run) - This block is for analytics.py itself
# In a multi-file Streamlit app, it's better to initialize Firebase in main.py
# and pass the db client or rely on it being globally accessible after main.py runs.
# However, for robustness if analytics.py might be run standalone, this block is needed.
# For this setup, we assume main.py handles the primary initialization.
# If you run analytics.py directly, you'd need the full initialization here too.

# Assuming Firebase is initialized by main.py and db is globally accessible
# If not, uncomment and adapt the initialization block from main.py here.
# if 'firebase_initialized' not in st.session_state:
#     try:
#         firebase_config = json.loads(os.environ.get('__firebase_config', '{}'))
#         if not firebase_config:
#             st.error("Firebase configuration not found. Please ensure __firebase_config is set.")
#             st.stop()
#         cred = credentials.Certificate(firebase_config)
#         initialize_app(cred)
#         st.session_state['firebase_initialized'] = True
#         st.success("✅ Firebase initialized successfully in analytics.py!")
#     except Exception as e:
#         st.error(f"❌ Firebase initialization failed in analytics.py: {e}")
#         st.stop()

# db = firestore.client() # Assume db is set by main.py or initialized above if standalone


def app(): # Define the app() function for analytics.py
    st.markdown("""
    <style>
    .analytics-container {
        padding: 2rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        animation: fadeInSlide 0.6s ease-in-out;
    }
    @keyframes fadeInSlide {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="analytics-container">', unsafe_allow_html=True)
    st.subheader("📊 Screening Analytics Dashboard")

    df = pd.DataFrame()
    
    # Access Firestore client (assuming it's initialized globally by main.py)
    try:
        db = firestore.client() # Get the client instance
        app_id = os.environ.get('__app_id', 'default-app-id')
        public_collection_ref = db.collection('artifacts').document(app_id).collection('public').document('data').collection('screening_results')
        
        doc_ref = public_collection_ref.document('latest_results')
        doc = doc_ref.get()
        if doc.exists:
            firestore_data = doc.to_dict()
            if 'data' in firestore_data and firestore_data['data']:
                df = pd.DataFrame(firestore_data['data'])
                st.success("✅ Loaded screening results from Firestore for analytics.")
            else:
                st.info("Firestore document 'latest_results' found, but no screening data within it.")
        else:
            st.warning("⚠️ No screening results found in Firestore. Please run the Resume Screener first.")
            st.markdown('</div>', unsafe_allow_html=True)
            return # Exit if no data
    except Exception as e:
        st.error(f"Error loading results from Firestore in analytics.py: {e}")
        st.warning("⚠️ Falling back to session state if available, or please run Resume Screener.")
        # Fallback to session state if Firestore fails (optional, but good for debugging)
        if 'screening_results' in st.session_state and st.session_state['screening_results']:
            df = pd.DataFrame(st.session_state['screening_results'])
            st.info("✅ Loaded screening results from session state (Firestore fallback).")
        else:
            st.markdown('</div>', unsafe_allow_html=True)
            return # Exit if no data

    # --- Score Distribution ---
    st.markdown("### 📊 Score Distribution")
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(df['Score (%)'], bins=10, kde=True, color="#00cec9", ax=ax_hist)
    ax_hist.set_xlabel("Score (%)")
    ax_hist.set_ylabel("Number of Candidates")
    ax_hist.set_title("Distribution of Candidate Scores")
    st.pyplot(fig_hist)

    # --- Experience vs. Score Scatter Plot ---
    st.markdown("### 📈 Experience vs. Score")
    fig_scatter, ax_scatter = plt.subplots()
    sns.scatterplot(x='Years Experience', y='Score (%)', data=df, hue='Tag', size='Score (%)', sizes=(50, 400), palette='viridis', ax=ax_scatter)
    ax_scatter.set_xlabel("Years of Experience")
    ax_scatter.set_ylabel("Score (%)")
    ax_scatter.set_title("Candidate Experience vs. Screening Score")
    st.pyplot(fig_scatter)

    # --- Top Matched Keywords WordCloud ---
    st.markdown("### ☁️ Top Matched Keywords")
    all_matched_keywords = " ".join(df['Matched Keywords'].dropna().astype(str).tolist())
    if all_matched_keywords:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_matched_keywords)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        ax_wc.set_title("Most Common Matched Keywords")
        st.pyplot(fig_wc)
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

    st.markdown('</div>', unsafe_allow_html=True)

