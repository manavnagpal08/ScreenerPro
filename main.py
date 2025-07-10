import streamlit as st
import os
import json # Corrected: Added import for the json module
import pandas as pd # Added for Dashboard section
import matplotlib.pyplot as plt # Added for Dashboard section
import seaborn as sns # Added for Dashboard section
import json # Explicitly import json module
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Firebase imports (moved here for clarity, but already imported below)
from firebase_admin import credentials, initialize_app, auth, firestore

# Import your different application modules.
# Ensure these files exist and have an 'app()' function defined within them.
@@ -19,24 +22,30 @@
# --- Page Config (Should be called only once in main.py) ---
st.set_page_config(page_title="ScreenerPro – AI Hiring Dashboard", layout="wide")

# --- Firebase Initialization (Consistent and Centralized) ---
# This block ensures Firebase is initialized only once and handles authentication.
# --- Firebase Initialization (Consistent and Centralized for Local & Canvas) ---
if 'firebase_initialized' not in st.session_state:
    try:
        # Use the global __firebase_config variable provided by the environment
        firebase_config = json.loads(os.environ.get('__firebase_config', '{}'))
        if not firebase_config:
            st.error("Firebase configuration not found. Please ensure __firebase_config is set in the environment.")
            st.stop() # Stop here if Firebase config is critical for any part of the app

        from firebase_admin import credentials, initialize_app, auth, firestore

        # Check if an app is already initialized to prevent multiple initializations
        if not initialize_app(): # This checks if any app is initialized
            cred = credentials.Certificate(firebase_config)
            initialize_app(cred)
        st.session_state['firebase_initialized'] = True
        # st.success("✅ Firebase initialized successfully!") # Avoid too many success messages
        service_account_key_path = "firebase_service_account.json"
        
        if os.path.exists(service_account_key_path):
            # Use local service account key if found
            cred = credentials.Certificate(service_account_key_path)
            if not initialize_app(): # Check if an app is already initialized
                initialize_app(cred)
            st.session_state['firebase_initialized'] = True
            st.success("✅ Firebase initialized successfully using local service account key!")
        else:
            # Fallback to Canvas environment variable if local key not found
            firebase_config = json.loads(os.environ.get('__firebase_config', '{}'))
            if not firebase_config or not firebase_config.get('apiKey'):
                st.error("Firebase configuration not found. Please ensure 'firebase_service_account.json' is in the root directory or '__firebase_config' is set in the environment.")
                st.stop() # Stop here if Firebase config is critical for any part of the app

            if not initialize_app(): # Check if an app is already initialized
                cred = credentials.Certificate(firebase_config)
                initialize_app(cred)
            st.session_state['firebase_initialized'] = True
            st.success("✅ Firebase initialized successfully using environment config!")

        # Authenticate user (anonymously for simplicity in this demo)
        if 'user_authenticated' not in st.session_state:
@@ -52,9 +61,10 @@
                    st.error(f"Authentication failed (Firebase Admin): {e}")
                    st.session_state['user_authenticated'] = False
            else:
                st.session_state['user_authenticated'] = True
                # For local running, we don't expect __initial_auth_token, so proceed to login
                st.session_state['user_authenticated'] = False # Let login.py handle authentication
                st.session_state['user_id'] = "anonymous_user_id" # Placeholder
                # st.warning("No custom auth token. Proceeding as anonymous user (simulated).")
                # st.warning("No custom auth token. Proceeding to login.")

    except Exception as e:
        st.error(f"❌ Firebase initialization or authentication failed: {e}")
@@ -131,12 +141,13 @@


# --- Authentication Check ---
if not st.session_state.get('user_authenticated'):
# The login.login_section() handles the 'authenticated' state.
# We only proceed to the main app if 'authenticated' is True.
if not st.session_state.get('authenticated'): # Check the 'authenticated' state from login.py
    st.sidebar.title("🔐 HR Login")
    login.login_section() # Call the login section from login.py
    # If login_section returns False (not authenticated), st.stop() will prevent further execution
    if not st.session_state.get('authenticated'): # Check the 'authenticated' state from login.py
        st.stop()
    if not st.session_state.get('authenticated'):
        st.stop() # Stop execution if not logged in
else:
    # If authenticated, show the main application with sidebar navigation
    st.sidebar.title("HR Dashboard")
@@ -159,155 +170,155 @@
    # ======================
    if page == "🏠 Dashboard":
        # Access Firestore client here after initialization
        from firebase_admin import firestore # Re-import firestore here if it's not globally available
        # from firebase_admin import firestore # Already imported at the top
        db = firestore.client()
        app_id = os.environ.get('__app_id', 'default-app-id')
        app_id = os.environ.get('__app_id', 'default-app-id') # Still use for collection path
        public_collection_ref = db.collection('artifacts').document(app_id).collection('public').document('data').collection('screening_results')

        # Initialize metrics
        resume_count = 0
        jd_count = len([f for f in os.listdir("data") if f.endswith(".txt")]) if os.path.exists("data") else 0
        shortlisted = 0
        avg_score = 0.0
        df_results = pd.DataFrame() # Initialize empty DataFrame

        try:
            doc_ref = public_collection_ref.document('latest_results')
            doc = doc_ref.get()
            if doc.exists:
                firestore_data = doc.to_dict()
                if 'data' in firestore_data and firestore_data['data']:
                    df_results = pd.DataFrame(firestore_data['data'])
                    st.info("✅ Loaded screening results from Firestore.")
                else:
                    st.info("Firestore document 'latest_results' found, but no screening data within it.")
            else:
                st.info("No screening results found in Firestore. Please run the Resume Screener.")
        except Exception as e:
            st.error(f"Error loading results from Firestore: {e}")
            df_results = pd.DataFrame() # Reset df_results if error occurs

        if not df_results.empty:
            resume_count = df_results["File Name"].nunique() # Count unique resumes screened

            # Define cutoff for shortlisted candidates (consistent with screener.py)
            cutoff_score = 80 
            min_exp_required = 2

            shortlisted = df_results[(df_results["Score (%)"] >= cutoff_score) & 
                                     (df_results["Years Experience"] >= min_exp_required)].shape[0]
            avg_score = df_results["Score (%)"].mean()

        st.markdown('<div class="dashboard-header">📊 Overview Dashboard</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"""<div class="dashboard-card">📂 <br><b>{resume_count}</b><br>Resumes Screened</div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div class="dashboard-card">📝 <br><b>{jd_count}</b><br>Job Descriptions</div>""", unsafe_allow_html=True)
        col3.markdown(f"""<div class="dashboard-card">✅ <br><b>{shortlisted}</b><br>Shortlisted Candidates</div>""", unsafe_allow_html=True)

        col4, col5, col6 = st.columns(3)
        col4.markdown(f"""<div class="dashboard-card">📈 <br><b>{avg_score:.1f}%</b><br>Avg Score</div>""", unsafe_allow_html=True)
        with col5:
            if st.button("🧠 Resume Screener", use_container_width=True):
                st.session_state.tab_override = "🧠 Resume Screener"
                st.rerun()
        with col6:
            if st.button("📤 Email Candidates", use_container_width=True):
                st.session_state.tab_override = "📤 Email Candidates"
                st.rerun()

        # Optional: Dashboard Insights
        if not df_results.empty: # Use df_results loaded from Firestore
            try:
                df_results['Tag'] = df_results.apply(lambda row:
                    "🔥 Top Talent" if row['Score (%)'] > 90 and row['Years Experience'] >= 3
                    else "✅ Good Fit" if row['Score (%)'] >= 75
                    else "⚠️ Needs Review", axis=1)

                st.markdown("### 📊 Dashboard Insights")

                col_g1, col_g2 = st.columns(2)

                with col_g1:
                    st.markdown("##### 🔥 Candidate Distribution")
                    pie_data = df_results['Tag'].value_counts().reset_index()
                    pie_data.columns = ['Tag', 'Count']
                    fig_pie, ax1 = plt.subplots(figsize=(4.5, 4.5))
                    ax1.pie(pie_data['Count'], labels=pie_data['Tag'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
                    ax1.axis('equal')
                    st.pyplot(fig_pie)

                with col_g2:
                    st.markdown("##### 📊 Experience Distribution")
                    bins = [0, 2, 5, 10, 20]
                    labels = ['0-2 yrs', '3-5 yrs', '6-10 yrs', '10+ yrs']
                    df_results['Experience Group'] = pd.cut(df_results['Years Experience'], bins=bins, labels=labels, right=False)
                    exp_counts = df_results['Experience Group'].value_counts().sort_index()
                    fig_bar, ax2 = plt.subplots(figsize=(5, 4))
                    sns.barplot(x=exp_counts.index, y=exp_counts.values, palette="coolwarm", ax=ax2)
                    ax2.set_ylabel("Candidates")
                    ax2.set_xlabel("Experience Range")
                    ax2.tick_params(axis='x', labelrotation=0)
                    st.pyplot(fig_bar)

                # 📋 Top 5 Most Common Skills - Enhanced & Resized
                st.markdown("##### 🧠 Top 5 Most Common Skills")

                if 'Matched Keywords' in df_results.columns: # Use df_results
                    all_skills = []
                    for skills in df_results['Matched Keywords'].dropna(): # Use df_results
                        all_skills.extend([s.strip().lower() for s in skills.split(",") if s.strip()])

                    skill_counts = pd.Series(all_skills).value_counts().head(5)

                    fig_skills, ax3 = plt.subplots(figsize=(5.8, 3))
                    sns.barplot(
                        x=skill_counts.values,
                        y=skill_counts.index,
                        palette=sns.color_palette("cool", len(skill_counts)),
                        ax=ax3
                    )
                    ax3.set_title("Top 5 Skills", fontsize=13, fontweight='bold')
                    ax3.set_xlabel("Frequency", fontsize=11)
                    ax3.set_ylabel("Skill", fontsize=11)
                    ax3.tick_params(labelsize=10)
                    for i, v in enumerate(skill_counts.values):
                        ax3.text(v + 0.3, i, str(v), color='black', va='center', fontweight='bold', fontsize=9)

                    fig_skills.tight_layout()
                    st.pyplot(fig_skills)

                else:
                    st.info("No skill data available in results.")

            except Exception as e: # Catch specific exceptions or log for debugging
                st.warning(f"⚠️ Could not render insights due to data error: {e}")

# ======================
# Page Routing via function calls
# ======================
    elif page == "🧠 Resume Screener":
        screener.app() # Call the app function in screener.py

    elif page == "📁 Manage JDs":
        manage_jds.app() # Call the app function in manage_jds.py

    elif page == "📊 Screening Analytics":
        analytics.app() # Call the app function in analytics.py

    elif page == "📤 Email Candidates":
        email_page.app() # Call the app function in email_page.py

    elif page == "🔍 Search Resumes":
        search.app() # Call the app function in search.py

    elif page == "📝 Candidate Notes":
        notes.app() # Call the app function in notes.py

    elif page == "🚪 Logout":
        st.session_state.authenticated = False
        st.session_state.firebase_initialized = False # Reset Firebase state on logout
        st.session_state.user_authenticated = False
        st.session_state.user_id = None
        st.success("✅ Logged out. Please refresh the page to log in again.")
        st.stop() # Stop the app after logout
