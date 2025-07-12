import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud # Although imported, WordCloud isn't used in the provided dashboard code
import os
import json

# Import the page functions from their respective files
from login import login_section
# from email_sender import send_email_to_candidate # Not directly called, assuming email_page handles this
from screener import resume_screener_page # Import the screener page function
from analytics import analytics_dashboard_page # Import the analytics page function

# You will need to ensure that manage_jds.py, search.py, and notes.py
# define their main logic within functions (e.g., manage_jds_page(), search_page(), candidate_notes_page())
# and that these functions are imported here if you want to call them as functions.
# For now, the original exec(f.read()) approach is kept for these files as per your snippet.
# Example if you refactor them into functions:
# from manage_jds import manage_jds_page
# from search import search_resumes_page
# from notes import candidate_notes_page


# --- Page Config (Should only be in main.py) ---
st.set_page_config(page_title="ScreenerPro – AI Hiring Dashboard", layout="wide", page_icon="🧠")


# --- Dark Mode Toggle ---
dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_main")

# --- Global Fonts & UI Styling & Hiding Specific Streamlit UI Elements ---
# This entire markdown block will contain both global styles, dynamic dark mode styles,
# and the rules for hiding Streamlit UI elements.
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
/* Global Styles - apply to both modes unless overridden */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {'white' if dark_mode else '#333333'}; /* Main text color */
}}

.main .block-container {{
    padding: 2rem;
    border-radius: 20px;
    background: {'#1e1e1e' if dark_mode else 'rgba(255, 255, 255, 0.96)'}; /* Container background */
    box-shadow: 0 12px 30px {'rgba(0,0,0,0.4)' if dark_mode else 'rgba(0,0,0,0.1)'};
    animation: fadeIn 0.8s ease-in-out;
}}

@keyframes fadeIn {{
    0% {{ opacity: 0; transform: translateY(20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

.dashboard-card {{
    padding: 2rem;
    text-align: center;
    font-weight: 600;
    border-radius: 16px;
    background: {'#2a2a2a' if dark_mode else 'linear-gradient(145deg, #f1f2f6, #ffffff)'}; /* Card background */
    border: 1px solid {'#3a3a3a' if dark_mode else '#e0e0e0'};
    box-shadow: 0 6px 18px {'rgba(0,0,0,0.2)' if dark_mode else 'rgba(0,0,0,0.05)'};
    transition: transform 0.2s ease, box-shadow 0.3s ease;
    cursor: pointer;
    color: {'white' if dark_mode else '#333'}; /* Card text color */
}}

.dashboard-card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 10px 24px {'rgba(0,0,0,0.3)' if dark_mode else 'rgba(0,0,0,0.1)'};
    background: {'#3a3a3a' if dark_mode else 'linear-gradient(145deg, #e0f7fa, #f1f1f1)'};
}}

.dashboard-header {{
    font-size: 2.2rem;
    font-weight: 700;
    color: {'#00cec9' if dark_mode else '#222'}; /* Header color, use accent in dark mode */
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #00cec9;
    display: inline-block;
    margin-bottom: 2rem;
    animation: slideInLeft 0.8s ease-out;
}}

@keyframes slideInLeft {{
    0% {{ transform: translateX(-40px); opacity: 0; }}
    100% {{ transform: translateX(0); opacity: 1; }}
}}

.custom-dashboard-button {{
    width: 100%;
    height: 100%;
    padding: 2rem;
    text-align: center;
    font-weight: 600;
    border-radius: 16px;
    background: {'#2a2a2a' if dark_mode else 'linear-gradient(145deg, #f1f2f6, #ffffff)'};
    border: 1px solid {'#3a3a3a' if dark_mode else '#e0e0e0'};
    box-shadow: 0 6px 18px {'rgba(0,0,0,0.2)' if dark_mode else 'rgba(0,0,0,0.05)'};
    transition: transform 0.2s ease, box-shadow 0.3s ease;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: {'white' if dark_mode else '#333'}; /* Button text color */
    min-height: 120px;
}}

.custom-dashboard-button:hover {{
    transform: translateY(-6px);
    box-shadow: 0 10px 24px {'rgba(0,0,0,0.3)' if dark_mode else 'rgba(0,0,0,0.1)'};
    background: {'#3a3a3a' if dark_mode else 'linear-gradient(145deg, #e0f7fa, #f1f1f1)'};
}}

.custom-dashboard-button span {{
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}}

.custom-dashboard-button div {{
    font-size: 1rem;
    font-weight: 600;
}}

/* Streamlit Specific Overrides for Dark Mode Readability */
h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stCode, .stProgress, .stAlert {{
    color: {'white' if dark_mode else '#333333'} !important;
}}

.stAlert {{
    background-color: {'#333333' if dark_mode else 'inherit'} !important;
    color: {'white' if dark_mode else 'inherit'} !important;
    border-color: {'#555555' if dark_mode else 'inherit'} !important;
}}

/* For sidebar elements */
.stSidebar {{
    background-color: {'#1a1a1a' if dark_mode else '#f0f2f6'} !important;
    color: {'white' if dark_mode else '#333333'} !important;
}}
.stSidebar .stRadio div, .stSidebar .stToggle label {{
    color: {'white' if dark_mode else '#333333'} !important;
}}

/* Input fields, text areas, number inputs */
div[data-testid="stTextInput"],
div[data-testid="stTextArea"],
div[data-testid="stNumberInput"] {{
    background-color: {'#2a2a2a' if dark_mode else 'white'};
    color: {'white' if dark_mode else 'black'};
    border: 1px solid {'#3a3a3a' if dark_mode else '#ccc'};
    border-radius: 0.5rem;
}}
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input {{
    background-color: {'#2a2a2a' if dark_mode else 'white'} !important;
    color: {'white' if dark_mode else 'black'} !important;
}}

/* Buttons */
.stButton>button {{
    background-color: {'#007bff' if dark_mode else '#00cec9'} !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 8px {'rgba(0,0,0,0.3)' if dark_mode else 'rgba(0,0,0,0.1)'};
}}
.stButton>button:hover {{
    background-color: {'#0056b3' if dark_mode else '#00a8a3'} !important;
}}


/* --- Start of Hiding Streamlit UI elements CSS --- */

/* Your provided hide_st_style rules */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* More specific and robust hiding rules (from previous suggestions) */
header[data-testid="stHeader"] {{
    display: none !important;
    visibility: hidden !important;
}}

div[data-testid="stToolbar"] {{
    display: none !important;
    visibility: hidden !important;
}}

.stDeployButton {{
    display: none !important;
    visibility: hidden !important;
}}

.viewerBadge_container__1QSob,
.styles_viewerBadge__1yB5_,
.viewerBadge_link__1S137,
.viewerBadge_text__1JaDK,
#GithubIcon,
.css-1jc7ptx, .e1ewe7hr3, .e1ewe7hr1 {{
    display: none !important;
    visibility: hidden !important;
}}

div[data-testid="stConnectionStatus"] {{
    display: none !important;
    visibility: hidden !important;
}}
/* This specific selector was causing issues in some Streamlit versions, including it for completeness */
.st-emotion-cache-ch5fef {{
    display: none !important;
    visibility: hidden !important;
}}

/* --- End of Hiding Streamlit UI elements CSS --- */

</style>
""", unsafe_allow_html=True)

# Set Matplotlib style for dark mode if active
if dark_mode:
    plt.style.use('dark_background')
else:
    plt.style.use('default') # Or 'seaborn-v0_8' or any other light theme you prefer


# --- Branding ---
# Assuming 'logo.png' exists in the same directory
try:
    st.image("logo.png", width=300)
except FileNotFoundError:
    st.warning("Logo file 'logo.png' not found. Please ensure it's in the correct directory.")
st.title("🧠 ScreenerPro – AI Hiring Assistant")

# --- Auth ---
if not login_section():
    st.stop()

# --- Navigation Control ---
default_tab = st.session_state.get("tab_override", "🏠 Dashboard")
tab = st.sidebar.radio("📍 Navigate", [
    "🏠 Dashboard", "🧠 Resume Screener", "📁 Manage JDs", "📊 Screening Analytics",
    "📤 Email Candidates", "🔍 Search Resumes", "📝 Candidate Notes", "🚪 Logout"
], index=[
    "🏠 Dashboard", "🧠 Resume Screener", "📁 Manage JDs", "📊 Screening Analytics",
    "📤 Email Candidates", "🔍 Search Resumes", "📝 Candidate Notes", "🚪 Logout"
].index(default_tab))

if "tab_override" in st.session_state:
    del st.session_state.tab_override

# ======================
# 🏠 Dashboard Section
# ======================
if tab == "🏠 Dashboard":
    st.markdown('<div class="dashboard-header">📊 Overview Dashboard</div>', unsafe_allow_html=True)

    # Initialize metrics
    resume_count = 0
    # Ensure 'data' directory exists before listing files
    jd_count = len([f for f in os.listdir("data") if f.endswith(".txt")]) if os.path.exists("data") else 0
    shortlisted = 0
    avg_score = 0.0
    df_results = pd.DataFrame() # Initialize empty DataFrame

    # Load results from session state
    if 'screening_results' in st.session_state and st.session_state['screening_results']:
        try:
            df_results = pd.DataFrame(st.session_state['screening_results'])
            resume_count = df_results["File Name"].nunique() # Count unique resumes screened
            
            # Retrieve cutoff values from session state, with defaults
            # These keys must match what's stored in screener.py
            cutoff_score = st.session_state.get('screening_cutoff_score', 75) # Default to 75 if not set
            min_exp_required = st.session_state.get('screening_min_experience', 2) # Default to 2 if not set

            shortlisted_df = df_results[(df_results["Score (%)"] >= cutoff_score) & 
                                        (df_results["Years Experience"] >= min_exp_required)]
            shortlisted = shortlisted_df.shape[0]
            avg_score = df_results["Score (%)"].mean()
        except Exception as e:
            st.error(f"Error processing screening results from session state: {e}")
            df_results = pd.DataFrame() # Reset df_results if error occurs
            shortlisted_df = pd.DataFrame() # Ensure this is also reset
    else:
        st.info("No screening results available in this session yet. Please run the Resume Screener.")
        shortlisted_df = pd.DataFrame() # Ensure this is initialized even if no results


    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Make the "Resumes Screened" card interactive
        st.markdown(f"""<div class="dashboard-card">📂 <br><b>{resume_count}</b><br>Resumes Screened</div>""", unsafe_allow_html=True)
        if resume_count > 0:
            with st.expander(f"View {resume_count} Screened Names"):
                for idx, row in df_results.iterrows():
                    st.markdown(f"- **{row['Candidate Name']}** (Score: {row['Score (%)']:.1f}%)")
        elif 'screening_results' in st.session_state and st.session_state['screening_results']:
            st.info("No resumes have been screened yet.")
        else:
            st.info("Run the screener to see screened resumes.")

    col2.markdown(f"""<div class="dashboard-card">📝 <br><b>{jd_count}</b><br>Job Descriptions</div>""", unsafe_allow_html=True)
    
    with col3:
        # Make the "Shortlisted Candidates" card interactive
        st.markdown(f"""<div class="dashboard-card">✅ <br><b>{shortlisted}</b><br>Shortlisted Candidates</div>""", unsafe_allow_html=True)
        if shortlisted > 0:
            with st.expander(f"View {shortlisted} Shortlisted Names"):
                for idx, row in shortlisted_df.iterrows():
                    st.markdown(f"- **{row['Candidate Name']}** (Score: {row['Score (%)']:.1f}%, Exp: {row['Years Experience']:.1f} yrs)")
        elif 'screening_results' in st.session_state and st.session_state['screening_results']:
            st.info("No candidates met the current shortlisting criteria.")
        else:
            st.info("Run the screener to see shortlisted candidates.")


    col4, col5, col6 = st.columns(3)
    col4.markdown(f"""<div class="dashboard-card">📈 <br><b>{avg_score:.1f}%</b><br>Avg Score</div>""", unsafe_allow_html=True)
    
    with col5:
        # This JavaScript snippet will set the session state and trigger a rerun
        # Streamlit will then switch the tab based on the 'tab_override'
        st.markdown("""
        <div class="custom-dashboard-button" onclick="parent.window.postMessage({ type: 'streamlit:setSessionState', args: [{ tab_override: '🧠 Resume Screener' }] }, '*');">
            <span>🧠</span>
            <div>Resume Screener</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        # This JavaScript snippet will set the session state and trigger a rerun
        st.markdown("""
        <div class="custom-dashboard-button" onclick="parent.window.postMessage({ type: 'streamlit:setSessionState', args: [{ tab_override: '📤 Email Candidates' }] }, '*');">
            <span>📤</span>
            <div>Email Candidates</div>
        </div>
        """, unsafe_allow_html=True)

    # Optional: Dashboard Insights
    if not df_results.empty: # Use df_results loaded from session state
        try:
            # Updated Tagging logic from screener.py (ensure consistency with screener.py)
            df_results['Tag'] = df_results.apply(lambda row: 
                "👑 Exceptional Match" if row['Score (%)'] >= 90 and row['Years Experience'] >= 5 and row['Semantic Similarity'] >= 0.85 else (
                "🔥 Strong Candidate" if row['Score (%)'] >= 80 and row['Years Experience'] >= 3 and row['Semantic Similarity'] >= 0.7 else (
                "✨ Promising Fit" if row['Score (%)'] >= 60 and row['Years Experience'] >= 1 else (
                "⚠️ Needs Review" if row['Score (%)'] >= 40 else 
                "❌ Limited Match"))), axis=1)

            st.markdown("### 📊 Dashboard Insights")

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                st.markdown("##### 🔥 Candidate Distribution")
                pie_data = df_results['Tag'].value_counts().reset_index()
                pie_data.columns = ['Tag', 'Count']
                fig_pie, ax1 = plt.subplots(figsize=(4.5, 4.5))
                # Define colors for the pie chart to fit dark/light mode
                if dark_mode:
                    colors = plt.cm.Dark2.colors # A set of colors good for dark backgrounds
                    text_color = 'white'
                else:
                    colors = plt.cm.Pastel1.colors # A set of colors good for light backgrounds
                    text_color = 'black'

                wedges, texts, autotexts = ax1.pie(pie_data['Count'], labels=pie_data['Tag'], autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 10, 'color': text_color})
                for autotext in autotexts:
                    autotext.set_color(text_color) # Ensure percentages are readable
                ax1.axis('equal')
                st.pyplot(fig_pie)
                plt.close(fig_pie) # Close the figure to free up memory

            with col_g2:
                st.markdown("##### 📊 Experience Distribution")
                bins = [0, 2, 5, 10, 20]
                labels = ['0-2 yrs', '3-5 yrs', '6-10 yrs', '10+ yrs']
                df_results['Experience Group'] = pd.cut(df_results['Years Experience'], bins=bins, labels=labels, right=False)
                exp_counts = df_results['Experience Group'].value_counts().sort_index()
                fig_bar, ax2 = plt.subplots(figsize=(5, 4))
                
                # Adjust palette for dark/light mode
                if dark_mode:
                    sns.barplot(x=exp_counts.index, y=exp_counts.values, palette="viridis", ax=ax2)
                else:
                    sns.barplot(x=exp_counts.index, y=exp_counts.values, palette="coolwarm", ax=ax2)
                
                ax2.set_ylabel("Candidates", color='white' if dark_mode else 'black')
                ax2.set_xlabel("Experience Range", color='white' if dark_mode else 'black')
                ax2.tick_params(axis='x', labelrotation=0, colors='white' if dark_mode else 'black')
                ax2.tick_params(axis='y', colors='white' if dark_mode else 'black')
                ax2.title.set_color('white' if dark_mode else 'black') # Title color
                st.pyplot(fig_bar)
                plt.close(fig_bar) # Close the figure to free up memory
            
            # --- Candidate Distribution Summary Table ---
            st.markdown("##### 📋 Candidate Quality Breakdown")
            tag_summary = df_results['Tag'].value_counts().reset_index()
            tag_summary.columns = ['Candidate Tag', 'Count']
            st.dataframe(tag_summary, use_container_width=True, hide_index=True)


            # 📋 Top 5 Most Common Skills - Enhanced & Resized
            st.markdown("##### 🧠 Top 5 Most Common Skills")

            if 'Matched Keywords' in df_results.columns: # Use df_results
                all_skills = []
                for skills in df_results['Matched Keywords'].dropna(): # Use df_results
                    # The Matched Keywords are already comma-separated and cleaned by screener.py
                    all_skills.extend([s.strip().lower() for s in skills.split(",") if s.strip()])

                skill_counts = pd.Series(all_skills).value_counts().head(5)

                if not skill_counts.empty:
                    fig_skills, ax3 = plt.subplots(figsize=(5.8, 3))
                    
                    if dark_mode:
                        palette = sns.color_palette("magma", len(skill_counts))
                    else:
                        palette = sns.color_palette("cool", len(skill_counts))

                    sns.barplot(
                        x=skill_counts.values,
                        y=skill_counts.index,
                        palette=palette,
                        ax=ax3
                    )
                    ax3.set_title("Top 5 Skills", fontsize=13, fontweight='bold', color='white' if dark_mode else 'black')
                    ax3.set_xlabel("Frequency", fontsize=11, color='white' if dark_mode else 'black')
                    ax3.set_ylabel("Skill", fontsize=11, color='white' if dark_mode else 'black')
                    ax3.tick_params(labelsize=10, colors='white' if dark_mode else 'black')
                    
                    for i, v in enumerate(skill_counts.values):
                        ax3.text(v + 0.3, i, str(v), color='white' if dark_mode else 'black', va='center', fontweight='bold', fontsize=9)

                    fig_skills.tight_layout()
                    st.pyplot(fig_skills)
                    plt.close(fig_skills) # Close the figure to free up memory
                else:
                    st.info("No skill data available in results for the Top 5 Skills chart.")

            else:
                st.info("No 'Matched Keywords' column found in results for skill analysis.")

        except Exception as e: # Catch specific exceptions or log for debugging
            st.warning(f"⚠️ Could not render insights due to data error: {e}")

# ======================
# Page Routing via function calls
# ======================
elif tab == "🧠 Resume Screener":
    resume_screener_page() # Call the function from screener.py

elif tab == "📁 Manage JDs":
    # Ensure manage_jds.py exists in the same directory and its logic is not in a function
    # If you have a function like 'manage_jds_page' in manage_jds.py, import and call it:
    # from manage_jds import manage_jds_page
    # manage_jds_page()
    # Otherwise, if it's a script meant to be executed directly:
    try:
        with open("manage_jds.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.error("`manage_jds.py` not found. Please ensure the file exists in the same directory.")


elif tab == "📊 Screening Analytics":
    analytics_dashboard_page() # Call the function from analytics.py

elif tab == "📤 Email Candidates":
    # Ensure email_page.py exists in the same directory and its logic is not in a function
    # If you have a function like 'email_candidates_page' in email_page.py, import and call it:
    # from email_page import email_candidates_page
    # email_candidates_page()
    # Otherwise, if it's a script meant to be executed directly:
    try:
        with open("email_page.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.error("`email_page.py` not found. Please ensure the file exists in the same directory.")

elif tab == "🔍 Search Resumes":
    # Ensure search.py exists in the same directory and its logic is not in a function
    # If you have a function like 'search_resumes_page' in search.py, import and call it:
    # from search import search_resumes_page
    # search_resumes_page()
    # Otherwise, if it's a script meant to be executed directly:
    try:
        with open("search.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.error("`search.py` not found. Please ensure the file exists in the same directory.")

elif tab == "📝 Candidate Notes":
    # Ensure notes.py exists in the same directory and its logic is not in a function
    # If you have a function like 'candidate_notes_page' in notes.py, import and call it:
    # from notes import candidate_notes_page
    # candidate_notes_page()
    # Otherwise, if it's a script meant to be executed directly:
    try:
        with open("notes.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.error("`notes.py` not found. Please ensure the file exists in the same directory.")

elif tab == "🚪 Logout":
    st.session_state.authenticated = False
    st.success("✅ Logged out.")
    st.stop()

# --- About Us Section in Sidebar (Moved to be always visible at the bottom) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### About ScreenerPro")
st.sidebar.info(
    "ScreenerPro is an AI-powered hiring assistant designed to streamline your "
    "recruitment process. It helps you quickly screen resumes, manage job descriptions, "
    "and analyze candidate data to find the best fit for your roles."
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Connect with Manav Nagpal")
st.sidebar.markdown(
    "[LinkedIn Profile](https://www.linkedin.com/in/manav-nagpal-83b935209/) "
    "&nbsp; 🔗" # Using a link emoji as a simple icon
)
st.sidebar.markdown(
    "[Portfolio Website](https://manavnagpal.netlify.app/) "
    "&nbsp; 🌐" # Using a globe emoji for portfolio
)
