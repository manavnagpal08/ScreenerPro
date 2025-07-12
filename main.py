import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import json

# Import the page functions from their respective files
from login import login_section
from email_sender import send_email_to_candidate
from screener import resume_screener_page # Import the screener page function
from analytics import analytics_dashboard_page # Import the analytics page function


# --- Page Config (Should only be in main.py) ---
st.set_page_config(page_title="ScreenerPro – AI Hiring Dashboard", layout="wide", page_icon="🧠")


# --- Dark Mode Toggle ---
dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_main")
# Determine the current background color based on dark mode state
# This variable will be used in the injected CSS
current_bg_color = "#FFFFFF" # Default light mode background
if dark_mode:
    st.markdown("""
    <style>
    body { background-color: #121212 !important; color: white !important; }
    .block-container { background-color: #1e1e1e !important; }
    </style>
    """, unsafe_allow_html=True)
    current_bg_color = "#121212" # Dark mode background

# --- Global Fonts & UI Styling & Specific Streamlit UI Element Hiding ---
# Note: I'm embedding the current_bg_color into the CSS string directly.
# This makes the CSS dynamic based on your dark mode toggle.
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main .block-container {
    padding: 2rem;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 12px 30px rgba(0,0,0,0.1);
    animation: fadeIn 0.8s ease-in-out;
}
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
.dashboard-card {
    padding: 2rem;
    text-align: center;
    font-weight: 600;
    border-radius: 16px;
    background: linear-gradient(145deg, #f1f2f6, #ffffff);
    border: 1px solid #e0e0e0;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.3s ease;
    cursor: pointer;
}
.dashboard-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.1);
    background: linear-gradient(145deg, #e0f7fa, #f1f1f1);
}
.dashboard-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #222;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #00cec9;
    display: inline-block;
    margin-bottom: 2rem;
    animation: slideInLeft 0.8s ease-out;
}
@keyframes slideInLeft {
    0% { transform: translateX(-40px); opacity: 0; }
    100% { transform: translateX(0); opacity: 1; }
}
/* New CSS for custom buttons to look like cards */
.custom-dashboard-button {
    width: 100%;
    height: 100%; /* Ensure it takes full height of its column */
    padding: 2rem;
    text-align: center;
    font-weight: 600;
    border-radius: 16px;
    background: linear-gradient(145deg, #f1f2f6, #ffffff);
    border: 1px solid #e0e0e0;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.3s ease;
    cursor: pointer;
    display: flex;
    flex-direction: column; /* Stack icon and text vertically */
    justify-content: center;
    align-items: center;
    color: #333; /* Ensure text color is visible */
    min-height: 120px; /* Ensure a consistent height for the buttons */
}
.custom-dashboard-button:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.1);
    background: linear-gradient(145deg, #e0f7fa, #f1f1f1);
}
.custom-dashboard-button span { /* For the icon */
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}
.custom-dashboard-button div { /* For the text */
    font-size: 1rem;
    font-weight: 600;
}

/* --- Start of "invisibilize" CSS --- */

/* Target the main header bar and make its background transparent */
/* This is crucial for the top-right elements (Fork, Share) to blend in */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    color: {current_bg_color} !important; /* Try to make text/icons blend */
}

/* Target the toolbar/buttons within the header and try to blend them */
div[data-testid="stToolbar"],
.stDeployButton {
    background-color: transparent !important;
    color: {current_bg_color} !important; /* Make icons/text blend */
    border: none !important; /* Remove any borders */
}

/* Specific selectors for the icons and text inside the toolbar/buttons */
.viewerBadge_container__1QSob,
.styles_viewerBadge__1yB5_,
.viewerBadge_link__1S137,
.viewerBadge_text__1JaDK,
#GithubIcon,
.css-1jc7ptx, .e1ewe7hr3, .e1ewe7hr1 {
    color: {current_bg_color} !important; /* Make icons/text blend */
    background-color: transparent !important; /* Ensure background is clear */
    /* If still visible, try to shrink or move off-screen as a last resort */
    /* width: 0 !important; height: 0 !important; overflow: hidden !important; */
    /* font-size: 0 !important; */
}

/* Hide the hamburger menu (if it's not needed, but can hide useful debug options) */
#MainMenu {
    visibility: hidden;
    display: none !important;
}

/* "Hosted with Streamlit" badge at the bottom */
div[data-testid="stConnectionStatus"] {
    background-color: {current_bg_color} !important; /* Match app background */
    color: {current_bg_color} !important; /* Make text blend */
    border: none !important; /* Remove any border */
    box-shadow: none !important; /* Remove any shadow */
    /* Potentially make it smaller or move it if still visible */
    /* transform: scale(0.1); opacity: 0; */
}

/* Fallback for the badge if data-testid changes or isn't present */
.st-emotion-cache-ch5fef { /* This class is often associated with the badge - inspect if it changes */
    background-color: {current_bg_color} !important;
    color: {current_bg_color} !important;
    border: none !important;
    box-shadow: none !important;
}

/* Ensure the Streamlit footer itself also blends if it contains the badge */
footer {
    background-color: {current_bg_color} !important;
    color: {current_bg_color} !important;
    border: none !important;
}

/* --- End of "invisibilize" CSS --- */

</style>
""", unsafe_allow_html=True)


# --- Branding ---
st.image("logo.png", width=300)
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
    
    # Modified buttons to use custom HTML with dashboard-card styling
    with col5:
        st.markdown("""
        <div class="custom-dashboard-button" onclick="window.parent.postMessage({streamlit: {type: 'setSessionState', args: ['tab_override', '🧠 Resume Screener']}}, '*');">
            <span>🧠</span>
            <div>Resume Screener</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown("""
        <div class="custom-dashboard-button" onclick="window.parent.postMessage({streamlit: {type: 'setSessionState', args: ['tab_override', '📤 Email Candidates']}}, '*');">
            <span>📤</span>
            <div>Email Candidates</div>
        </div>
        """, unsafe_allow_html=True)

    # Optional: Dashboard Insights
    if not df_results.empty: # Use df_results loaded from session state
        try:
            # Updated Tagging logic from screener.py
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
                ax1.pie(pie_data['Count'], labels=pie_data['Tag'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
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
                sns.barplot(x=exp_counts.index, y=exp_counts.values, palette="coolwarm", ax=ax2)
                ax2.set_ylabel("Candidates")
                ax2.set_xlabel("Experience Range")
                ax2.tick_params(axis='x', labelrotation=0)
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
    resume_screener_page()

elif tab == "📁 Manage JDs":
    with open("manage_jds.py", encoding="utf-8") as f:
        exec(f.read())

elif tab == "📊 Screening Analytics":
    analytics_dashboard_page()

elif tab == "📤 Email Candidates":
    with open("email_page.py", encoding="utf-8") as f:
        exec(f.read())

elif tab == "🔍 Search Resumes":
    with open("search.py", encoding="utf-8") as f:
        exec(f.read())

elif tab == "📝 Candidate Notes":
    with open("notes.py", encoding="utf-8") as f:
        exec(f.read())

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
