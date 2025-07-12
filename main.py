import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import json

# Import the page functions from their respective files
# Updated imports for new admin functions
from login import (
    login_section, load_users, admin_registration_section,
    admin_password_reset_section, admin_disable_enable_user_section, # NEW IMPORTS
    is_current_user_admin
)
from email_sender import send_email_to_candidate
from screener import resume_screener_page
from analytics import analytics_dashboard_page


# --- Page Config ---
st.set_page_config(page_title="ScreenerPro – AI Hiring Dashboard", layout="wide", page_icon="🧠")


# --- Dark Mode Toggle ---
# If you want to keep dark mode functionality, you'll need to implement
# Streamlit's native dark mode setting if available or use CSS dynamically.
# For now, this toggle will just exist without custom CSS effects.
dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_main")

# --- Branding ---
try:
    st.image("logo.png", width=300)
except FileNotFoundError:
    st.warning("Logo file 'logo.png' not found. Please ensure it's in the correct directory.")
st.title("🧠 ScreenerPro – AI Hiring Assistant")

# --- Auth ---
if not login_section():
    st.stop()

# Determine if the logged-in user is an admin
is_admin = is_current_user_admin()

# --- Navigation Control ---
navigation_options = [
    "🏠 Dashboard", "🧠 Resume Screener", "📁 Manage JDs", "📊 Screening Analytics",
    "📤 Email Candidates", "🔍 Search Resumes", "📝 Candidate Notes"
]
if is_admin: # Only add Admin Tools if the user is an admin
    navigation_options.append("⚙️ Admin Tools")
navigation_options.append("🚪 Logout") # Always add Logout last

default_tab = st.session_state.get("tab_override", "🏠 Dashboard")
if default_tab not in navigation_options: # Handle cases where default_tab might be Admin Tools for non-admins
    default_tab = "🏠 Dashboard"

tab = st.sidebar.radio("📍 Navigate", navigation_options, index=navigation_options.index(default_tab))

if "tab_override" in st.session_state:
    del st.session_state.tab_override

# ======================
# 🏠 Dashboard Section
# ======================
if tab == "🏠 Dashboard":
    st.header("📊 Overview Dashboard") # Changed to st.header since custom CSS is removed

    # Initialize metrics
    resume_count = 0
    jd_count = len([f for f in os.listdir("data") if f.endswith(".txt")]) if os.path.exists("data") else 0
    shortlisted = 0
    avg_score = 0.0
    df_results = pd.DataFrame()

    # Load results from session state
    if 'screening_results' in st.session_state and st.session_state['screening_results']:
        try:
            df_results = pd.DataFrame(st.session_state['screening_results'])
            resume_count = df_results["File Name"].nunique()
            
            cutoff_score = st.session_state.get('screening_cutoff_score', 75)
            min_exp_required = st.session_state.get('screening_min_experience', 2)

            shortlisted_df = df_results[
                (df_results["Score (%)"] >= cutoff_score) &
                (df_results["Years Experience"] >= min_exp_required)
            ].copy()
            shortlisted = shortlisted_df.shape[0]
            avg_score = df_results["Score (%)"].mean()
        except Exception as e:
            st.error(f"Error processing screening results from session state: {e}")
            df_results = pd.DataFrame()
            shortlisted_df = pd.DataFrame()
    else:
        st.info("No screening results available in this session yet. Please run the Resume Screener.")
        shortlisted_df = pd.DataFrame()

    # Load registered user count for dashboard card
    registered_users_count = 0
    try:
        users_data = load_users()
        registered_users_count = len(users_data)
    except Exception as e:
        st.warning(f"Could not load user data for dashboard count: {e}")


    col1, col2, col3, col_users = st.columns(4)

    with col1:
        # Reverted to basic markdown for dashboard cards
        st.metric(label="Resumes Screened", value=resume_count)
        if resume_count > 0:
            with st.expander(f"View {resume_count} Screened Names"):
                for idx, row in df_results.iterrows():
                    st.markdown(f"- **{row['Candidate Name']}** (Score: {row['Score (%)']:.1f}%)")
        elif 'screening_results' in st.session_state and st.session_state['screening_results']:
            st.info("No resumes have been screened yet.")
        else:
            st.info("Run the screener to see screened resumes.")

    with col2:
        st.metric(label="Job Descriptions", value=jd_count)

    with col3:
        st.metric(label="Shortlisted Candidates", value=shortlisted)
        if shortlisted > 0:
            with st.expander(f"View {shortlisted} Shortlisted Names"):
                for idx, row in shortlisted_df.iterrows():
                    st.markdown(f"- **{row['Candidate Name']}** (Score: {row['Score (%)']:.1f}%, Exp: {row['Years Experience']:.1f} yrs)")
        elif 'screening_results' in st.session_state and st.session_state['screening_results']:
            st.info("No candidates met the current shortlisting criteria.")
        else:
            st.info("Run the screener to see shortlisted candidates.")

    with col_users:
        st.metric(label="Registered Users", value=registered_users_count)


    col4, col5, col6 = st.columns(3)
    col4.metric(label="Avg Score", value=f"{avg_score:.1f}%")

    # Reverted custom buttons to standard Streamlit buttons or links
    with col5:
        if st.button("🧠 Resume Screener"):
            st.session_state.tab_override = '🧠 Resume Screener'
            st.rerun() # Trigger rerun to change tab
    with col6:
        if st.button("📤 Email Candidates"):
            st.session_state.tab_override = '📤 Email Candidates'
            st.rerun() # Trigger rerun to change tab

    # Optional: Dashboard Insights
    if not df_results.empty:
        try:
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
                # Adjust colors and text for default Streamlit theme
                colors = plt.cm.Pastel1.colors
                text_color = 'black' # Default text color for plots without dark_mode styling
                wedges, texts, autotexts = ax1.pie(pie_data['Count'], labels=pie_data['Tag'], autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 10, 'color': text_color})
                for autotext in autotexts:
                    autotext.set_color(text_color)
                ax1.axis('equal')
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            with col_g2:
                st.markdown("##### 📊 Experience Distribution")
                bins = [0, 2, 5, 10, 20]
                labels = ['0-2 yrs', '3-5 yrs', '6-10 yrs', '10+ yrs']
                df_results['Experience Group'] = pd.cut(df_results['Years Experience'], bins=bins, labels=labels, right=False)
                exp_counts = df_results['Experience Group'].value_counts().sort_index()
                fig_bar, ax2 = plt.subplots(figsize=(5, 4))
                
                # Use default palette as custom dark mode styling is gone
                sns.barplot(x=exp_counts.index, y=exp_counts.values, palette="viridis", ax=ax2)
                
                ax2.set_ylabel("Candidates")
                ax2.set_xlabel("Experience Range")
                ax2.tick_params(axis='x', labelrotation=0)
                st.pyplot(fig_bar)
                plt.close(fig_bar)
            
            st.markdown("##### 📋 Candidate Quality Breakdown")
            tag_summary = df_results['Tag'].value_counts().reset_index()
            tag_summary.columns = ['Candidate Tag', 'Count']
            st.dataframe(tag_summary, use_container_width=True, hide_index=True)


            st.markdown("##### 🧠 Top 5 Most Common Skills")

            if 'Matched Keywords' in df_results.columns:
                all_skills = []
                for skills in df_results['Matched Keywords'].dropna():
                    all_skills.extend([s.strip().lower() for s in skills.split(",") if s.strip()])

                skill_counts = pd.Series(all_skills).value_counts().head(5)

                if not skill_counts.empty:
                    fig_skills, ax3 = plt.subplots(figsize=(5.8, 3))
                    
                    palette = sns.color_palette("magma", len(skill_counts))

                    sns.barplot(
                        x=skill_counts.values,
                        y=skill_counts.index,
                        palette=palette,
                        ax=ax3
                    )
                    ax3.set_title("Top 5 Skills", fontsize=13, fontweight='bold')
                    ax3.set_xlabel("Frequency", fontsize=11)
                    ax3.set_ylabel("Skill", fontsize=11)
                    ax3.tick_params(labelsize=10)
                    
                    for i, v in enumerate(skill_counts.values):
                        ax3.text(v + 0.3, i, str(v), va='center', fontweight='bold', fontsize=9)

                    fig_skills.tight_layout()
                    st.pyplot(fig_skills)
                    plt.close(fig_skills)
                else:
                    st.info("No skill data available in results for the Top 5 Skills chart.")

            else:
                st.info("No 'Matched Keywords' column found in results for skill analysis.")

        except Exception as e:
            st.warning(f"⚠️ Could not render insights due to data error: {e}")

# ======================
# ⚙️ Admin Tools Section
# ======================
elif tab == "⚙️ Admin Tools":
    st.header("⚙️ Admin Tools")
    if is_admin:
        st.write("Welcome, Administrator! Here you can manage user accounts.")
        st.markdown("---")

        admin_registration_section() # Create New User Form
        st.markdown("---")

        admin_password_reset_section() # Reset User Password Form
        st.markdown("---")

        admin_disable_enable_user_section() # Disable/Enable User Form
        st.markdown("---")

        st.subheader("👥 All Registered Users")
        st.warning("⚠️ **SECURITY WARNING:** This table displays usernames (email IDs) and **hashed passwords**. This is for **ADMINISTRATIVE DEBUGGING ONLY IN A SECURE ENVIRONMENT**. **NEVER expose this in a public or production application.**")
        try:
            users_data = load_users()
            if users_data:
                display_users = []
                for user, data in users_data.items():
                    # Ensure compatibility with old entries that might just be a string (hashed_password)
                    hashed_pass = data.get("password", data) if isinstance(data, dict) else data
                    status = data.get("status", "N/A") if isinstance(data, dict) else "N/A"
                    display_users.append([user, hashed_pass, status])
                st.dataframe(pd.DataFrame(display_users, columns=["Email/Username", "Hashed Password (DO NOT EXPOSE)", "Status"]), use_container_width=True)
            else:
                st.info("No users registered yet.")
        except Exception as e:
            st.error(f"Error loading user data: {e}")
    else:
        st.error("🔒 Access Denied: You must be an administrator to view this page.")

# ======================
# Page Routing via function calls (remaining pages)
# ======================
elif tab == "🧠 Resume Screener":
    resume_screener_page()

elif tab == "📁 Manage JDs":
    try:
        with open("manage_jds.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.error("`manage_jds.py` not found. Please ensure the file exists in the same directory.")


elif tab == "📊 Screening Analytics":
    analytics_dashboard_page()

elif tab == "📤 Email Candidates":
    send_email_to_candidate()

elif tab == "🔍 Search Resumes":
    try:
        with open("search.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.error("`search.py` not found. Please ensure the file exists in the same directory.")

elif tab == "📝 Candidate Notes":
    try:
        with open("notes.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.error("`notes.py` not found. Please ensure the file exists in the same directory.")

elif tab == "🚪 Logout":
    st.session_state.authenticated = False
    st.session_state.pop('username', None)
    st.success("✅ Logged out.")
    st.stop()
