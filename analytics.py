import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import json

# No Firebase imports or initialization here, assuming data comes from session state

def app():
    st.markdown("""
    <style>
    .analytics-container {
        padding: 2rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        animation: fadeIn 0.8s ease-in-out;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .metric-card {
        background: linear-gradient(135deg, #e0f2f7, #f7f9ff);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .chart-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="analytics-container">', unsafe_allow_html=True)
    st.subheader("📊 Screening Analytics Dashboard")

    # Load data from session state
    if 'screening_results' in st.session_state and st.session_state['screening_results']:
        df = pd.DataFrame(st.session_state['screening_results'])
        st.success("✅ Loaded screening results for analytics.")
    else:
        st.warning("⚠️ No screening results found. Please run the Resume Screener first to generate data for analytics.")
        st.markdown('</div>', unsafe_allow_html=True) # Close the container div
        return

    if df.empty:
        st.info("The screening results DataFrame is empty. Please screen some resumes to see analytics.")
        st.markdown('</div>', unsafe_allow_html=True) # Close the container div
        return
    
    # Ensure numerical columns are correctly typed
    for col in ["Score (%)", "Years Experience"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=["Score (%)", "Years Experience"], inplace=True)


    # --- Key Metrics ---
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>Total Resumes</h4><p>{len(df)}</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4>Avg Score</h4><p>{df['Score (%)'].mean():.1f}%</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4>Avg Experience</h4><p>{df['Years Experience'].mean():.1f} yrs</p></div>", unsafe_allow_html=True)
    with col4:
        # Assuming a cutoff score is defined in screener.py (e.g., 80)
        # We need to make sure 'cutoff' and 'min_experience' are consistent or passed.
        # For simplicity, let's assume default values or get them from session_state if set by screener.
        cutoff = st.session_state.get('cutoff_score', 80) # Default to 80 if not set
        min_experience = st.session_state.get('min_exp_required', 2) # Default to 2 if not set
        shortlisted_count = df[(df['Score (%)'] >= cutoff) & (df['Years Experience'] >= min_experience)].shape[0]
        st.markdown(f"<div class='metric-card'><h4>Shortlisted</h4><p>{shortlisted_count}</p></div>", unsafe_allow_html=True)

    st.divider()

    # --- Top Candidates ---
    st.markdown("### 🥇 Top 5 Candidates by Score")
    # Corrected: Use 'File Name' instead of 'Candidate Name'
    st.dataframe(df.sort_values(by="Score (%)", ascending=False).head(5)[['File Name', 'Score (%)', 'Years Experience', 'Feedback']], use_container_width=True)


    # --- WordCloud ---
    if 'Matched Keywords' in df.columns and not df['Matched Keywords'].empty:
        st.markdown("### 🧠 Most Common Keywords")
        all_keywords = pd.Series([s.strip() for row in df['Matched Keywords'].dropna() for s in row.split(',') if s.strip()])
        if not all_keywords.empty:
            word_freq = all_keywords.value_counts().to_dict()
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(word_freq)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        else:
            st.info("No 'Matched Keywords' data available for WordCloud.")
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
    ax_hist.set_xlabel("Score (%)\t")
    ax_hist.set_ylabel("Number of Candidates")
    st.pyplot(fig_hist)


    # --- Experience vs. Score Scatter Plot ---
    st.markdown("### 📈 Experience vs. Score")
    fig_scatter, ax_scatter = plt.subplots()
    sns.scatterplot(x='Years Experience', y='Score (%)', data=df, hue='Tag', palette='viridis', s=100, ax=ax_scatter)
    ax_scatter.set_xlabel("Years of Experience")
    ax_scatter.set_ylabel("Score (%)")
    st.pyplot(fig_scatter)

    st.markdown('</div>', unsafe_allow_html=True) # Close the main container div
