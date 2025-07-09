import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

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

# --- Load Data ---
data_source = "results.csv"
uploaded = st.file_uploader("📥 Upload Screening Results CSV", type="csv", help="Optional: Upload custom CSV to view analytics.")

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("✅ Uploaded CSV loaded successfully.")
elif os.path.exists(data_source):
    df = pd.read_csv(data_source)
    st.info("📁 Loaded existing results from `results.csv`.")
else:
    st.warning("⚠️ No data found. Please upload a CSV or run the screener first.")
    st.stop()

# --- Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("📈 Avg. Score", f"{df['Score (%)'].mean():.2f}%")
col2.metric("🧓 Avg. Experience", f"{df['Years Experience'].mean():.1f} yrs")
col3.metric("✅ Shortlisted", f"{(df['Score (%)'] >= 80).sum()}")

st.divider()

# --- Top Candidates ---
st.markdown("### 🥇 Top 5 Candidates by Score")
st.dataframe(df.sort_values(by="Score (%)", ascending=False).head(5)[['File Name', 'Score (%)', 'Years Experience']], use_container_width=True)

# --- WordCloud ---
if 'Matched Keywords' in df.columns:
    st.markdown("### ☁️ Common Skills WordCloud")
    all_keywords = [kw.strip() for kws in df['Matched Keywords'].dropna() for kw in kws.split(',')]
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_keywords))
    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
else:
    st.info("No 'Matched Keywords' data available.")

# --- Missing Skills ---
if 'Missing Skills' in df.columns:
    st.markdown("### ❌ Top Missing Skills")
    all_missing = pd.Series([s.strip() for row in df['Missing Skills'].dropna() for s in row.split(',')])
    top_missing = all_missing.value_counts().head(10)
    sns.set_style("whitegrid")
    fig_ms, ax_ms = plt.subplots(figsize=(8, 4))
    sns.barplot(x=top_missing.values, y=top_missing.index, ax=ax_ms, palette="coolwarm")
    ax_ms.set_xlabel("Count")
    ax_ms.set_ylabel("Missing Skill")
    st.pyplot(fig_ms)
else:
    st.info("No 'Missing Skills' column found in data.")

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
