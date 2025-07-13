import streamlit as st
import os

# --- JD Folder ---
jd_folder = "data"
os.makedirs(jd_folder, exist_ok=True)

# --- UI Styling ---
st.markdown("""
<style>
.manage-jd-container {
    padding: 2rem;
    background: rgba(255, 255, 255, 0.96);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    animation: fadeSlideUp 0.7s ease-in-out;
    margin-bottom: 2rem;
}
@keyframes fadeSlideUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
h3 {
    color: #00cec9;
    font-weight: 700;
}
.upload-box {
    background: #f9f9f9;
    padding: 1rem;
    border-radius: 10px;
    border: 1px dashed #ccc;
    margin-bottom: 1.5rem; /* Added margin for spacing */
}
.select-box, .text-box {
    background: #fff;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    margin-bottom: 1.5rem; /* Added margin for spacing */
}
.stButton>button {
    width: 100%;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: 600;
    color: white;
    background-color: #00cec9;
    border: none;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #00b0a8;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}
.stDownloadButton>button {
    background-color: #2ecc71; /* Green for download */
}
.stDownloadButton>button:hover {
    background-color: #27ae60;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="manage-jd-container">', unsafe_allow_html=True)
st.markdown("### 📁 Job Description Manager")

# --- JD Upload ---
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.markdown("#### 📤 Upload New JD(s) (.txt, .pdf)")
    # Allow multiple files and specify accepted types
    uploaded_jds = st.file_uploader("Select file(s)", type=["txt", "pdf"], accept_multiple_files=True, key="upload_jds")
    
    if uploaded_jds:
        for uploaded_jd in uploaded_jds:
            jd_path = os.path.join(jd_folder, uploaded_jd.name)
            try:
                with open(jd_path, "wb") as f:
                    f.write(uploaded_jd.read())
                st.success(f"✅ Uploaded: `{uploaded_jd.name}`")
            except Exception as e:
                st.error(f"❌ Error uploading {uploaded_jd.name}: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- JD Listing & Viewer ---
# Update jd_files to include both .txt and .pdf
jd_files = [f for f in os.listdir(jd_folder) if f.endswith((".txt", ".pdf"))]

if jd_files:
    st.markdown('<div class="select-box">', unsafe_allow_html=True)
    selected_jd = st.selectbox("📄 Select JD to view or delete", jd_files)
    st.markdown('</div>', unsafe_allow_html=True)

    if selected_jd:
        file_path = os.path.join(jd_folder, selected_jd)
        file_extension = os.path.splitext(selected_jd)[1].lower()

        st.markdown('<div class="text-box">', unsafe_allow_html=True)
        st.markdown("#### 📜 Job Description Content")

        if file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                jd_content = f.read()
            st.text_area("View or Copy", jd_content, height=300, key="jd_content")
        elif file_extension == ".pdf":
            st.info("📄 This is a PDF file. Content cannot be displayed directly here. Please use the download button below.")
            # Read PDF content as bytes for download
            with open(file_path, "rb") as f:
                pdf_content_bytes = f.read()
            st.download_button("⬇️ Download PDF", data=pdf_content_bytes, file_name=selected_jd, mime="application/pdf")
        else:
            st.warning("Unsupported file type for preview.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"🗑️ Delete `{selected_jd}`"):
                try:
                    os.remove(file_path)
                    st.success(f"🗑️ Deleted: `{selected_jd}`")
                    st.experimental_rerun() # Rerun to update the file list
                except Exception as e:
                    st.error(f"❌ Error deleting {selected_jd}: {e}")
        with col2:
            # If it's a TXT file, provide download button for text
            if file_extension == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    jd_content_for_download = f.read()
                st.download_button("⬇️ Download JD", data=jd_content_for_download, file_name=selected_jd, mime="text/plain")
            # PDF download button is already handled above within the elif block for PDF

        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("📂 No JD files uploaded yet.")

st.markdown('</div>', unsafe_allow_html=True)
