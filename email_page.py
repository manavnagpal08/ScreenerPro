# 📤 Email Candidates Page UI Enhancer
import streamlit as st
import pandas as pd
from email_sender import send_email_to_candidate # Assuming this function is defined elsewhere

# --- Style Enhancements ---
st.markdown("""
<style>
.email-box {
    padding: 2rem;
    background: #f9f9fb;
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.07);
    animation: fadeInUp 0.7s ease;
    margin-top: 1.5rem;
}
.email-entry {
    margin-bottom: 1.2rem;
    padding: 1rem;
    background: white;
    border-radius: 12px;
    border-left: 4px solid #00cec9;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

st.subheader("📧 Send Email to Shortlisted Candidates")

# Retrieve cutoff values from session state, with defaults
cutoff_score = st.session_state.get('screening_cutoff_score', 75) # Default to 75 if not set
min_exp_required = st.session_state.get('screening_min_experience', 2) # Default to 2 if not set

try:
    # Attempt to load from session state first, then from CSV if not found
    if 'screening_results' in st.session_state and st.session_state['screening_results']:
        df = pd.DataFrame(st.session_state['screening_results'])
    else:
        df = pd.read_csv("results.csv") # Fallback to CSV if session state is empty
except FileNotFoundError:
    st.warning("⚠️ No screening results found. Please run the **Resume Screener** first to generate candidate data.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading screening results: {e}. Please ensure data is available.")
    st.stop()

# Filter shortlisted candidates using the cutoff values from session state
shortlisted = df[(df["Score (%)"] >= cutoff_score) & (df["Years Experience"] >= min_exp_required)]

if not shortlisted.empty:
    st.success(f"✅ {len(shortlisted)} candidates shortlisted based on your criteria (Score ≥ {cutoff_score}%, Experience ≥ {min_exp_required} yrs).")
    
    # Display a more informative dataframe for shortlisted candidates
    st.dataframe(
        shortlisted[["Candidate Name", "Score (%)", "Years Experience", "AI Suggestion"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score (%)": st.column_config.ProgressColumn(
                "Score (%)",
                help="Matching score against job requirements",
                format="%f",
                min_value=0,
                max_value=100,
            ),
            "Years Experience": st.column_config.NumberColumn(
                "Years Experience",
                help="Total years of professional experience",
                format="%.1f years",
            ),
            "AI Suggestion": st.column_config.Column(
                "AI Suggestion",
                help="AI's concise overall assessment and recommendation"
            )
        }
    )

    st.markdown("<div class='email-box'>", unsafe_allow_html=True)

    st.markdown("### ✉️ Assign Emails")
    email_map = {}
    for i, row in shortlisted.iterrows():
        candidate_name = row["Candidate Name"]
        # Use existing email if found, otherwise default to empty
        default_email = row.get("Email", "") if row.get("Email") != "Not Found" else ""
        email_input = st.text_input(f"📧 Email for **{candidate_name}**", value=default_email, key=f"email_{i}")
        email_map[candidate_name] = email_input # Map by candidate name for clarity

    st.markdown("### 📝 Customize Email Template")
    subject = st.text_input("Subject", value="🎉 Invitation for Interview - ScreenerPro")
    body_template = st.text_area("Body", height=250, value="""
Hi {{candidate_name}},

Congratulations! 🎉

We were very impressed with your profile and would like to invite you for an interview for the position.

Based on our AI screening, your resume showed:
✅ Score: {{score}}%
💬 AI Assessment: {{ai_suggestion}}

We'll be in touch with next steps shortly.

Warm regards,  
The HR Team
""")

    if st.button("📤 Send All Emails"):
        sent_count = 0
        failed_count = 0
        for _, row in shortlisted.iterrows():
            candidate_name = row["Candidate Name"]
            score = row["Score (%)"]
            ai_suggestion = row["AI Suggestion"] # Use the new column name
            recipient = email_map.get(candidate_name)

            if recipient and "@" in recipient:
                # Replace placeholders in the email body
                message = body_template.replace("{{candidate_name}}", candidate_name)\
                                       .replace("{{score}}", str(score))\
                                       .replace("{{ai_suggestion}}", ai_suggestion) # Use new placeholder
                
                try:
                    send_email_to_candidate(
                        name=candidate_name,
                        score=score,
                        ai_suggestion=ai_suggestion, # Pass the new argument
                        recipient=recipient,
                        subject=subject,
                        message=message
                    )
                    sent_count += 1
                except Exception as e:
                    st.error(f"Failed to send email to {candidate_name} ({recipient}): {e}")
                    failed_count += 1
            else:
                st.warning(f"Skipping email for {candidate_name}: No valid email address provided.")
                failed_count += 1
        
        if sent_count > 0:
            st.success(f"✅ Successfully sent {sent_count} email(s).")
        if failed_count > 0:
            st.warning(f"⚠️ Failed to send {failed_count} email(s). Check console for details.")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning(f"⚠️ No candidates met the defined shortlisting criteria (score ≥ {cutoff_score}% and experience ≥ {min_exp_required} yrs). Please adjust criteria in the **Resume Screener** tab or upload more resumes.")
