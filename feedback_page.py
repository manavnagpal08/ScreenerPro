import streamlit as st
import requests
import urllib.parse
from utils.logger import log_user_action

def feedback_and_help_page():
    """
    Provides a feedback form.
    Allows users to send feedback via Formspree.
    The user's email is automatically filled from the session state.
    """
    user_email = st.session_state.get('user_email', 'anonymous')
    log_user_action(user_email, "FEEDBACK_PAGE_ACCESSED")

    # The 'screener-container' CSS is expected to be defined in main.py for global consistency.
    st.markdown('<div class="screener-container">', unsafe_allow_html=True)
    st.markdown("## ❓ Feedback")
    st.caption("We value your input! Please use the form below to send us your feedback or questions.")

    with st.form("feedback_form", clear_on_submit=True):
        feedback_name = st.text_input("Your Name (Optional)", key="feedback_name")

        # Auto-filled email from session, displayed as disabled
        current_user_email = st.session_state.get('user_email', '')
        st.text_input("Your Email (Auto-Filled)", value=current_user_email, disabled=True, key="feedback_email_display")

        feedback_subject = st.text_input("Subject", "Feedback on ScreenerPro", key="feedback_subject")
        feedback_message = st.text_area("Your Message", height=150, key="feedback_message")
        
        submit_button = st.form_submit_button("Send Feedback")

        if submit_button:
            if not feedback_message.strip():
                st.error("❌ Please enter your message before sending feedback.")
                log_user_action(user_email, "FEEDBACK_SUBMIT_FAILED", {"reason": "Empty message"})
            else:
                # Formspree endpoint - REPLACE WITH YOUR ACTUAL FORMSPREE URL
                # Example: "https://formspree.io/f/yourformid"
                formspree_url = "https://formspree.io/f/mwpqevno"  # Your real Formspree link
                
                payload = {
                    "name": feedback_name,
                    "email": current_user_email, # Use the email from session state for submission
                    "subject": feedback_subject,
                    "message": feedback_message
                }

                try:
                    response = requests.post(formspree_url, data=payload)

                    if response.status_code == 200:
                        st.success("✅ Thank you! Your feedback has been submitted successfully.")
                        log_user_action(user_email, "FEEDBACK_SUBMITTED_FORMSPREE", {"subject": feedback_subject, "status_code": response.status_code})
                    else:
                        st.error(f"⚠️ Something went wrong (Status: {response.status_code}). Please try again later.")
                        log_user_action(user_email, "FEEDBACK_SUBMIT_FAILED", {"status_code": response.status_code, "response_text": response.text})
                except requests.exceptions.RequestException as e:
                    st.error(f"⚠️ Network error or problem connecting to Formspree: {e}")
                    log_user_action(user_email, "FEEDBACK_SUBMIT_NETWORK_ERROR", {"error": str(e)})

    st.markdown("</div>", unsafe_allow_html=True)

