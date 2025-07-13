import streamlit as st
import urllib.parse
from utils.logger import log_user_action

def feedback_and_help_page():
    """
    Provides a feedback form.
    Allows users to send feedback via their default email client.
    """
    user_email = st.session_state.get('user_email', 'anonymous')
    log_user_action(user_email, "FEEDBACK_HELP_PAGE_ACCESSED")

    st.markdown('<div class="screener-container">', unsafe_allow_html=True) # Reusing screener-container for consistent styling
    st.markdown("## ❓ Feedback")
    st.caption("We value your input! Please use the form below to send us your feedback or questions.")

    st.markdown("### Send Us Your Feedback")
    with st.form("feedback_form", clear_on_submit=True):
        feedback_name = st.text_input("Your Name (Optional)", key="feedback_name")
        feedback_email = st.text_input("Your Email (Optional, for reply)", key="feedback_email")
        feedback_subject = st.text_input("Subject", "Feedback on ScreenerPro", key="feedback_subject")
        feedback_message = st.text_area("Your Message", height=150, key="feedback_message")
        
        submit_button = st.form_submit_button("Send Feedback")

        if submit_button:
            if not feedback_message.strip():
                st.error("Please enter your message before sending feedback.")
                log_user_action(user_email, "FEEDBACK_SUBMIT_FAILED", {"reason": "Empty message"})
            else:
                # Define the recipient email for feedback
                # You can change this to your actual email address
                recipient_email = "your_feedback_email@example.com" 
                
                # Construct the email body
                email_body = f"From: {feedback_name if feedback_name else 'Anonymous User'}\n"
                email_body += f"Email: {feedback_email if feedback_email else 'N/A'}\n\n"
                email_body += "Message:\n"
                email_body += feedback_message

                # Encode subject and body for mailto link
                encoded_subject = urllib.parse.quote(feedback_subject)
                encoded_body = urllib.parse.quote(email_body)

                mailto_link = f"mailto:{recipient_email}?subject={encoded_subject}&body={encoded_body}"
                
                st.success("✅ Your feedback is ready to be sent! Click the button below to open your email client.")
                st.markdown(f"""
                    <a href="{mailto_link}" target="_blank">
                        <button style="background-color:#00cec9;color:white;border:none;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;border-radius:8px;">
                            📧 Open Email Client to Send
                        </button>
                    </a>
                """, unsafe_allow_html=True)
                log_user_action(user_email, "FEEDBACK_EMAIL_LINK_GENERATED", {"subject": feedback_subject, "recipient": recipient_email})

    st.markdown("</div>", unsafe_allow_html=True)

