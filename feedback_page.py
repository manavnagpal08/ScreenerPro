import streamlit as st
import urllib.parse

# --- Placeholder for log_user_action function ---
# You should replace this with your actual logging implementation.
def log_user_action(user_email, action, details=None):
    """
    Placeholder for logging user actions.
    Replace with your actual logging logic (e.g., writing to a file, database, or analytics service).
    """
    if details:
        print(f"LOG: User '{user_email}' performed action '{action}' with details: {details}")
    else:
        print(f"LOG: User '{user_email}' performed action '{action}'")

# --- Feedback and Help Page Function ---
def feedback_and_help_page():
    """
    Provides a feedback form.
    Allows users to send feedback via their default email client.
    """
    # Get user email from session_state, defaulting to 'anonymous'
    user_email = st.session_state.get('user_email', 'anonymous')
    log_user_action(user_email, "FEEDBACK_HELP_PAGE_ACCESSED")

    # Reusing screener-container for consistent styling (assuming you have CSS for this)
    st.markdown('<div class="screener-container">', unsafe_allow_html=True)
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
                # *** IMPORTANT: Change this to your actual email address ***
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

# --- Main application logic ---
if __name__ == "__main__":
    # Optional: Set page configuration (can be done once at the top of your main script)
    st.set_page_config(
        page_title="Feedback Page Example",
        page_icon="⭐",
        layout="centered" # or "wide"
    )

    st.title("Welcome to Our App! 🚀")
    st.write("This is a simple example to demonstrate a feedback page in Streamlit.")
    
    # You might have a sidebar or navigation. For this example, we'll just call the feedback page directly.
    # If you had a multi-page app, you'd integrate this function into your page routing logic.

    # Example of setting a user email in session state (e.g., after login)
    if 'user_email' not in st.session_state:
        st.session_state['user_email'] = 'example_user@streamlit.com'

    st.sidebar.header("Navigation")
    if st.sidebar.button("Go to Feedback Page"):
        st.session_state.current_page = "feedback"
    
    # Simple page routing based on session state
    if st.session_state.get('current_page') == "feedback":
        feedback_and_help_page()
    else:
        st.info("Click 'Go to Feedback Page' in the sidebar to see the form.")
