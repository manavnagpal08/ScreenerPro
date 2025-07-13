import streamlit as st
import requests

def feedback_and_help_page():
    st.markdown("""
        <style>
        .form-box {
            background: rgba(255, 255, 255, 0.95);
            padding: 2.5rem;
            border-radius: 20px;
            max-width: 600px;
            margin: auto;
            box-shadow: 0 12px 30px rgba(0,0,0,0.1);
        }
        .form-box h2 {
            font-size: 2rem;
            color: #2d3436;
            margin-bottom: 1rem;
        }
        .form-box label {
            font-weight: 600;
            margin-top: 1rem;
            display: block;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="form-box">', unsafe_allow_html=True)
    st.markdown("## ❓ Feedback & Help")
    st.markdown("We'd love to hear from you. Please fill out the form below:")

    with st.form("feedback_form"):
        name = st.text_input("Your Name (Optional)")
        email = st.text_input("Your Email (Optional, for reply)")
        subject = st.text_input("Subject", value="Feedback on ScreenerPro")
        message = st.text_area("Your Message", height=150)

        submit = st.form_submit_button("📩 Submit Feedback")

        if submit:
            if not message.strip():
                st.error("Please enter a message before submitting.")
            else:
                formspree_url = "https://formspree.io/f/mwpqevno"  # ✅ Your endpoint

                payload = {
                    "name": name,
                    "email": email,
                    "subject": subject,
                    "message": message
                }

                try:
                    response = requests.post(formspree_url, data=payload)
                    if response.status_code in (200, 202):
                        st.success("✅ Thank you! Your feedback has been submitted successfully.")
                    else:
                        st.error("❌ Something went wrong. Please try again later.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
