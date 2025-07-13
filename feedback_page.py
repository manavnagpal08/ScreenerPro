import streamlit as st

def feedback_and_help_page():
    st.markdown("""
        <style>
        .form-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.1);
            max-width: 600px;
            margin: auto;
        }
        .form-container h2 {
            font-size: 2rem;
            color: #2d3436;
            margin-bottom: 1rem;
        }
        .form-container input, .form-container textarea {
            width: 100%;
            padding: 12px;
            margin-top: 8px;
            margin-bottom: 16px;
            border-radius: 10px;
            border: 1px solid #ccc;
            box-shadow: inset 1px 1px 3px rgba(0,0,0,0.05);
            font-size: 1rem;
        }
        .form-container button {
            background-color: #00cec9;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            cursor: pointer;
            transition: 0.3s ease;
        }
        .form-container button:hover {
            background-color: #00b5b0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown("""
    <h2>❓ Feedback & Help</h2>
    <p>We'd love to hear from you. Please fill out the form below:</p>

    <form action="https://formspree.io/f/mwpqevno" method="POST">
        <label for="name">Your Name (Optional)</label>
        <input type="text" name="name" placeholder="e.g. John Doe">

        <label for="email">Your Email (Optional, for reply)</label>
        <input type="email" name="email" placeholder="e.g. you@example.com">

        <label for="subject">Subject</label>
        <input type="text" name="subject" value="Feedback on ScreenerPro">

        <label for="message">Your Message</label>
        <textarea name="message" rows="5" placeholder="Write your message here..."></textarea>

        <button type="submit">📩 Submit Feedback</button>
    </form>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
