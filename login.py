import streamlit as st
import json
import bcrypt # For secure password hashing

# File to store user credentials
USER_DB_FILE = "users.json"

def load_users():
    """Loads user data from the JSON file."""
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w") as f:
            json.dump({}, f) # Create an empty JSON object if file doesn't exist
    with open(USER_DB_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    """Saves user data to the JSON file."""
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4) # Use indent for readability

def hash_password(password):
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    """Checks a password against its bcrypt hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def register_section():
    st.subheader("📝 New User Registration")
    with st.form("registration_form", clear_on_submit=True):
        new_username = st.text_input("Choose Username (Email address recommended)", key="new_username_reg")
        new_password = st.text_input("Choose Password", type="password", key="new_password_reg")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password_reg")
        register_button = st.form_submit_button("Register")

        if register_button:
            if not new_username or not new_password or not confirm_password:
                st.error("Please fill in all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                users = load_users()
                if new_username in users:
                    st.error("Username already exists. Please choose a different one.")
                else:
                    users[new_username] = hash_password(new_password)
                    save_users(users)
                    st.success("✅ Registration successful! You can now log in.")
                    # Optionally clear inputs after successful registration
                    st.session_state["new_username_reg"] = ""
                    st.session_state["new_password_reg"] = ""
                    st.session_state["confirm_password_reg"] = ""

def login_section():
    """Handles user login and registration."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True # User is already authenticated

    # Display login/register tabs
    login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])

    with login_tab:
        st.subheader("🔐 HR Login")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", key="username_login")
            password = st.text_input("Password", type="password", key="password_login")
            submitted = st.form_submit_button("Login")

            if submitted:
                users = load_users()
                if username in users and check_password(password, users[username]):
                    st.session_state.authenticated = True
                    st.session_state.username = username # Store username in session state
                    st.success("✅ Login successful!")
                    st.rerun() # Rerun to refresh the app state after login
                else:
                    st.error("❌ Invalid username or password")
    
    with register_tab:
        register_section() # Call the registration function within its tab

    return st.session_state.authenticated

# Example of how to use it if running login.py directly for testing
if __name__ == "__main__":
    import os # Import os for file existence check

    st.set_page_config(page_title="Login/Register", layout="centered")

    st.title("ScreenerPro Authentication")

    if login_section():
        st.write(f"Welcome, {st.session_state.username}!")
        st.write("You are logged in.")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.pop('username', None)
            st.rerun()
    else:
        st.info("Please login or register to continue.")
