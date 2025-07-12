import streamlit as st
import json
import bcrypt
import os

# File to store user credentials
USER_DB_FILE = "users.json"
ADMIN_USERNAME = "admin@forscreenerpro" # Define your admin username here

def load_users():
    """Loads user data from the JSON file."""
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w") as f:
            json.dump({}, f)
    with open(USER_DB_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    """Saves user data to the JSON file."""
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    """Checks a password against its bcrypt hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def register_section():
    """Public self-registration form."""
    st.subheader("📝 New User Registration")
    with st.form("registration_form", clear_on_submit=True):
        new_username = st.text_input("Choose Username (Email address recommended)", key="new_username_reg_public")
        new_password = st.text_input("Choose Password", type="password", key="new_password_reg_public")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password_reg_public")
        register_button = st.form_submit_button("Register New Account")

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

def admin_registration_section():
    """Admin-driven user creation form."""
    st.subheader("➕ Create New User Account (Admin Only)")
    with st.form("admin_registration_form", clear_on_submit=True):
        new_username = st.text_input("New User's Username (Email)", key="new_username_admin_reg")
        new_password = st.text_input("New User's Password", type="password", key="new_password_admin_reg")
        admin_register_button = st.form_submit_button("Add New User")

        if admin_register_button:
            if not new_username or not new_password:
                st.error("Please fill in all fields.")
            else:
                users = load_users()
                if new_username in users:
                    st.error(f"User '{new_username}' already exists.")
                else:
                    users[new_username] = hash_password(new_password)
                    save_users(users)
                    st.success(f"✅ User '{new_username}' added successfully!")
                    # You might want to automatically clear inputs if not using clear_on_submit=True
                    # st.session_state["new_username_admin_reg"] = ""
                    # st.session_state["new_password_admin_reg"] = ""

def login_section():
    """Handles user login and public registration."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state: # Ensure username is initialized
        st.session_state.username = None

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
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
    
    with register_tab:
        register_section() # Call the public registration function within its tab

    return st.session_state.authenticated

# Helper function to check if the current user is an admin
def is_current_user_admin():
    return st.session_state.get("authenticated", False) and st.session_state.get("username") == ADMIN_USERNAME

# Example of how to use it if running login.py directly for testing
if __name__ == "__main__":
    st.set_page_config(page_title="Login/Register", layout="centered")

    st.title("ScreenerPro Authentication (Test Mode)")

    if login_section():
        st.write(f"Welcome, {st.session_state.username}!")
        st.write("You are logged in.")
        
        # Test the admin section
        if is_current_user_admin():
            st.markdown("---")
            st.header("Admin Test Section")
            admin_registration_section()
            st.subheader("All Registered Users (Test View - Admin Only):")
            users_data = load_users()
            st.dataframe(pd.DataFrame(users_data.items(), columns=["Email/Username", "Hashed Password"]), use_container_width=True)
            st.warning("This is a test view. Do not expose sensitive data in production.")
        else:
            st.info("Log in as 'admin@screenerpro' to see admin features.")
            
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.pop('username', None)
            st.rerun()
    else:
        st.info("Please login or register to continue.")
