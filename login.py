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
        users = json.load(f)
        # Ensure each user has a 'status' key for backward compatibility
        for username, data in users.items():
            if isinstance(data, str): # Old format: "username": "hashed_password"
                users[username] = {"password": data, "status": "active"}
            elif "status" not in data:
                data["status"] = "active"
        return users

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
                    users[new_username] = {"password": hash_password(new_password), "status": "active"}
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
                    users[new_username] = {"password": hash_password(new_password), "status": "active"}
                    save_users(users)
                    st.success(f"✅ User '{new_username}' added successfully!")

def admin_password_reset_section():
    """Admin-driven password reset form."""
    st.subheader("🔑 Reset User Password (Admin Only)")
    users = load_users()
    user_options = [user for user in users.keys() if user != ADMIN_USERNAME] # Cannot reset admin's own password here
    
    if not user_options:
        st.info("No other users to reset passwords for.")
        return

    with st.form("admin_reset_password_form", clear_on_submit=True):
        selected_user = st.selectbox("Select User to Reset Password For", user_options, key="reset_user_select")
        new_password = st.text_input("New Password", type="password", key="new_password_reset")
        reset_button = st.form_submit_button("Reset Password")

        if reset_button:
            if not new_password:
                st.error("Please enter a new password.")
            else:
                users[selected_user]["password"] = hash_password(new_password)
                save_users(users)
                st.success(f"✅ Password for '{selected_user}' has been reset.")

def admin_disable_enable_user_section():
    """Admin-driven user disable/enable form."""
    st.subheader("⛔ Toggle User Status (Admin Only)")
    users = load_users()
    user_options = [user for user in users.keys() if user != ADMIN_USERNAME] # Cannot disable admin's own account here

    if not user_options:
        st.info("No other users to manage status for.")
        return
        
    with st.form("admin_toggle_user_status_form", clear_on_submit=False): # Keep values after submit for easier toggling
        selected_user = st.selectbox("Select User to Toggle Status", user_options, key="toggle_user_select")
        
        current_status = users[selected_user]["status"]
        st.info(f"Current status of '{selected_user}': **{current_status.upper()}**")

        if st.form_submit_button(f"Toggle to {'Disable' if current_status == 'active' else 'Enable'} User"):
            new_status = "disabled" if current_status == "active" else "active"
            users[selected_user]["status"] = new_status
            save_users(users)
            st.success(f"✅ User '{selected_user}' status set to **{new_status.upper()}**.")
            st.rerun() # Rerun to update the displayed status immediately


def login_section():
    """Handles user login and public registration."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None

    if st.session_state.authenticated:
        return True

    login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])

    with login_tab:
        st.subheader("🔐 HR Login")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", key="username_login")
            password = st.text_input("Password", type="password", key="password_login")
            submitted = st.form_submit_button("Login")

            if submitted:
                users = load_users()
                if username not in users:
                    st.error("❌ Invalid username or password")
                else:
                    user_data = users[username]
                    if user_data["status"] == "disabled":
                        st.error("❌ Your account has been disabled. Please contact an administrator.")
                    elif check_password(password, user_data["password"]):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
    
    with register_tab:
        register_section()

    return st.session_state.authenticated

# Helper function to check if the current user is an admin
def is_current_user_admin():
    return st.session_state.get("authenticated", False) and st.session_state.get("username") == ADMIN_USERNAME

# Example of how to use it if running login.py directly for testing
if __name__ == "__main__":
    st.set_page_config(page_title="Login/Register", layout="centered")
    st.title("ScreenerPro Authentication (Test)")
    
    # Ensure admin user exists for testing
    users = load_users()
    if ADMIN_USERNAME not in users:
        users[ADMIN_USERNAME] = {"password": hash_password("adminpass"), "status": "active"} # Set a default admin password for testing
        save_users(users)
        st.info(f"Created default admin user: {ADMIN_USERNAME} with password 'adminpass'")

    if login_section():
        st.write(f"Welcome, {st.session_state.username}!")
        st.write("You are logged in.")
        
        if is_current_user_admin():
            st.markdown("---")
            st.header("Admin Test Section (You are admin)")
            admin_registration_section()
            admin_password_reset_section()
            admin_disable_enable_user_section()

            st.subheader("All Registered Users (Admin View):")
            users_data = load_users()
            display_users = []
            for user, data in users_data.items():
                display_users.append([user, data.get("password", "N/A"), data.get("status", "N/A")])
            st.dataframe(pd.DataFrame(display_users, columns=["Email/Username", "Hashed Password (DO NOT EXPOSE)", "Status"]), use_container_width=True)
            st.warning("This is a test view. Do not expose sensitive data in production.")
        else:
            st.info("Log in as 'admin@screenerpro' to see admin features.")
            
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.pop('username', None)
            st.rerun()
    else:
        st.info("Please login or register to continue.")
