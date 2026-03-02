"""Main Streamlit application with multi-page navigation."""

import streamlit as st
from upload_page import render_upload_page
from chat_page import render_chat_page
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="IN'SMART",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state() -> None:
    """Initialize Streamlit session state variables.
    
    Should initialize:
    - API URL
    - Conversation history
    - Uploaded documents list
    - Current conversation ID
    """
    pass


def render_sidebar() -> str:
    """Render sidebar navigation and return selected page.
    
    Returns:
        str: Selected page name ('upload' or 'chat')
    """
    pass


def render_home_page() -> None:
    """Render home/welcome page.
    
    Should display:
    - Project overview
    - Quick start instructions
    - System status
    """
    pass


def main() -> None:
    """Main application entry point.
    
    Handles:
    - Session state initialization
    - Page navigation
    - Routing to appropriate page
    """
    initialize_session_state()
    
    # Sidebar navigation
    page = render_sidebar()
    
    # Route to selected page
    if page == "upload":
        render_upload_page()
    elif page == "chat":
        render_chat_page()
    else:
        render_home_page()


if __name__ == "__main__":
    main()
