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
    if "api_url" not in st.session_state:
        st.session_state.api_url = os.getenv("API_URL", "http://localhost:8000")
    
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    
    if "documents" not in st.session_state:
        st.session_state.documents = []
    
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "home"


def render_sidebar() -> str:
    """Render sidebar navigation and return selected page.
    
    Returns:
        str: Selected page name ('home', 'upload', or 'chat')
    """
    st.sidebar.title("IN'SMART")
    st.sidebar.markdown("---")
    
    # Determine current index
    current_page = st.session_state.get("selected_page", "home")
    page_options = ["🏠 Accueil", "📤 Téléversement", "💬 Chat"]
    page_map = {"home": 0, "upload": 1, "chat": 2}
    current_index = page_map.get(current_page, 0)
    
    page = st.sidebar.radio(
        "Navigation",
        page_options,
        index=current_index
    )
    
    # Map emoji prefix to page name
    if page.startswith("🏠"):
        selected = "home"
    elif page.startswith("📤"):
        selected = "upload"
    elif page.startswith("💬"):
        selected = "chat"
    else:
        selected = "home"
    
    st.session_state.selected_page = selected
    return selected


def render_home_page() -> None:
    """Render home/welcome page.
    
    Should display:
    - Project overview
    - Quick start instructions
    - System status
    """
    st.title("📚 Bienvenue sur IN'SMART")
    st.markdown("### Votre Assistant Éducatif Intelligent")
    
    st.markdown("""
    IN'SMART est un système de Génération Augmentée par Récupération (RAG) conçu pour vous aider 
    à interagir avec vos supports de cours grâce à une interface de chatbot intelligente.
    """)
    
    st.markdown("---")
    
    # Quick Start
    st.subheader("🚀 Démarrage Rapide")
    st.markdown("""
    1. **Téléverser des Documents** : Allez sur la page Téléversement et téléversez vos supports de cours
       - PDF pour les notes de cours et manuels
       - Fichiers audio pour les cours enregistrés
       - Images pour les notes manuscrites
    
    2. **Ingérer les Documents** : Cliquez sur le bouton "Ingérer" pour traiter et indexer vos documents
    
    3. **Commencer à Discuter** : Allez sur la page Chat et posez des questions sur vos supports
    """)


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
