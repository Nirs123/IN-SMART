"""Chat page for chatbot interface."""

import streamlit as st
import httpx
from typing import List, Dict, Any, Optional
import os


def get_api_url() -> str:
    """Get API URL from environment or session state.
    
    Returns:
        str: API base URL
    """
    if "api_url" in st.session_state:
        return st.session_state.api_url
    return os.getenv("API_URL", "http://localhost:8000")


def send_chat_message(
    message: str,
    conversation_id: Optional[str],
    api_url: str
) -> Optional[Dict[str, Any]]:
    """Send chat message to API.
    
    Args:
        message: User message text
        conversation_id: Optional conversation ID for context
        api_url: API base URL
    
    Returns:
        Optional[Dict[str, Any]]: Chat response with:
            - response: AI response text
            - sources: List of source document IDs
            - conversation_id: Conversation identifier
    
    Raises:
        httpx.HTTPError: If chat request fails
    """
    try:
        payload = {"message": message}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{api_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        st.error(f"Échec de l'envoi du message: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue: {str(e)}")
        return None


def get_chat_history(
    conversation_id: str,
    api_url: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Get chat history for conversation.
    
    Args:
        conversation_id: Conversation identifier
        api_url: API base URL
        limit: Maximum number of messages to retrieve
    
    Returns:
        List[Dict[str, Any]]: List of messages in conversation
    
    Raises:
        httpx.HTTPError: If history request fails
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                f"{api_url}/api/chat/history",
                params={"conversation_id": conversation_id, "limit": limit}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("messages", [])
    except httpx.HTTPError:
        return []
    except Exception:
        return []


def render_chat_interface() -> None:
    """Render main chat interface.
    
    Should:
    - Display chat messages
    - Show user input field
    - Handle message sending
    - Display source citations
    - Show typing indicators
    """
    api_url = get_api_url()
    conversation_id = st.session_state.get("current_conversation_id")
    
    # Initialize conversation messages in session state
    if conversation_id:
        if conversation_id not in st.session_state.conversations:
            st.session_state.conversations[conversation_id] = []
        messages = st.session_state.conversations[conversation_id]
    else:
        messages = []
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        if not messages:
            st.info("👋 Commencez une conversation en tapant un message ci-dessous!")
        else:
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                sources = msg.get("sources", [])
                
                render_message_bubble(content, role == "user", sources if role == "assistant" else None)
    
    # Chat input
    user_input = st.chat_input("Posez une question sur vos documents...")
    
    if user_input:
        # Add user message to session state
        if conversation_id:
            if conversation_id not in st.session_state.conversations:
                st.session_state.conversations[conversation_id] = []
            st.session_state.conversations[conversation_id].append({
                "role": "user",
                "content": user_input,
                "timestamp": None
            })
        
        # Display user message immediately
        render_message_bubble(user_input, is_user=True)
        
        # Show typing indicator and get response
        with st.spinner("🤔 Réflexion en cours..."):
            response = send_chat_message(user_input, conversation_id, api_url)
        
        if response:
            # Update conversation ID if new
            new_conversation_id = response.get("conversation_id")
            if new_conversation_id and new_conversation_id != conversation_id:
                st.session_state.current_conversation_id = new_conversation_id
                if new_conversation_id not in st.session_state.conversations:
                    st.session_state.conversations[new_conversation_id] = []
                # Move user message to new conversation
                if conversation_id and conversation_id in st.session_state.conversations:
                    last_msg = st.session_state.conversations[conversation_id].pop()
                    st.session_state.conversations[new_conversation_id].append(last_msg)
                conversation_id = new_conversation_id
            
            # Add assistant response
            assistant_response = response.get("response", "")
            sources = response.get("sources", [])
            
            st.session_state.conversations[conversation_id].append({
                "role": "assistant",
                "content": assistant_response,
                "sources": sources,
                "timestamp": None
            })
            
            # Display assistant message
            render_message_bubble(assistant_response, is_user=False, sources=sources)
            
            # Rerun to update UI
            st.rerun()
        else:
            st.error("Échec de la récupération de la réponse. Veuillez réessayer.")


def render_message_bubble(
    message: str,
    is_user: bool,
    sources: Optional[List[str]] = None
) -> None:
    """Render a single chat message bubble.
    
    Args:
        message: Message text
        is_user: True if user message, False if AI response
        sources: Optional list of source document IDs
    """
    role = "user" if is_user else "assistant"
    
    with st.chat_message(role):
        st.markdown(message)
        
        # Show sources for assistant messages
        if not is_user and sources:
            render_source_citations(sources)


def render_source_citations(sources: List[str]) -> None:
    """Render source document citations.
    
    Args:
        sources: List of source document IDs
    """
    if not sources:
        return
    
    api_url = get_api_url()
    
    # Try to get document names
    document_names = {}
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{api_url}/api/documents")
            if response.status_code == 200:
                documents = response.json()
                for doc in documents:
                    doc_id = doc.get("document_id")
                    if doc_id in sources:
                        document_names[doc_id] = doc.get("filename", doc_id)
    except Exception:
        pass  # If we can't fetch, just use IDs
    
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for source_id in sources:
            display_name = document_names.get(source_id, source_id)
            st.markdown(f"• {display_name}")


def render_conversation_sidebar() -> None:
    """Render sidebar with conversation management.
    
    Should:
    - List conversations
    - Allow creating new conversation
    - Allow switching conversations
    - Show conversation metadata
    """
    st.subheader("💬 Conversations")
    
    # New conversation button
    if st.button("➕ Nouvelle Conversation", type="primary", use_container_width=True):
        st.session_state.current_conversation_id = None
        st.rerun()
    
    st.divider()
    
    # List conversations
    conversations = st.session_state.get("conversations", {})
    current_id = st.session_state.get("current_conversation_id")
    
    if not conversations:
        st.info("Aucune conversation pour le moment. Commencez à discuter pour en créer une!")
        return
    
    # Display conversation list
    for conv_id, messages in conversations.items():
        # Get conversation metadata
        message_count = len(messages)
        last_message = messages[-1] if messages else None
        last_time = last_message.get("timestamp", "") if last_message else ""
        
        # Truncate ID for display
        display_id = conv_id[:8] + "..." if len(conv_id) > 8 else conv_id
        
        # Check if this is the current conversation
        is_current = conv_id == current_id
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button(
                f"💬 {display_id}",
                key=f"conv_{conv_id}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                st.session_state.current_conversation_id = conv_id
                st.rerun()
            
            # Show metadata
            st.caption(f"{message_count} messages")
            if last_time:
                st.caption(f"Dernier: {last_time[:10] if len(last_time) > 10 else last_time}")
        
        with col2:
            if st.button("🗑️", key=f"delete_conv_{conv_id}", help="Supprimer la conversation"):
                if conv_id in st.session_state.conversations:
                    del st.session_state.conversations[conv_id]
                if st.session_state.get("current_conversation_id") == conv_id:
                    st.session_state.current_conversation_id = None
                st.rerun()
        
        st.divider()


def render_chat_page() -> None:
    """Render complete chat page.
    
    Main page component that includes:
    - Chat interface
    - Conversation sidebar
    - Message history
    - Source citations
    """
    st.title("💬 Chat avec Vos Documents")
    st.markdown("Posez des questions sur vos supports de cours téléversés")
    
    # Initialize conversation if none exists
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    
    # Layout: sidebar for conversations, main area for chat
    col1, col2 = st.columns([1, 3])
    
    with col1:
        render_conversation_sidebar()
    
    with col2:
        render_chat_interface()
