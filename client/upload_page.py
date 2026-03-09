"""Upload page for document upload interface."""

import streamlit as st
import httpx
from typing import Optional
import os


def get_api_url() -> str:
    """Get API URL from environment or session state.
    
    Returns:
        str: API base URL
    """
    if "api_url" in st.session_state:
        return st.session_state.api_url
    return os.getenv("API_URL", "http://localhost:8000")


def upload_file_to_api(
    file_data: bytes,
    filename: str,
    file_type: str,
    api_url: str
) -> Optional[dict]:
    """Upload file to API.
    
    Args:
        file_data: File content as bytes
        filename: Original filename
        file_type: File type ('pdf', 'audio', 'image') - auto-detected, not used in API call
        api_url: API base URL
    
    Returns:
        Optional[dict]: Upload response with document_id, or None if failed
    
    Raises:
        httpx.HTTPError: If upload request fails
    """
    try:
        files = {"file": (filename, file_data)}
        # document_type parameter removed - file type is auto-detected by API
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{api_url}/api/documents/upload",
                files=files
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        st.error(f"Échec du téléversement du fichier: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du téléversement: {str(e)}")
        return None


def render_file_uploader() -> None:
    """Render file uploader component.
    
    Should:
    - Display file uploader
    - Support PDF, audio, and image files
    - Show upload progress
    - Handle file validation
    """
    api_url = get_api_url()
    
    uploaded_file = st.file_uploader(
        "Choisir un fichier",
        type=["pdf", "mp3", "wav", "flac", "ogg", "m4a", "webm", "png", "jpg", "jpeg", "gif", "webp"],
        help="Téléversez des PDF, fichiers audio ou images de notes manuscrites"
    )
    
    if uploaded_file is not None:
        # Determine file type from extension
        filename = uploaded_file.name
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        
        file_type = None
        if ext == "pdf":
            file_type = "pdf"
        elif ext in ["mp3", "wav", "flac", "ogg", "m4a", "webm"]:
            file_type = "audio"
        elif ext in ["png", "jpg", "jpeg", "gif", "webp"]:
            file_type = "image"
        else:
            st.warning(f"Type de fichier inconnu pour l'extension: {ext}")
            return
        
        # Display file info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📄 **Fichier:** {filename}")
        with col2:
            st.info(f"📊 **Taille:** {len(uploaded_file.getvalue()) / 1024:.2f} KB")
        
        # Upload button
        if st.button("📤 Téléverser le Fichier", type="primary"):
            with st.spinner(f"Téléversement de {filename}..."):
                file_data = uploaded_file.getvalue()
                result = upload_file_to_api(file_data, filename, file_type, api_url)
                
                if result:
                    st.success("✅ Fichier téléversé avec succès!")
                    st.json(result)
                    # Clear the uploaded file from session
                    st.session_state.uploaded_file = None
                    # Refresh document list
                    if "documents" in st.session_state:
                        st.session_state.documents = []
                    st.rerun()
                else:
                    st.error("❌ Échec du téléversement. Veuillez réessayer.")


def render_document_list() -> None:
    """Render list of uploaded documents.
    
    Should:
    - Fetch documents from API
    - Display document metadata
    - Show document status (uploaded, processing, ingested)
    - Allow document deletion
    """
    api_url = get_api_url()
    
    # Fetch documents
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{api_url}/api/documents")
            response.raise_for_status()
            documents = response.json()
            st.session_state.documents = documents
    except httpx.HTTPError as e:
        st.error(f"Échec de la récupération des documents: {str(e)}")
        documents = st.session_state.get("documents", [])
    except Exception as e:
        st.error(f"Erreur inattendue: {str(e)}")
        documents = st.session_state.get("documents", [])
    
    if not documents:
        st.info("📭 Aucun document téléversé pour le moment. Utilisez le téléverseur ci-dessus pour ajouter des documents.")
        return
    
    # Display documents in a table
    for doc in documents:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            
            with col1:
                # File icon based on type
                file_type = doc.get("file_type", "unknown")
                if file_type == "pdf":
                    icon = "📄"
                elif file_type == "audio":
                    icon = "🎵"
                elif file_type == "image":
                    icon = "🖼️"
                else:
                    icon = "📎"
                
                st.markdown(f"{icon} **{doc.get('filename', doc.get('document_id', 'Unknown'))}**")
            
            with col2:
                file_type = doc.get("file_type", "unknown")
                st.markdown(f"**Type:** {file_type}")
            
            with col3:
                upload_date = doc.get("upload_date", "Inconnue")
                st.markdown(f"**Date:** {upload_date[:10] if len(upload_date) > 10 else upload_date}")
            
            with col4:
                status = doc.get("status", "uploaded")
                if status == "ingested":
                    st.success("✅ Ingéré")
                else:
                    st.info("📤 Téléversé")
            
            with col5:
                document_id = doc.get("document_id")
                delete_key = f"delete_{document_id}"
                
                # Check if this is a confirmation click
                if st.session_state.get(f"confirm_{delete_key}", False):
                    if st.button("✅ Confirmer", key=f"confirm_btn_{document_id}", type="primary"):
                        try:
                            with httpx.Client(timeout=10.0) as client:
                                delete_response = client.delete(f"{api_url}/api/documents/{document_id}")
                                delete_response.raise_for_status()
                                st.success(f"✅ Supprimé {doc.get('filename', document_id)}")
                                # Clear confirmation state
                                if f"confirm_{delete_key}" in st.session_state:
                                    del st.session_state[f"confirm_{delete_key}"]
                                # Refresh list
                                st.session_state.documents = []
                                st.rerun()
                        except Exception as e:
                            st.error(f"Échec de la suppression: {str(e)}")
                    if st.button("❌ Annuler", key=f"cancel_{document_id}"):
                        if f"confirm_{delete_key}" in st.session_state:
                            del st.session_state[f"confirm_{delete_key}"]
                        st.rerun()
                else:
                    if st.button("🗑️", key=delete_key, help="Supprimer le document"):
                        st.session_state[f"confirm_{delete_key}"] = True
                        st.rerun()
            
            st.divider()


def render_ingestion_status() -> None:
    """Render ingestion status for documents.
    
    Should:
    - Show ingestion status per document
    - Allow manual ingestion trigger
    - Display chunk count
    - Update local status after successful ingestion
    """
    api_url = get_api_url()
    documents = st.session_state.get("documents", [])
    
    if not documents:
        st.info("Aucun document à ingérer. Téléversez d'abord des documents.")
        return
    
    # Group documents by status
    ingested_docs = [d for d in documents if d.get("status") == "ingested"]
    uploaded_docs = [d for d in documents if d.get("status") == "uploaded"]
    
    if uploaded_docs:
        st.subheader("📤 Documents Prêts pour l'Ingestion")
        for doc in uploaded_docs:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{doc.get('filename', doc.get('document_id'))}**")
                with col2:
                    document_id = doc.get("document_id")
                    if st.button("🔄 Ingérer", key=f"ingest_{document_id}", type="primary"):
                        with st.spinner(f"Ingestion de {doc.get('filename', document_id)}..."):
                            try:
                                with httpx.Client(timeout=300.0) as client:  # 5 min timeout for ingestion
                                    response = client.post(f"{api_url}/api/ingest/{document_id}")
                                    response.raise_for_status()
                                    result = response.json()
                                    
                                    chunks_created = result.get("chunks_created", 0)
                                    processing_time = result.get("processing_time", 0)
                                    
                                    st.success(
                                        f"✅ Ingestion réussie! "
                                        f"Créé {chunks_created} chunks en {processing_time:.2f}s"
                                    )
                                    
                                    # Update local document status to "ingested"
                                    for i, d in enumerate(documents):
                                        if d.get("document_id") == document_id:
                                            documents[i]["status"] = "ingested"
                                            break
                                    
                                    # Update session state
                                    st.session_state.documents = documents
                                    st.rerun()
                            except httpx.HTTPError as e:
                                st.error(f"Échec de l'ingestion du document: {str(e)}")
                            except Exception as e:
                                st.error(f"Erreur inattendue: {str(e)}")
                st.divider()
    
    if ingested_docs:
        st.subheader("✅ Documents Ingérés")
        for doc in ingested_docs:
            st.markdown(f"✅ **{doc.get('filename', doc.get('document_id'))}** - Prêt pour le chat")


def render_upload_page() -> None:
    """Render complete upload page.
    
    Main page component that includes:
    - File uploader
    - Document list
    - Ingestion status
    - Upload history
    """
    st.title("📤 Téléversement de Documents")
    st.markdown("Téléversez vos supports de cours (PDF, enregistrements audio ou notes manuscrites)")
    
    # File upload section
    with st.container():
        st.subheader("Téléverser un Nouveau Document")
        render_file_uploader()
    
    # Document list section
    with st.container():
        st.subheader("Documents Téléversés")
        render_document_list()
    
    # Ingestion status section
    with st.container():
        st.subheader("Statut d'Ingestion")
        render_ingestion_status()
