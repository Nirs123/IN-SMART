"""FastAPI main application with route declarations."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import uuid
import mimetypes
from datetime import datetime
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

from api.storage.vector_store import VectorStore
from api.storage.file_store import FileStore
from api.services.ingestion import IngestionService
from api.services.retrieval import RetrievalService
from api.services.llm import LLMService
from api.services.chunking import ChunkingService
from api.services.embeddings import EmbeddingService
from api.processing.audio import AudioProcessor
from api.processing.text import TextProcessor
from minio.error import S3Error

load_dotenv()

app = FastAPI(
    title="IN'SMART API",
    description="RAG-based educational assistant API",
    version="0.1.0",
)

# Initialize services (to be implemented)
vector_store: Optional[VectorStore] = None
file_store: Optional[FileStore] = None
ingestion_service: Optional[IngestionService] = None
retrieval_service: Optional[RetrievalService] = None
llm_service: Optional[LLMService] = None

# Conversation history storage (in-memory)
conversations: Dict[str, List[Dict[str, Any]]] = {}


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize services on application startup.
    
    This method should:
    - Initialize VectorStore connection to Weaviate
    - Initialize FileStore connection to MinIO
    - Create IngestionService instance
    - Create RetrievalService instance
    - Create LLMService instance
    """
    global vector_store, file_store, ingestion_service, retrieval_service, llm_service
    
    try:
        # Get environment variables
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        
        # Initialize VectorStore
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        weaviate_class_name = os.getenv("WEAVIATE_CLASS_NAME", "DocumentChunk")
        vector_store = VectorStore(url=weaviate_url, class_name=weaviate_class_name)
        vector_store.connect()
        vector_store.create_schema()
        
        # Initialize FileStore
        minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "in-smart-documents")
        minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        
        file_store = FileStore(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            bucket_name=minio_bucket_name,
            secure=minio_secure
        )
        file_store.connect()
        
        # Initialize processors
        audio_processor = AudioProcessor(api_key=mistral_api_key)
        audio_processor.initialize()
        
        text_processor = TextProcessor(api_key=mistral_api_key)
        text_processor.initialize()
        
        # Initialize services
        chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        chunking_service = ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        embedding_model = os.getenv("EMBEDDING_MODEL", "mistral-embed")
        embedding_service = EmbeddingService(
            api_key=mistral_api_key,
            model=embedding_model
        )
        
        mistral_model = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")
        llm_service = LLMService(
            api_key=mistral_api_key,
            model=mistral_model
        )
        llm_service.initialize()
        
        # Initialize IngestionService
        ingestion_service = IngestionService(
            vector_store=vector_store,
            file_store=file_store,
            audio_processor=audio_processor,
            text_processor=text_processor,
            chunking_service=chunking_service,
            embedding_service=embedding_service
        )
        
        # Initialize RetrievalService
        top_k = int(os.getenv("TOP_K_RESULTS", "5"))
        similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        retrieval_service = RetrievalService(
            vector_store=vector_store,
            embedding_service=embedding_service,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
    except Exception as e:
        print(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources on application shutdown.
    
    This method should:
    - Close database connections
    - Clean up any open resources
    """
    global vector_store
    if vector_store:
        try:
            vector_store.disconnect()
        except Exception as e:
            print(f"Error during shutdown: {e}")


# Pydantic models for request/response
class ChatMessage(BaseModel):
    """Chat message request model."""
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    sources: List[str]
    conversation_id: str


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    document_id: str
    filename: str
    file_type: str
    upload_date: str
    status: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    services: dict


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and service connectivity.
    
    Returns:
        HealthResponse: Status of API and all connected services
            (Weaviate, MinIO, Mistral API)
    """
    services = {}
    all_healthy = True
    
    # Check Weaviate
    try:
        if vector_store and vector_store.client:
            weaviate_ready = vector_store.client.is_ready()
            services["weaviate"] = "healthy" if weaviate_ready else "unhealthy"
            if not weaviate_ready:
                all_healthy = False
        else:
            services["weaviate"] = "not_initialized"
            all_healthy = False
    except Exception as e:
        services["weaviate"] = f"error: {str(e)}"
        all_healthy = False
    
    # Check MinIO
    try:
        if file_store and file_store.client:
            # Try to list buckets as a health check
            list(file_store.client.list_buckets())
            services["minio"] = "healthy"
        else:
            services["minio"] = "not_initialized"
            all_healthy = False
    except Exception as e:
        services["minio"] = f"error: {str(e)}"
        all_healthy = False
    
    # Check Mistral API (verify key is set)
    try:
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if mistral_api_key:
            services["mistral"] = "configured"
        else:
            services["mistral"] = "not_configured"
            all_healthy = False
    except Exception as e:
        services["mistral"] = f"error: {str(e)}"
        all_healthy = False
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        services=services
    )


def _determine_file_type(filename: str, content_type: Optional[str], document_type: Optional[str]) -> str:
    """Determine file type from filename, content type, or explicit parameter."""
    if document_type:
        return document_type.lower()
    
    # Check file extension
    ext = Path(filename).suffix.lstrip(".").lower()
    
    # PDF
    if ext == "pdf" or content_type == "application/pdf":
        return "pdf"
    
    # Audio formats
    audio_extensions = ["mp3", "wav", "flac", "ogg", "m4a", "webm"]
    audio_types = ["audio/mpeg", "audio/wav", "audio/flac", "audio/ogg", "audio/m4a", "audio/webm"]
    if ext in audio_extensions or (content_type and any(audio_type in content_type for audio_type in audio_types)):
        return "audio"
    
    # Image formats
    image_extensions = ["png", "jpg", "jpeg", "gif", "webp"]
    image_types = ["image/png", "image/jpeg", "image/gif", "image/webp"]
    if ext in image_extensions or (content_type and any(image_type in content_type for image_type in image_types)):
        return "image"
    
    raise ValueError(f"Unsupported file type. Extension: {ext}, Content-Type: {content_type}")


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
) -> JSONResponse:
    """Upload a document for processing.
    
    Args:
        file: The file to upload (PDF, audio, or image)
    
    Returns:
        JSONResponse: Document ID and upload status
    
    Raises:
        HTTPException: If upload fails or file type is unsupported
    """
    if not file_store:
        raise HTTPException(status_code=503, detail="File store not initialized")
    
    try:
        # Determine file type automatically
        try:
            file_type = _determine_file_type(file.filename or "unknown", file.content_type, None)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Generate unique document ID (preserve extension for processing)
        document_id = str(uuid.uuid4())
        if file.filename:
            # Preserve extension in document ID
            ext = Path(file.filename).suffix
            document_id = f"{document_id}{ext}"
        
        # Store original filename separately
        original_filename = file.filename or "unknown"
        
        # Read file content
        file_content = await file.read()
        file_stream = BytesIO(file_content)
        
        # Prepare metadata with proper format for MinIO
        upload_date = datetime.now().isoformat()
        # MinIO metadata keys should be prefixed with X-Amz-Meta- or use user_metadata
        # For better compatibility, we'll store as user metadata
        metadata = {
            "X-Amz-Meta-filename": original_filename,
            "X-Amz-Meta-file-type": file_type,
            "X-Amz-Meta-upload-date": upload_date,
            "X-Amz-Meta-status": "uploaded"
        }
        
        # Determine content type
        content_type = file.content_type or mimetypes.guess_type(original_filename)[0] or "application/octet-stream"
        
        # Upload to MinIO
        file_store.upload_file(
            file_data=file_stream,
            object_name=document_id,
            content_type=content_type,
            metadata=metadata
        )
        
        return JSONResponse(
            status_code=201,
            content={
                "document_id": document_id,
                "filename": original_filename,
                "file_type": file_type,
                "status": "uploaded",
                "upload_date": upload_date
            }
        )
    
    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/api/documents", response_model=List[DocumentMetadata])
async def list_documents() -> List[DocumentMetadata]:
    """List all uploaded documents.
    
    Returns:
        List[DocumentMetadata]: List of all documents with metadata
    """
    if not file_store:
        raise HTTPException(status_code=503, detail="File store not initialized")
    
    try:
        # List all files (returns iterator from MinIO)
        files = file_store.list_files()
        
        documents = []
        for file_obj in files:
            try:
                # Get object name - handle both dict and object attributes
                object_name = file_obj.object_name if hasattr(file_obj, 'object_name') else file_obj.get('object_name', '')
                if not object_name:
                    continue
                
                # Get metadata for each file
                metadata = file_store.get_file_metadata(object_name)
                
                # Extract metadata fields - handle both X-Amz-Meta- prefixed and direct keys
                filename = (
                    metadata.get("X-Amz-Meta-filename") or 
                    metadata.get("filename") or 
                    object_name
                )
                file_type = (
                    metadata.get("X-Amz-Meta-file-type") or 
                    metadata.get("file_type") or 
                    "unknown"
                )
                upload_date = (
                    metadata.get("X-Amz-Meta-upload-date") or 
                    metadata.get("upload_date") or 
                    metadata.get("last_modified") or 
                    ""
                )
                status = (
                    metadata.get("X-Amz-Meta-status") or 
                    metadata.get("status") or 
                    "uploaded"
                )
                
                documents.append(DocumentMetadata(
                    document_id=object_name,
                    filename=filename,
                    file_type=file_type,
                    upload_date=upload_date,
                    status=status
                ))
            except Exception as e:
                # Skip files that can't be processed
                object_name = file_obj.object_name if hasattr(file_obj, 'object_name') else 'unknown'
                print(f"Error processing file {object_name}: {e}")
                continue
        
        return documents
    
    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/api/documents/{document_id}", response_model=DocumentMetadata)
async def get_document(document_id: str) -> DocumentMetadata:
    """Get document metadata by ID.
    
    Args:
        document_id: Unique document identifier
    
    Returns:
        DocumentMetadata: Document information
    
    Raises:
        HTTPException: If document not found
    """
    if not file_store:
        raise HTTPException(status_code=503, detail="File store not initialized")
    
    try:
        # Check if file exists
        if not file_store.file_exists(document_id):
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
        
        # Get metadata
        metadata = file_store.get_file_metadata(document_id)
        
        # Extract metadata fields - handle both X-Amz-Meta- prefixed and direct keys
        filename = (
            metadata.get("X-Amz-Meta-filename") or 
            metadata.get("filename") or 
            document_id
        )
        file_type = (
            metadata.get("X-Amz-Meta-file-type") or 
            metadata.get("file_type") or 
            "unknown"
        )
        upload_date = (
            metadata.get("X-Amz-Meta-upload-date") or 
            metadata.get("upload_date") or 
            metadata.get("last_modified") or 
            ""
        )
        status = (
            metadata.get("X-Amz-Meta-status") or 
            metadata.get("status") or 
            "uploaded"
        )
        
        return DocumentMetadata(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            upload_date=upload_date,
            status=status
        )
    
    except HTTPException:
        raise
    except S3Error as e:
        if "not found" in str(e).lower() or "404" in str(e):
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str) -> JSONResponse:
    """Delete a document and all associated chunks.
    
    Args:
        document_id: Unique document identifier
    
    Returns:
        JSONResponse: Deletion status
    
    Raises:
        HTTPException: If document not found or deletion fails
    """
    if not file_store or not ingestion_service:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Check if file exists
        if not file_store.file_exists(document_id):
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
        
        # Delete chunks from Weaviate
        try:
            chunks_deleted = ingestion_service.delete_document_chunks(document_id)
        except Exception as e:
            print(f"Warning: Failed to delete chunks for document '{document_id}': {e}")
            chunks_deleted = 0
        
        # Delete file from MinIO
        file_store.delete_file(document_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "document_id": document_id,
                "status": "deleted",
                "chunks_deleted": chunks_deleted
            }
        )
    
    except HTTPException:
        raise
    except S3Error as e:
        if "not found" in str(e).lower() or "404" in str(e):
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/api/ingest/{document_id}")
async def ingest_document(document_id: str) -> JSONResponse:
    """Process and ingest a document into the vector database.
    
    This endpoint:
    1. Retrieves the document from file storage
    2. Processes it (STT/OCR/text extraction)
    3. Chunks the text
    4. Generates embeddings
    5. Stores chunks in vector database
    
    Args:
        document_id: Unique document identifier
    
    Returns:
        JSONResponse: Ingestion status and chunk count
    
    Raises:
        HTTPException: If document not found or ingestion fails
    """
    if not file_store or not ingestion_service:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Check if document exists
        if not file_store.file_exists(document_id):
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
        
        # Get file type from metadata
        try:
            metadata = file_store.get_file_metadata(document_id)
            file_type = (
                metadata.get("X-Amz-Meta-file-type") or 
                metadata.get("file_type") or 
                "unknown"
            )
            if file_type == "unknown":
                # Try to determine from filename
                ext = Path(document_id).suffix.lstrip(".").lower()
                if ext == "pdf":
                    file_type = "pdf"
                elif ext in ["mp3", "wav", "flac", "ogg", "m4a", "webm"]:
                    file_type = "audio"
                elif ext in ["png", "jpg", "jpeg", "gif", "webp"]:
                    file_type = "image"
                else:
                    raise HTTPException(status_code=400, detail=f"Cannot determine file type for document '{document_id}'")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=400, detail=f"Failed to get document metadata: {str(e)}")
        
        # Ingest document
        result = ingestion_service.ingest_document(
            document_id=document_id,
            file_type=file_type,
            metadata=metadata
        )
        
        # Update document status in MinIO metadata
        try:
            # Note: MinIO doesn't support updating metadata directly
            # We'll need to re-upload with updated metadata or store status elsewhere
            # For now, we'll just return the result
            pass
        except Exception as e:
            print(f"Warning: Failed to update document status: {e}")
        
        return JSONResponse(
            status_code=200,
            content={
                "document_id": document_id,
                "status": result.get("status", "success"),
                "chunks_created": result.get("chunks_created", 0),
                "processing_time": result.get("processing_time", 0)
            }
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage) -> ChatResponse:
    """Send a message and get AI response with RAG.
    
    This endpoint:
    1. Retrieves relevant document chunks using semantic search
    2. Assembles context from retrieved chunks
    3. Generates response using LLM with context
    4. Returns response with source citations
    
    Args:
        message: Chat message request with optional conversation_id
    
    Returns:
        ChatResponse: AI response with source document references
    
    Raises:
        HTTPException: If retrieval or generation fails
    """
    if not retrieval_service or not llm_service:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Generate or use provided conversation_id
        conversation_id = message.conversation_id or str(uuid.uuid4())
        
        # Retrieve relevant chunks
        chunks = retrieval_service.retrieve(message.message)
        
        # Format context from retrieved chunks
        context = retrieval_service.format_context(chunks)
        
        # Build system prompt
        system_prompt = (
            "You are a helpful educational assistant. Try to use the provided context from course materials "
            "to answer the user's question. If the context doesn't contain relevant information, "
            "say so. Never cite which document sources you used."
        )
        
        # Build user message with context
        user_message_with_context = (
            f"Context from course materials:\n{context}\n\n"
            f"Question: {message.message}"
        )
        
        # Generate response using LLM
        llm_response = llm_service.generate_response(
            user_message=user_message_with_context,
            system_prompt=system_prompt
        )
        
        # Extract response text
        # The LLM response structure depends on Mistral API version
        if hasattr(llm_response, 'choices') and len(llm_response.choices) > 0:
            response_text = llm_response.choices[0].message.content
        elif isinstance(llm_response, dict):
            response_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            response_text = str(llm_response)
        
        # Extract unique source document IDs
        source_doc_ids = list(set(chunk.get("document_id", "") for chunk in chunks if chunk.get("document_id")))
        
        # Store messages in conversation history
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        conversations[conversation_id].append({
            "role": "user",
            "content": message.message,
            "timestamp": datetime.now().isoformat()
        })
        
        conversations[conversation_id].append({
            "role": "assistant",
            "content": response_text,
            "sources": source_doc_ids,
            "timestamp": datetime.now().isoformat()
        })
        
        return ChatResponse(
            response=response_text,
            sources=source_doc_ids,
            conversation_id=conversation_id
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/api/chat/history")
async def get_chat_history(
    conversation_id: str,
    limit: int = 50
) -> JSONResponse:
    """Get chat history for a conversation.
    
    Args:
        conversation_id: Unique conversation identifier
        limit: Maximum number of messages to return
    
    Returns:
        JSONResponse: List of messages in the conversation
    
    Raises:
        HTTPException: If conversation not found
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found")
    
    # Get conversation messages
    messages = conversations[conversation_id]
    
    # Apply limit (return most recent messages)
    if limit > 0:
        messages = messages[-limit:]
    
    return JSONResponse(
        status_code=200,
        content={
            "conversation_id": conversation_id,
            "messages": messages,
            "total_messages": len(conversations[conversation_id])
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True
    )
