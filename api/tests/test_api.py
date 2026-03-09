"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from io import BytesIO
import uuid

from api.main import app,  conversations


# Fixtures
@pytest.fixture
def mock_vector_store():
    """Create a mocked VectorStore."""
    mock = MagicMock()
    mock.client = MagicMock()
    mock.client.is_ready.return_value = True
    return mock


@pytest.fixture
def mock_file_store():
    """Create a mocked FileStore."""
    mock = MagicMock()
    mock.client = MagicMock()
    mock.client.list_buckets.return_value = []
    return mock


@pytest.fixture
def mock_ingestion_service():
    """Create a mocked IngestionService."""
    mock = MagicMock()
    mock.ingest_document.return_value = {
        "document_id": "test-doc-123",
        "chunks_created": 5,
        "status": "success",
        "processing_time": 1.5
    }
    mock.delete_document_chunks.return_value = 5
    return mock


@pytest.fixture
def mock_retrieval_service():
    """Create a mocked RetrievalService."""
    mock = MagicMock()
    mock.retrieve.return_value = [
        {
            "text": "Test chunk content",
            "document_id": "doc-1",
            "chunk_index": 0,
            "similarity": 0.85,
            "metadata": {},
            "uuid": "uuid-1"
        }
    ]
    mock.format_context.return_value = "[Source 1 - doc-1]\nTest chunk content"
    return mock


@pytest.fixture
def mock_llm_service():
    """Create a mocked LLMService."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "This is a test response"
    mock.generate_response.return_value = mock_response
    return mock


@pytest.fixture
def sample_pdf_file():
    """Create a sample PDF file in memory."""
    # Minimal PDF content (just enough to be recognized as PDF)
    pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 0\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"
    return ("test.pdf", pdf_content, "application/pdf")


@pytest.fixture
def sample_audio_file():
    """Create a sample audio file in memory."""
    # Minimal MP3 header
    mp3_content = b"ID3\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    return ("test.mp3", mp3_content, "audio/mpeg")


@pytest.fixture
def sample_image_file():
    """Create a sample image file in memory."""
    # Minimal PNG (1x1 white pixel)
    png_content = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
    )
    return ("test.png", png_content, "image/png")


@pytest.fixture
def sample_document_metadata():
    """Sample document metadata."""
    return {
        "document_id": "test-doc-123.pdf",
        "filename": "test.pdf",
        "file_type": "pdf",
        "upload_date": "2024-01-01T00:00:00",
        "status": "uploaded"
    }


@pytest.fixture
def test_client(mock_vector_store, mock_file_store, mock_ingestion_service, 
                mock_retrieval_service, mock_llm_service):
    """Create a TestClient with mocked services."""
    with patch('api.main.vector_store', mock_vector_store), \
         patch('api.main.file_store', mock_file_store), \
         patch('api.main.ingestion_service', mock_ingestion_service), \
         patch('api.main.retrieval_service', mock_retrieval_service), \
         patch('api.main.llm_service', mock_llm_service):
        # Clear conversations for clean test state
        conversations.clear()
        yield TestClient(app)


class TestHealthEndpoint:
    """Test cases for health check endpoint."""

    def test_health_check_success(self, test_client, mock_vector_store, mock_file_store):
        """Test health check endpoint with all services healthy."""
        mock_vector_store.client.is_ready.return_value = True
        mock_file_store.client.list_buckets.return_value = []
        
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert data["services"]["weaviate"] == "healthy"
        assert data["services"]["minio"] == "healthy"
        assert data["services"]["mistral"] == "configured"

    def test_health_check_degraded(self, test_client, mock_vector_store, mock_file_store):
        """Test health check endpoint with some services failing."""
        mock_vector_store.client.is_ready.return_value = False
        mock_file_store.client.list_buckets.return_value = []
        
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["weaviate"] == "unhealthy"

    def test_health_check_not_initialized(self, test_client):
        """Test health check when services are not initialized."""
        with patch('api.main.vector_store', None), \
             patch('api.main.file_store', None):
            response = test_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["services"]["weaviate"] == "not_initialized"
            assert data["services"]["minio"] == "not_initialized"


class TestDocumentEndpoints:
    """Test cases for document management endpoints."""

    def test_upload_document_pdf(self, test_client, mock_file_store, sample_pdf_file):
        """Test PDF document upload."""
        filename, content, content_type = sample_pdf_file
        mock_file_store.upload_file.return_value = "test-doc-123.pdf"
        
        files = {"file": (filename, BytesIO(content), content_type)}
        response = test_client.post("/api/documents/upload", files=files)
        
        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == filename
        assert data["file_type"] == "pdf"
        assert data["status"] == "uploaded"
        mock_file_store.upload_file.assert_called_once()

    def test_upload_document_audio(self, test_client, mock_file_store, sample_audio_file):
        """Test audio document upload."""
        filename, content, content_type = sample_audio_file
        mock_file_store.upload_file.return_value = "test-doc-123.mp3"
        
        files = {"file": (filename, BytesIO(content), content_type)}
        response = test_client.post("/api/documents/upload", files=files)
        
        assert response.status_code == 201
        data = response.json()
        assert data["file_type"] == "audio"
        assert data["filename"] == filename

    def test_upload_document_image(self, test_client, mock_file_store, sample_image_file):
        """Test image document upload."""
        filename, content, content_type = sample_image_file
        mock_file_store.upload_file.return_value = "test-doc-123.png"
        
        files = {"file": (filename, BytesIO(content), content_type)}
        response = test_client.post("/api/documents/upload", files=files)
        
        assert response.status_code == 201
        data = response.json()
        assert data["file_type"] == "image"
        assert data["filename"] == filename

    def test_upload_document_unsupported_type(self, test_client):
        """Test upload of unsupported file type."""
        files = {"file": ("test.txt", BytesIO(b"text content"), "text/plain")}
        response = test_client.post("/api/documents/upload", files=files)
        
        # The endpoint raises ValueError which gets caught and returns 500
        assert response.status_code in [400, 500]
        assert "Unsupported file type" in response.json()["detail"] or "error" in response.json()["detail"].lower()

    def test_upload_document_file_store_error(self, test_client, mock_file_store, sample_pdf_file):
        """Test upload when MinIO fails."""
        filename, content, content_type = sample_pdf_file
        from minio.error import S3Error
        mock_file_store.upload_file.side_effect = S3Error("Upload failed", 500, "error", "error", "error", "error")
        
        files = {"file": (filename, BytesIO(content), content_type)}
        response = test_client.post("/api/documents/upload", files=files)
        
        assert response.status_code == 500
        assert "Failed to upload file" in response.json()["detail"]

    def test_list_documents(self, test_client, mock_file_store):
        """Test document listing."""
        # Mock file list
        mock_file_obj = MagicMock()
        mock_file_obj.object_name = "test-doc-123.pdf"
        mock_file_store.list_files.return_value = [mock_file_obj]
        mock_file_store.get_file_metadata.return_value = {
            "filename": "test.pdf",
            "file_type": "pdf",
            "upload_date": "2024-01-01T00:00:00",
            "status": "uploaded"
        }
        
        response = test_client.get("/api/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "document_id" in data[0]
            assert "filename" in data[0]
            assert "file_type" in data[0]

    def test_list_documents_empty(self, test_client, mock_file_store):
        """Test listing when no documents exist."""
        mock_file_store.list_files.return_value = []
        
        response = test_client.get("/api/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_get_document(self, test_client, mock_file_store):
        """Test document retrieval."""
        document_id = "test-doc-123.pdf"
        mock_file_store.file_exists.return_value = True
        mock_file_store.get_file_metadata.return_value = {
            "filename": "test.pdf",
            "file_type": "pdf",
            "upload_date": "2024-01-01T00:00:00",
            "status": "uploaded"
        }
        
        response = test_client.get(f"/api/documents/{document_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == document_id
        assert data["filename"] == "test.pdf"
        assert data["file_type"] == "pdf"

    def test_get_document_not_found(self, test_client, mock_file_store):
        """Test retrieval of non-existent document."""
        document_id = "nonexistent-doc.pdf"
        mock_file_store.file_exists.return_value = False
        
        response = test_client.get(f"/api/documents/{document_id}")
        
        assert response.status_code == 404
        assert document_id in response.json()["detail"]

    def test_delete_document(self, test_client, mock_file_store, mock_ingestion_service):
        """Test document deletion."""
        document_id = "test-doc-123.pdf"
        mock_file_store.file_exists.return_value = True
        mock_ingestion_service.delete_document_chunks.return_value = 5
        
        response = test_client.delete(f"/api/documents/{document_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["document_id"] == document_id
        assert data["chunks_deleted"] == 5
        mock_file_store.delete_file.assert_called_once_with(document_id)
        mock_ingestion_service.delete_document_chunks.assert_called_once_with(document_id)

    def test_delete_document_not_found(self, test_client, mock_file_store):
        """Test deletion of non-existent document."""
        document_id = "nonexistent-doc.pdf"
        mock_file_store.file_exists.return_value = False
        
        response = test_client.delete(f"/api/documents/{document_id}")
        
        assert response.status_code == 404
        assert document_id in response.json()["detail"]

    def test_delete_document_chunks_deleted(self, test_client, mock_file_store, mock_ingestion_service):
        """Test that chunks are deleted when document is deleted."""
        document_id = "test-doc-123.pdf"
        mock_file_store.file_exists.return_value = True
        mock_ingestion_service.delete_document_chunks.return_value = 10
        
        response = test_client.delete(f"/api/documents/{document_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_deleted"] == 10
        mock_ingestion_service.delete_document_chunks.assert_called_once_with(document_id)


class TestIngestionEndpoint:
    """Test cases for ingestion endpoint."""

    def test_ingest_document_pdf(self, test_client, mock_file_store, mock_ingestion_service):
        """Test PDF document ingestion."""
        document_id = "test-doc-123.pdf"
        mock_file_store.file_exists.return_value = True
        mock_file_store.get_file_metadata.return_value = {
            "filename": "test.pdf",
            "file_type": "pdf",
            "upload_date": "2024-01-01T00:00:00",
            "status": "uploaded"
        }
        mock_ingestion_service.ingest_document.return_value = {
            "document_id": document_id,
            "chunks_created": 10,
            "status": "success",
            "processing_time": 2.5
        }
        
        response = test_client.post(f"/api/ingest/{document_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == document_id
        assert data["chunks_created"] == 10
        assert data["status"] == "success"
        assert "processing_time" in data
        mock_ingestion_service.ingest_document.assert_called_once()

    def test_ingest_document_audio(self, test_client, mock_file_store, mock_ingestion_service):
        """Test audio document ingestion."""
        document_id = "test-doc-123.mp3"
        mock_file_store.file_exists.return_value = True
        mock_file_store.get_file_metadata.return_value = {
            "filename": "test.mp3",
            "file_type": "audio",
            "upload_date": "2024-01-01T00:00:00",
            "status": "uploaded"
        }
        mock_ingestion_service.ingest_document.return_value = {
            "document_id": document_id,
            "chunks_created": 8,
            "status": "success",
            "processing_time": 3.0
        }
        
        response = test_client.post(f"/api/ingest/{document_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_created"] == 8
        assert data["status"] == "success"

    def test_ingest_document_image(self, test_client, mock_file_store, mock_ingestion_service):
        """Test image document ingestion."""
        document_id = "test-doc-123.png"
        mock_file_store.file_exists.return_value = True
        mock_file_store.get_file_metadata.return_value = {
            "filename": "test.png",
            "file_type": "image",
            "upload_date": "2024-01-01T00:00:00",
            "status": "uploaded"
        }
        mock_ingestion_service.ingest_document.return_value = {
            "document_id": document_id,
            "chunks_created": 3,
            "status": "success",
            "processing_time": 1.2
        }
        
        response = test_client.post(f"/api/ingest/{document_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_created"] == 3

    def test_ingest_nonexistent_document(self, test_client, mock_file_store):
        """Test ingestion of non-existent document."""
        document_id = "nonexistent-doc.pdf"
        mock_file_store.file_exists.return_value = False
        
        response = test_client.post(f"/api/ingest/{document_id}")
        
        assert response.status_code == 404
        assert document_id in response.json()["detail"]

    def test_ingest_document_processing_error(self, test_client, mock_file_store, mock_ingestion_service):
        """Test ingestion when processing fails."""
        document_id = "test-doc-123.pdf"
        mock_file_store.file_exists.return_value = True
        mock_file_store.get_file_metadata.return_value = {
            "filename": "test.pdf",
            "file_type": "pdf",
            "upload_date": "2024-01-01T00:00:00",
            "status": "uploaded"
        }
        mock_ingestion_service.ingest_document.side_effect = ValueError("Processing failed")
        
        response = test_client.post(f"/api/ingest/{document_id}")
        
        assert response.status_code == 400
        assert "Processing failed" in response.json()["detail"]

    def test_ingest_document_unknown_file_type(self, test_client, mock_file_store):
        """Test ingestion with unknown file type."""
        document_id = "test-doc-123.xyz"
        mock_file_store.file_exists.return_value = True
        mock_file_store.get_file_metadata.return_value = {
            "filename": "test.xyz",
            "file_type": "unknown",
            "upload_date": "2024-01-01T00:00:00",
            "status": "uploaded"
        }
        
        response = test_client.post(f"/api/ingest/{document_id}")
        
        # Should try to determine from extension, but .xyz is not supported
        assert response.status_code in [400, 404]


class TestChatEndpoint:
    """Test cases for chat endpoint."""

    def test_chat_basic(self, test_client, mock_retrieval_service, mock_llm_service):
        """Test basic chat without conversation_id."""
        # Setup mocks
        mock_retrieval_service.retrieve.return_value = [
            {
                "text": "Test chunk",
                "document_id": "doc-1",
                "chunk_index": 0,
                "similarity": 0.9,
                "metadata": {},
                "uuid": "uuid-1"
            }
        ]
        mock_retrieval_service.format_context.return_value = "[Source 1 - doc-1]\nTest chunk"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "This is a test response"
        mock_llm_service.generate_response.return_value = mock_response
        
        payload = {"message": "What is this about?"}
        response = test_client.post("/api/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "sources" in data
        assert "conversation_id" in data
        assert data["response"] == "This is a test response"
        assert "doc-1" in data["sources"]
        assert len(data["conversation_id"]) > 0

    def test_chat_with_conversation_id(self, test_client, mock_retrieval_service, mock_llm_service):
        """Test chat with existing conversation_id."""
        conversation_id = str(uuid.uuid4())
        
        mock_retrieval_service.retrieve.return_value = [
            {
                "text": "Test chunk",
                "document_id": "doc-1",
                "chunk_index": 0,
                "similarity": 0.9,
                "metadata": {},
                "uuid": "uuid-1"
            }
        ]
        mock_retrieval_service.format_context.return_value = "[Source 1 - doc-1]\nTest chunk"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Follow-up response"
        mock_llm_service.generate_response.return_value = mock_response
        
        payload = {"message": "Tell me more", "conversation_id": conversation_id}
        response = test_client.post("/api/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == conversation_id
        # Verify conversation was stored
        assert conversation_id in conversations
        assert len(conversations[conversation_id]) == 2  # user + assistant

    def test_chat_no_chunks_found(self, test_client, mock_retrieval_service, mock_llm_service):
        """Test chat when no chunks are found."""
        mock_retrieval_service.retrieve.return_value = []
        mock_retrieval_service.format_context.return_value = ""
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "No relevant information found"
        mock_llm_service.generate_response.return_value = mock_response
        
        payload = {"message": "What is this about?"}
        response = test_client.post("/api/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 0

    def test_chat_retrieval_error(self, test_client, mock_retrieval_service):
        """Test chat when retrieval fails."""
        mock_retrieval_service.retrieve.side_effect = ValueError("Retrieval failed")
        
        payload = {"message": "What is this about?"}
        response = test_client.post("/api/chat", json=payload)
        
        assert response.status_code == 400
        assert "Retrieval failed" in response.json()["detail"]

    def test_chat_llm_error(self, test_client, mock_retrieval_service, mock_llm_service):
        """Test chat when LLM fails."""
        mock_retrieval_service.retrieve.return_value = [
            {
                "text": "Test chunk",
                "document_id": "doc-1",
                "chunk_index": 0,
                "similarity": 0.9,
                "metadata": {},
                "uuid": "uuid-1"
            }
        ]
        mock_retrieval_service.format_context.return_value = "[Source 1 - doc-1]\nTest chunk"
        mock_llm_service.generate_response.side_effect = ValueError("LLM failed")
        
        payload = {"message": "What is this about?"}
        response = test_client.post("/api/chat", json=payload)
        
        # ValueError is caught as 400 in the chat endpoint
        assert response.status_code in [400, 500]
        assert "LLM failed" in response.json()["detail"] or "Chat failed" in response.json()["detail"]

    def test_chat_sources_extraction(self, test_client, mock_retrieval_service, mock_llm_service):
        """Test that sources are correctly extracted from chunks."""
        mock_retrieval_service.retrieve.return_value = [
            {
                "text": "Chunk 1",
                "document_id": "doc-1",
                "chunk_index": 0,
                "similarity": 0.9,
                "metadata": {},
                "uuid": "uuid-1"
            },
            {
                "text": "Chunk 2",
                "document_id": "doc-2",
                "chunk_index": 0,
                "similarity": 0.85,
                "metadata": {},
                "uuid": "uuid-2"
            },
            {
                "text": "Chunk 3",
                "document_id": "doc-1",  # Duplicate document
                "chunk_index": 1,
                "similarity": 0.8,
                "metadata": {},
                "uuid": "uuid-3"
            }
        ]
        mock_retrieval_service.format_context.return_value = "Context"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Response"
        mock_llm_service.generate_response.return_value = mock_response
        
        payload = {"message": "Test"}
        response = test_client.post("/api/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        # Should have unique document IDs
        assert len(data["sources"]) == 2
        assert "doc-1" in data["sources"]
        assert "doc-2" in data["sources"]

    def test_get_chat_history(self, test_client):
        """Test chat history retrieval."""
        conversation_id = str(uuid.uuid4())
        conversations[conversation_id] = [
            {
                "role": "user",
                "content": "Hello",
                "timestamp": "2024-01-01T00:00:00"
            },
            {
                "role": "assistant",
                "content": "Hi there!",
                "sources": ["doc-1"],
                "timestamp": "2024-01-01T00:01:00"
            }
        ]
        
        response = test_client.get(f"/api/chat/history?conversation_id={conversation_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == conversation_id
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"

    def test_get_chat_history_not_found(self, test_client):
        """Test chat history for non-existent conversation."""
        conversation_id = str(uuid.uuid4())
        
        response = test_client.get(f"/api/chat/history?conversation_id={conversation_id}")
        
        assert response.status_code == 404
        assert conversation_id in response.json()["detail"]

    def test_get_chat_history_with_limit(self, test_client):
        """Test chat history with limit parameter."""
        conversation_id = str(uuid.uuid4())
        # Create 10 messages
        conversations[conversation_id] = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}",
                "timestamp": f"2024-01-01T00:{i:02d}:00"
            }
            for i in range(10)
        ]
        
        response = test_client.get(f"/api/chat/history?conversation_id={conversation_id}&limit=5")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 5
        assert data["total_messages"] == 10
