"""Tests for service modules."""

import pytest
import os
from dotenv import load_dotenv
from api.services.chunking import ChunkingService
from api.services.embeddings import EmbeddingService
from api.services.ingestion import IngestionService
from api.services.llm import LLMService
from api.services.retrieval import RetrievalService
from api.models import Chunk

from unittest.mock import MagicMock, patch
from api.services.llm import LLMService
from pathlib import Path 

load_dotenv(Path(__file__).parent.parent.parent / ".env")

@pytest.fixture
def api_key() -> str:
    """Récupère la vraie clé API depuis l'environnement."""
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        pytest.skip("MISTRAL_API_KEY non définie, test d'intégration ignoré")
    return key

@pytest.fixture
def llm_service(api_key: str) -> LLMService:
    return LLMService(api_key=api_key)

@pytest.fixture
def initialized_llm_service(llm_service: LLMService) -> LLMService:
    """Service déjà initialisé avec un client mocké."""
    llm_service.client = MagicMock()
    return llm_service

class TestChunkingService:
    """Test cases for ChunkingService."""

    # chunk_text tests
    def test_chunk_text_basic(self) -> None:
        """Test basic text chunking with default parameters."""
        service = ChunkingService(chunk_size=100, chunk_overlap=20)
        text = "a" * 250  # 250 characters

        chunks = service.chunk_text(text)

        # With chunk_size=100 and overlap=20
        # Chunks: 0-100, 80-180, 160-250
        assert len(chunks) == 3
        assert chunks[0].text == "a" * 100
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 100
        assert chunks[0].chunk_index == 0

    def test_chunk_text_with_overlap(self) -> None:
        """Test that chunks overlap correctly."""
        service = ChunkingService(chunk_size=100, chunk_overlap=30)
        text = "a" * 200

        chunks = service.chunk_text(text)

        # Verify overlap: each chunk should start 70 chars after the previous
        # First chunk: 0-100, second chunk: 70-170, third chunk: 140-200
        assert len(chunks) == 3
        assert chunks[1].start_char == 70  # 100 - 30 = 70 overlap
        assert chunks[0].end_char == 100
        # Verify overlap content
        assert chunks[0].text[-30:] == chunks[1].text[:30]

    def test_chunk_text_shorter_than_chunk_size(self) -> None:
        """Test chunking text shorter than chunk_size returns single chunk."""
        service = ChunkingService(chunk_size=100, chunk_overlap=20)
        text = "short text"

        chunks = service.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == "short text"
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)

    def test_chunk_text_exact_multiple(self) -> None:
        """Test chunking text that is exact multiple of chunk_size."""
        service = ChunkingService(chunk_size=100, chunk_overlap=0)
        text = "a" * 300  # Exactly 3 chunks

        chunks = service.chunk_text(text)

        assert len(chunks) == 3
        assert chunks[0].text == "a" * 100
        assert chunks[1].text == "a" * 100
        assert chunks[2].text == "a" * 100

    def test_chunk_text_with_metadata(self) -> None:
        """Test that metadata is attached to each chunk."""
        service = ChunkingService(chunk_size=50, chunk_overlap=10)
        text = "a" * 100
        metadata = {"source": "test", "page": 1}

        chunks = service.chunk_text(text, metadata)

        # With chunk_size=50 and overlap=10
        # Chunks: 0-50, 40-90, 80-100
        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk.metadata == {"source": "test", "page": 1}

    def test_chunk_text_empty_raises_error(self) -> None:
        """Test that empty text raises ValueError."""
        service = ChunkingService()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            service.chunk_text("")

    def test_chunk_text_invalid_chunk_size(self) -> None:
        """Test that invalid chunk_size raises ValueError."""
        service = ChunkingService(chunk_size=100, chunk_overlap=20)
        service.chunk_size = 0  # Manually set invalid value

        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            service.chunk_text("some text")

    def test_chunk_text_invalid_overlap(self) -> None:
        """Test that overlap >= chunk_size raises ValueError."""
        service = ChunkingService(chunk_size=50, chunk_overlap=60)

        with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
            service.chunk_text("some text")

class TestEmbeddingService:
    """Test cases for EmbeddingService."""

    def test_initialize(self) -> None:
        """Test EmbeddingService initialization.
        
        Should:
        - Initialize with API key and model
        - Create Mistral client
        """
        pass

    def test_generate_embedding(self) -> None:
        """Test single embedding generation.
        
        Should:
        - Generate embedding for text
        - Return vector of correct dimension
        """
        pass

    def test_generate_embeddings(self) -> None:
        """Test batch embedding generation.
        
        Should:
        - Generate embeddings for multiple texts
        - Handle batching correctly
        - Return list of embeddings
        """
        pass

    def test_compute_similarity(self) -> None:
        """Test similarity computation.
        
        Should:
        - Compute cosine similarity
        - Return value between -1 and 1
        """
        pass


class TestIngestionService:
    """Test cases for IngestionService."""

    def test_ingest_document(self) -> None:
        """Test document ingestion pipeline.
        
        Should:
        - Process document
        - Chunk and embed text
        - Store in vector database
        - Return ingestion result
        """
        pass

    def test_process_document(self) -> None:
        """Test document processing.
        
        Should:
        - Extract text based on file type
        - Handle PDF, audio, and image files
        """
        pass

    def test_chunk_and_embed(self) -> None:
        """Test chunking and embedding.
        
        Should:
        - Chunk text
        - Generate embeddings
        - Return chunks with embeddings
        """
        pass

    def test_delete_document_chunks(self) -> None:
        """Test deleting document chunks.
        
        Should:
        - Delete all chunks for document
        - Return count of deleted chunks
        """
        pass


class TestLLMService:
    """Test cases for LLMService."""
    def test_initialize(self, api_key: str) -> None:
        """Test the initialization and creation of the client."""
        with patch("api.services.llm.Mistral") as mock_mistral:
            service = LLMService(api_key=api_key)
            service.initialize()
            
            mock_mistral.assert_called_once_with(api_key)
            assert service.client is not None

    def test_generate_response(self, initialized_llm_service: LLMService) -> None:
        """Testing response generation via the mocked client."""
        mock_response = MagicMock()
        mock_response.text = "Hello! How can I help you?"
        initialized_llm_service.client.chat.complete.return_value = mock_response

        result = initialized_llm_service.generate_response("Hello")

        assert result.text == "Hello! How can I help you?"
        initialized_llm_service.client.chat.complete.assert_called_once()

    def test_format_messages(self, llm_service: LLMService) -> None:
        """Test the message formatting (system + user)."""
        user_msg = "What time is it ?"
        sys_prompt = "You are a helpful assistant"
        
        messages = llm_service.format_messages(user_msg, system_prompt=sys_prompt)
        
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": sys_prompt}
        assert messages[1] == {"role": "user", "content": user_msg}

    def test_format_messages_no_system(self, llm_service: LLMService) -> None:
        """Test the formatting without a system prompt."""
        messages = llm_service.format_messages("Bonjour")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


class TestRetrievalService:
    """Test cases for RetrievalService."""

    def test_retrieve(self) -> None:
        """Test basic retrieval.
        
        Should:
        - Retrieve relevant chunks
        - Filter by similarity threshold
        - Return top_k results
        """
        pass

    def test_retrieve_with_reranking(self) -> None:
        """Test retrieval with reranking.
        
        Should:
        - Retrieve more chunks
        - Rerank for better relevance
        - Return top_k reranked results
        """
        pass

    def test_retrieve_by_document(self) -> None:
        """Test retrieval from specific document.
        
        Should:
        - Filter by document_id
        - Return relevant chunks
        """
        pass

    def test_format_context(self) -> None:
        """Test context formatting.
        
        Should:
        - Format chunks into context string
        - Include citations
        - Respect max_length if provided
        """
        pass
