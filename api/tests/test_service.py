"""Tests for service modules."""

import math

from requests import Response

import pytest
import os
from dotenv import load_dotenv
from api.services.chunking import ChunkingService
from api.services.embeddings import EmbeddingService
from api.services.ingestion import IngestionService
from api.services.llm import LLMService
from api.services.retrieval import RetrievalService
from api.models import Chunk
from mistralai import MistralError
from unittest.mock import MagicMock, patch
from api.services.llm import LLMService
from pathlib import Path 

load_dotenv(Path(__file__).parent.parent.parent / ".env")

@pytest.fixture
def api_key() -> str:
    """Retrieves the actual API key from the environment."""
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        pytest.skip("MISTRAL_API_KEY non définie, test d'intégration ignoré")
    return key

@pytest.fixture
def llm_service(api_key: str) -> LLMService:
    return LLMService(api_key=api_key)

@pytest.fixture
def initialized_llm_service(llm_service: LLMService) -> LLMService:
    """Service already initialized with a mocked client."""
    llm_service.client = MagicMock()
    return llm_service

@pytest.fixture
def embeddings_service(api_key: str) -> EmbeddingService:
    return EmbeddingService(api_key=api_key)

@pytest.fixture
def initialized_embeddings_service(embeddings_service: EmbeddingService) -> EmbeddingService:
    """Service already initialized with a mocked client."""
    embeddings_service.client = MagicMock()
    return embeddings_service

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
        with patch("api.services.embeddings.Mistral") as mock_mistral:
            service = EmbeddingService(api_key=api_key)
            
            mock_mistral.assert_called_once_with(api_key)
            assert service.client is not None

    def test_generate_embedding(self, initialized_embeddings_service: EmbeddingService) -> None:
        expected = [0.1, 0.2, 0.3]
        response = MagicMock()
        response.data = [MagicMock(embedding=expected)]
        initialized_embeddings_service.client.embeddings.create.return_value = response
        result = initialized_embeddings_service.generate_embedding("bonjour")

        assert result == expected

    def test_raises_on_mistral_error(self, initialized_embeddings_service: EmbeddingService):
        
        initialized_embeddings_service.client.embeddings.create.side_effect = MistralError("API down", Response())

        with pytest.raises(ValueError, match="Erreur lors de l'appel à l'API Mistral"):
            initialized_embeddings_service.generate_embedding("text")

    def test_empty_string(self, initialized_embeddings_service: EmbeddingService):
        with pytest.raises(ValueError, match="Le texte ne doit pas être vide"):
            initialized_embeddings_service.generate_embedding("")

    def test_identical_vectors_return_one(self, embeddings_service : EmbeddingService):
        v = [1.0, 2.0, 3.0]
        assert embeddings_service.compute_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self, embeddings_service : EmbeddingService):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert embeddings_service.compute_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self, embeddings_service : EmbeddingService):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert embeddings_service.compute_similarity(v1, v2) == pytest.approx(-1.0)

    def test_known_similarity(self, embeddings_service : EmbeddingService):
        v1 = [1.0, 1.0]
        v2 = [1.0, 0.0]
        expected = 1 / math.sqrt(2)
        assert embeddings_service.compute_similarity(v1, v2) == pytest.approx(expected)

    def test_zero_vector_returns_zero(self, embeddings_service : EmbeddingService):
        v1 = [0.0, 0.0]
        v2 = [1.0, 2.0]
        assert embeddings_service.compute_similarity(v1, v2) == 0.0

    def test_both_zero_vectors_return_zero(self, embeddings_service : EmbeddingService):
        v1 = [0.0, 0.0]
        assert embeddings_service.compute_similarity(v1, v1) == 0.0

    def test_different_dimensions_raise(self, embeddings_service : EmbeddingService):
        with pytest.raises(ValueError, match="Les embeddings doivent avoir la même dimension."):
            embeddings_service.compute_similarity([1.0, 2.0], [1.0])

    def test_symmetry(self, embeddings_service : EmbeddingService):
        v1 = [0.5, 0.3, 0.8]
        v2 = [0.1, 0.9, 0.2]
        assert embeddings_service.compute_similarity(v1, v2) == pytest.approx(
            embeddings_service.compute_similarity(v2, v1)
        )

    def test_result_bounded_between_minus_one_and_one(self, embeddings_service : EmbeddingService):
        v1 = [3.0, -1.0, 2.0]
        v2 = [-1.0, 4.0, 0.5]
        result = embeddings_service.compute_similarity(v1, v2)
        assert -1.0 <= result <= 1.0


class TestIngestionService:
    """Test cases for IngestionService."""

    @pytest.fixture
    def ingestion_service(self) -> IngestionService:
        """Create an IngestionService with all dependencies mocked."""
        vector_store = MagicMock()
        file_store = MagicMock()
        audio_processor = MagicMock()
        text_processor = MagicMock()
        chunking_service = MagicMock()
        embedding_service = MagicMock()

        service = IngestionService(
            vector_store=vector_store,
            file_store=file_store,
            audio_processor=audio_processor,
            text_processor=text_processor,
            chunking_service=chunking_service,
            embedding_service=embedding_service,
        )
        return service

    def test_ingest_document(self, ingestion_service: IngestionService) -> None:
        """Test full ingestion pipeline returns correct result dict."""
        # Setup mocks
        ingestion_service.file_store.download_file.return_value = b"pdf binary data"
        ingestion_service.text_processor.extract_text_from_pdf.return_value = {
            "text": "Hello world content"
        }
        ingestion_service.chunking_service.chunk_text.return_value = [
            Chunk(text="Hello world", chunk_index=0, start_char=0, end_char=11, metadata={}),
            Chunk(text="world content", chunk_index=1, start_char=6, end_char=19, metadata={}),
        ]
        ingestion_service.embedding_service.generate_embedding.return_value = [0.1, 0.2, 0.3]
        ingestion_service.vector_store.insert_chunks.return_value = ["uuid-1", "uuid-2"]

        result = ingestion_service.ingest_document("doc-1", "pdf")

        assert result["document_id"] == "doc-1"
        assert result["chunks_created"] == 2
        assert result["status"] == "success"
        assert "processing_time" in result

    def test_process_document_pdf(self, ingestion_service: IngestionService) -> None:
        """Test document processing for PDF files."""
        ingestion_service.file_store.download_file.return_value = b"pdf data"
        ingestion_service.text_processor.extract_text_from_pdf.return_value = {
            "text": "Extracted PDF text"
        }

        text = ingestion_service.process_document("doc.pdf", "pdf")

        assert text == "Extracted PDF text"
        ingestion_service.file_store.download_file.assert_called_once_with("doc.pdf")
        ingestion_service.text_processor.extract_text_from_pdf.assert_called_once_with(b"pdf data")

    def test_process_document_audio(self, ingestion_service: IngestionService) -> None:
        """Test document processing for audio files."""
        ingestion_service.file_store.download_file.return_value = b"audio data"
        ingestion_service.audio_processor.transcribe_bytes.return_value = {
            "text": "Transcribed audio"
        }

        text = ingestion_service.process_document("recording.mp3", "audio")

        assert text == "Transcribed audio"
        ingestion_service.audio_processor.transcribe_bytes.assert_called_once_with(
            b"audio data", file_extension="mp3"
        )

    def test_process_document_image(self, ingestion_service: IngestionService) -> None:
        """Test document processing for image files (OCR)."""
        ingestion_service.file_store.download_file.return_value = b"image data"
        ingestion_service.text_processor.ocr_image.return_value = {
            "text": "OCR extracted text"
        }

        text = ingestion_service.process_document("notes.png", "image")

        assert text == "OCR extracted text"
        ingestion_service.text_processor.ocr_image.assert_called_once_with(
            b"image data", filename="notes.png"
        )

    def test_process_document_unsupported_type(self, ingestion_service: IngestionService) -> None:
        """Test that unsupported file type raises ValueError."""
        ingestion_service.file_store.download_file.return_value = b"data"

        with pytest.raises(ValueError, match="Unsupported file type"):
            ingestion_service.process_document("doc.xyz", "xyz")

    def test_chunk_and_embed(self, ingestion_service: IngestionService) -> None:
        """Test chunking and embedding returns correct structure."""
        ingestion_service.chunking_service.chunk_text.return_value = [
            Chunk(text="chunk one", chunk_index=0, start_char=0, end_char=9, metadata={"document_id": "doc-1"}),
        ]
        ingestion_service.embedding_service.generate_embedding.return_value = [0.5, 0.6]

        result = ingestion_service.chunk_and_embed("chunk one", "doc-1")

        assert len(result) == 1
        assert result[0]["text"] == "chunk one"
        assert result[0]["chunk_index"] == 0
        assert result[0]["embedding"] == [0.5, 0.6]
        assert result[0]["document_id"] == "doc-1"

    def test_store_chunks(self, ingestion_service: IngestionService) -> None:
        """Test storing chunks delegates to vector_store correctly."""
        chunks = [
            {
                "text": "chunk text",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 10,
                "embedding": [0.1, 0.2],
                "document_id": "doc-1",
                "metadata": {},
            }
        ]
        ingestion_service.vector_store.insert_chunks.return_value = ["uuid-1"]

        uuids = ingestion_service.store_chunks(chunks)

        assert uuids == ["uuid-1"]
        ingestion_service.vector_store.insert_chunks.assert_called_once()

    def test_store_chunks_empty(self, ingestion_service: IngestionService) -> None:
        """Test storing empty chunk list returns empty list."""
        result = ingestion_service.store_chunks([])
        assert result == []

    def test_delete_document_chunks(self, ingestion_service: IngestionService) -> None:
        """Test deleting document chunks delegates to vector_store."""
        ingestion_service.vector_store.delete_by_document_id.return_value = 5

        count = ingestion_service.delete_document_chunks("doc-1")

        assert count == 5
        ingestion_service.vector_store.delete_by_document_id.assert_called_once_with("doc-1")

    def test_reingest_document(self, ingestion_service: IngestionService) -> None:
        """Test reingestion deletes old chunks then re-ingests."""
        ingestion_service.vector_store.delete_by_document_id.return_value = 3
        ingestion_service.file_store.download_file.return_value = b"pdf data"
        ingestion_service.text_processor.extract_text_from_pdf.return_value = {"text": "new text"}
        ingestion_service.chunking_service.chunk_text.return_value = [
            Chunk(text="new text", chunk_index=0, start_char=0, end_char=8, metadata={}),
        ]
        ingestion_service.embedding_service.generate_embedding.return_value = [0.1]
        ingestion_service.vector_store.insert_chunks.return_value = ["uuid-new"]

        result = ingestion_service.reingest_document("doc-1", "pdf")

        ingestion_service.vector_store.delete_by_document_id.assert_called_once_with("doc-1")
        assert result["status"] == "success"
        assert result["chunks_created"] == 1

    def test_get_ingestion_status_ingested(self, ingestion_service: IngestionService) -> None:
        """Test ingestion status for an ingested document."""
        ingestion_service.vector_store.count_chunks.return_value = 10

        status = ingestion_service.get_ingestion_status("doc-1")

        assert status["document_id"] == "doc-1"
        assert status["is_ingested"] is True
        assert status["chunk_count"] == 10

    def test_get_ingestion_status_not_ingested(self, ingestion_service: IngestionService) -> None:
        """Test ingestion status for a non-ingested document."""
        ingestion_service.vector_store.count_chunks.return_value = 0

        status = ingestion_service.get_ingestion_status("doc-new")

        assert status["is_ingested"] is False
        assert status["chunk_count"] == 0


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

    @pytest.fixture
    def retrieval_service(self) -> RetrievalService:
        """Create a RetrievalService with mocked dependencies."""
        vector_store = MagicMock()
        embedding_service = MagicMock()

        service = RetrievalService(
            vector_store=vector_store,
            embedding_service=embedding_service,
            top_k=5,
            similarity_threshold=0.7,
        )
        return service

    def test_retrieve(self, retrieval_service: RetrievalService) -> None:
        """Test basic retrieval with similarity threshold filtering."""
        retrieval_service.embedding_service.generate_embedding.return_value = [0.1, 0.2]
        retrieval_service.vector_store.search_similar.return_value = [
            {"text": "relevant chunk", "doc_id": "doc-1", "chunk_index": 0,
             "distance": 0.1, "metadata": {}, "uuid": "uuid-1"},
            {"text": "irrelevant chunk", "doc_id": "doc-2", "chunk_index": 1,
             "distance": 0.9, "metadata": {}, "uuid": "uuid-2"},
        ]

        results = retrieval_service.retrieve("test query")

        assert len(results) == 1  # Only the chunk with distance 0.1 (similarity 0.9) passes threshold 0.7
        assert results[0]["text"] == "relevant chunk"
        assert results[0]["similarity"] == 0.9
        assert results[0]["document_id"] == "doc-1"

    def test_retrieve_with_document_filter(self, retrieval_service: RetrievalService) -> None:
        """Test retrieval with document ID filter."""
        retrieval_service.embedding_service.generate_embedding.return_value = [0.1]
        retrieval_service.vector_store.search_similar.return_value = []

        retrieval_service.retrieve("query", document_ids=["doc-1"])

        call_args = retrieval_service.vector_store.search_similar.call_args
        assert call_args.kwargs["where_filter"] == {"doc_id": "doc-1"}

    def test_retrieve_with_multiple_document_filter(self, retrieval_service: RetrievalService) -> None:
        """Test retrieval with multiple document ID filter."""
        retrieval_service.embedding_service.generate_embedding.return_value = [0.1]
        retrieval_service.vector_store.search_similar.return_value = []

        retrieval_service.retrieve("query", document_ids=["doc-1", "doc-2"])

        call_args = retrieval_service.vector_store.search_similar.call_args
        assert call_args.kwargs["where_filter"] == {"doc_ids": ["doc-1", "doc-2"]}

    def test_retrieve_with_reranking(self, retrieval_service: RetrievalService) -> None:
        """Test retrieval with reranking re-scores and sorts results."""
        retrieval_service.embedding_service.generate_embedding.return_value = [0.5, 0.5]
        retrieval_service.vector_store.search_similar.return_value = [
            {"text": "chunk A", "doc_id": "doc-1", "chunk_index": 0,
             "distance": 0.2, "metadata": {}, "uuid": "uuid-1"},
            {"text": "chunk B", "doc_id": "doc-1", "chunk_index": 1,
             "distance": 0.3, "metadata": {}, "uuid": "uuid-2"},
        ]
        # After reranking, compute_similarity returns different scores
        retrieval_service.embedding_service.compute_similarity.side_effect = [0.95, 0.98]

        results = retrieval_service.retrieve_with_reranking("test query", top_k=2)

        # chunk B should now be first (score 0.98 > 0.95)
        assert len(results) == 2
        assert results[0]["similarity"] == 0.98
        assert results[1]["similarity"] == 0.95

    def test_retrieve_by_document(self, retrieval_service: RetrievalService) -> None:
        """Test retrieve_by_document delegates to retrieve with filter."""
        retrieval_service.embedding_service.generate_embedding.return_value = [0.1]
        retrieval_service.vector_store.search_similar.return_value = [
            {"text": "chunk", "doc_id": "doc-1", "chunk_index": 0,
             "distance": 0.1, "metadata": {}, "uuid": "uuid-1"},
        ]

        results = retrieval_service.retrieve_by_document("query", "doc-1")

        call_args = retrieval_service.vector_store.search_similar.call_args
        assert call_args.kwargs["where_filter"] == {"doc_id": "doc-1"}
        assert len(results) == 1

    def test_hybrid_retrieve(self, retrieval_service: RetrievalService) -> None:
        """Test hybrid retrieval combines semantic and keyword scores."""
        retrieval_service.embedding_service.generate_embedding.return_value = [0.1]
        retrieval_service.vector_store.search_similar.return_value = [
            {"text": "machine learning algorithms", "doc_id": "doc-1", "chunk_index": 0,
             "distance": 0.1, "metadata": {}, "uuid": "uuid-1"},
            {"text": "cooking recipes", "doc_id": "doc-2", "chunk_index": 1,
             "distance": 0.15, "metadata": {}, "uuid": "uuid-2"},
        ]

        results = retrieval_service.hybrid_retrieve("machine learning", top_k=2, keyword_weight=0.3)

        # "machine learning algorithms" has keyword overlap with query, so it should score higher
        assert len(results) >= 1

    def test_set_top_k_valid(self, retrieval_service: RetrievalService) -> None:
        """Test setting valid top_k."""
        retrieval_service.set_top_k(10)
        assert retrieval_service.top_k == 10

    def test_set_top_k_invalid(self, retrieval_service: RetrievalService) -> None:
        """Test setting invalid top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be greater than 0"):
            retrieval_service.set_top_k(0)

        with pytest.raises(ValueError, match="top_k must be greater than 0"):
            retrieval_service.set_top_k(-1)

    def test_set_similarity_threshold_valid(self, retrieval_service: RetrievalService) -> None:
        """Test setting valid similarity threshold."""
        retrieval_service.set_similarity_threshold(0.5)
        assert retrieval_service.similarity_threshold == 0.5

    def test_set_similarity_threshold_invalid(self, retrieval_service: RetrievalService) -> None:
        """Test setting invalid similarity threshold raises ValueError."""
        with pytest.raises(ValueError, match="Similarity threshold must be between 0.0 and 1.0"):
            retrieval_service.set_similarity_threshold(-0.1)

        with pytest.raises(ValueError, match="Similarity threshold must be between 0.0 and 1.0"):
            retrieval_service.set_similarity_threshold(1.5)

    def test_format_context(self, retrieval_service: RetrievalService) -> None:
        """Test context formatting with citations."""
        chunks = [
            {"text": "First chunk content", "document_id": "doc-1", "similarity": 0.9},
            {"text": "Second chunk content", "document_id": "doc-2", "similarity": 0.8},
        ]

        context = retrieval_service.format_context(chunks)

        assert "[Source 1 - doc-1]" in context
        assert "First chunk content" in context
        assert "[Source 2 - doc-2]" in context
        assert "Second chunk content" in context

    def test_format_context_empty(self, retrieval_service: RetrievalService) -> None:
        """Test formatting empty chunks returns empty string."""
        assert retrieval_service.format_context([]) == ""

    def test_format_context_max_length(self, retrieval_service: RetrievalService) -> None:
        """Test context truncation with max_length."""
        chunks = [
            {"text": "A" * 100, "document_id": "doc-1", "similarity": 0.9},
        ]

        context = retrieval_service.format_context(chunks, max_length=50)

        assert len(context) == 50

    def test_get_retrieval_statistics(self, retrieval_service: RetrievalService) -> None:
        """Test retrieval statistics computation."""
        chunks = [
            {"text": "a", "document_id": "doc-1", "similarity": 0.9},
            {"text": "b", "document_id": "doc-2", "similarity": 0.7},
            {"text": "c", "document_id": "doc-1", "similarity": 0.8},
        ]

        stats = retrieval_service.get_retrieval_statistics("test query", chunks)

        assert stats["query"] == "test query"
        assert stats["num_results"] == 3
        assert stats["avg_similarity"] == pytest.approx(0.8, abs=0.01)
        assert stats["min_similarity"] == 0.7
        assert stats["max_similarity"] == 0.9
        assert stats["unique_documents"] == 2

    def test_get_retrieval_statistics_empty(self, retrieval_service: RetrievalService) -> None:
        """Test statistics for empty results."""
        stats = retrieval_service.get_retrieval_statistics("query", [])

        assert stats["num_results"] == 0
        assert stats["avg_similarity"] == 0.0
        assert stats["unique_documents"] == 0