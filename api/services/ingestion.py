"""Ingestion service for processing and storing documents."""

from typing import Dict, Any, Optional, List
from api.storage.vector_store import VectorStore
from api.storage.file_store import FileStore
from api.processing.audio import AudioProcessor
from api.processing.text import TextProcessor
from api.services.chunking import ChunkingService
from api.services.embeddings import EmbeddingService


class IngestionService:
    """Service for ingesting documents into the RAG system."""

    def __init__(
        self,
        vector_store: VectorStore,
        file_store: FileStore,
        audio_processor: AudioProcessor,
        text_processor: TextProcessor,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService
    ) -> None:
        """Initialize ingestion service.
        
        Args:
            vector_store: Vector store instance
            file_store: File store instance
            audio_processor: Audio processor instance
            text_processor: Text processor instance
            chunking_service: Chunking service instance
            embedding_service: Embedding service instance
        """
        self.vector_store = vector_store
        self.file_store = file_store
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service

    def ingest_document(
        self,
        document_id: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest a document: process, chunk, embed, and store.
        
        Complete ingestion pipeline:
        1. Retrieve document from file store
        2. Process based on file type (STT/OCR/text extraction)
        3. Chunk the extracted text
        4. Generate embeddings for chunks
        5. Store chunks in vector database
        
        Args:
            document_id: Unique document identifier
            file_type: Document type ('pdf', 'audio', 'image')
            metadata: Optional document metadata
        
        Returns:
            Dict[str, Any]: Ingestion result with:
                - document_id: Document identifier
                - chunks_created: Number of chunks created
                - status: Ingestion status
                - processing_time: Time taken for ingestion
        
        Raises:
            ValueError: If document not found or processing fails
            FileNotFoundError: If document file doesn't exist
        """
        pass

    def process_document(
        self,
        document_id: str,
        file_type: str
    ) -> str:
        """Process document and extract text content.
        
        Args:
            document_id: Unique document identifier
            file_type: Document type ('pdf', 'audio', 'image')
        
        Returns:
            str: Extracted text content
        
        Raises:
            ValueError: If file type is unsupported or processing fails
            FileNotFoundError: If document file doesn't exist
        """
        pass

    def chunk_and_embed(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text and generate embeddings.
        
        Args:
            text: Text content to chunk and embed
            document_id: Document identifier for metadata
            metadata: Optional metadata to attach to chunks
        
        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries with embeddings:
                - text: Chunk text
                - chunk_index: Chunk position
                - embedding: Embedding vector
                - document_id: Source document ID
                - metadata: Chunk metadata
        
        Raises:
            ValueError: If chunking or embedding fails
        """
        pass

    def store_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """Store chunks with embeddings in vector database.
        
        Args:
            chunks: List of chunk dictionaries with embeddings
        
        Returns:
            List[str]: List of Weaviate UUIDs for stored chunks
        
        Raises:
            ValueError: If storage fails
        """
        pass

    def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks associated with a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            int: Number of chunks deleted
        
        Raises:
            ValueError: If deletion fails
        """
        pass

    def reingest_document(
        self,
        document_id: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Reingest a document (delete old chunks and create new ones).
        
        Args:
            document_id: Unique document identifier
            file_type: Document type
            metadata: Optional document metadata
        
        Returns:
            Dict[str, Any]: Ingestion result (same format as ingest_document)
        
        Raises:
            ValueError: If reingestion fails
        """
        pass

    def get_ingestion_status(self, document_id: str) -> Dict[str, Any]:
        """Get ingestion status for a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            Dict[str, Any]: Status information including:
                - document_id: Document identifier
                - is_ingested: Whether document is ingested
                - chunk_count: Number of chunks in vector store
                - last_ingested: Timestamp of last ingestion
        
        Raises:
            ValueError: If status check fails
        """
        pass
