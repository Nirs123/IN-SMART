"""Ingestion service for processing and storing documents."""

import time
from typing import Dict, Any, Optional, List
from api.storage.vector_store import VectorStore
from api.storage.file_store import FileStore
from api.processing.audio import AudioProcessor
from api.processing.text import TextProcessor
from api.services.chunking import ChunkingService
from api.services.embeddings import EmbeddingService
from api.models import Chunk


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
        start_time = time.time()

        try:
            # 1-2. Process document (download + extract text)
            text = self.process_document(document_id, file_type)

            # 3-4. Chunk and embed
            chunks = self.chunk_and_embed(text, document_id, metadata)

            # 5. Store in vector database
            uuids = self.store_chunks(chunks)

            processing_time = time.time() - start_time

            return {
                "document_id": document_id,
                "chunks_created": len(uuids),
                "status": "success",
                "processing_time": round(processing_time, 3),
            }
        except Exception as e:
            processing_time = time.time() - start_time
            raise ValueError(
                f"Ingestion failed for document '{document_id}': {e}"
            ) from e

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
        # Download file from file store
        file_data = self.file_store.download_file(document_id)

        file_type_lower = file_type.lower()

        if file_type_lower == "pdf":
            result = self.text_processor.extract_text_from_pdf(file_data)
            return result.get("text", "")

        elif file_type_lower == "audio":
            # Determine extension from document_id or default to mp3
            extension = document_id.rsplit(".", 1)[-1] if "." in document_id else "mp3"
            result = self.audio_processor.transcribe_bytes(file_data, file_extension=extension)
            return result.get("text", "")

        elif file_type_lower == "image":
            filename = document_id if "." in document_id else f"{document_id}.png"
            result = self.text_processor.ocr_image(file_data, filename=filename)
            return result.get("text", "")

        else:
            raise ValueError(f"Unsupported file type: '{file_type}'. Supported: pdf, audio, image")

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
        # Chunk the text
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata["document_id"] = document_id

        chunks = self.chunking_service.chunk_text(text, metadata=chunk_metadata)

        # Generate embeddings for each chunk
        result: List[Dict[str, Any]] = []
        for chunk in chunks:
            embedding = self.embedding_service.generate_embedding(chunk.text)
            result.append({
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "embedding": embedding,
                "document_id": document_id,
                "metadata": chunk.metadata,
            })

        return result

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
        if not chunks:
            return []

        # Extract document_id from the first chunk
        document_id = chunks[0]["document_id"]

        # Convert dicts back to Chunk models for vector_store
        chunk_models: List[Chunk] = []
        embeddings: List[List[float]] = []

        for chunk_data in chunks:
            chunk_model = Chunk(
                text=chunk_data["text"],
                chunk_index=chunk_data["chunk_index"],
                start_char=chunk_data["start_char"],
                end_char=chunk_data["end_char"],
                metadata=chunk_data.get("metadata", {}),
            )
            chunk_models.append(chunk_model)
            embeddings.append(chunk_data["embedding"])

        return self.vector_store.insert_chunks(chunk_models, embeddings, document_id)

    def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks associated with a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            int: Number of chunks deleted
        
        Raises:
            ValueError: If deletion fails
        """
        return self.vector_store.delete_by_document_id(document_id)

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
        # Delete old chunks first
        self.delete_document_chunks(document_id)

        # Re-ingest
        return self.ingest_document(document_id, file_type, metadata)

    def get_ingestion_status(self, document_id: str) -> Dict[str, Any]:
        """Get ingestion status for a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            Dict[str, Any]: Status information including:
                - document_id: Document identifier
                - is_ingested: Whether document is ingested
                - chunk_count: Number of chunks in vector store
        
        Raises:
            ValueError: If status check fails
        """
        chunk_count = self.vector_store.count_chunks(document_id)

        return {
            "document_id": document_id,
            "is_ingested": chunk_count > 0,
            "chunk_count": chunk_count,
        }
