"""Retrieval service for RAG document retrieval."""

from typing import List, Dict, Any, Optional
from api.storage.vector_store import VectorStore
from api.services.embeddings import EmbeddingService


class RetrievalService:
    """Service for retrieving relevant document chunks for RAG."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> None:
        """Initialize retrieval service.
        
        Args:
            vector_store: Vector store instance
            embedding_service: Embedding service instance
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score threshold
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: User query text
            document_ids: Optional list of document IDs to filter by
            top_k: Optional override for number of results
        
        Returns:
            List[Dict[str, Any]]: List of retrieved chunks with:
                - text: Chunk text
                - document_id: Source document ID
                - chunk_index: Chunk position
                - similarity: Similarity score
                - metadata: Chunk metadata
                - uuid: Weaviate UUID
        
        Raises:
            ValueError: If retrieval fails
        """
        k = top_k if top_k is not None else self.top_k

        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)

        # Build filter if document_ids provided
        where_filter: Optional[Dict[str, Any]] = None
        if document_ids:
            if len(document_ids) == 1:
                where_filter = {"doc_id": document_ids[0]}
            else:
                where_filter = {"doc_ids": document_ids}

        # Search vector store
        results = self.vector_store.search_similar(
            query_embedding=query_embedding,
            limit=k,
            where_filter=where_filter,
        )

        # Format results and compute similarity from distance
        formatted: List[Dict[str, Any]] = []
        for result in results:
            # Weaviate returns distance (lower = more similar)
            # Convert to similarity: similarity = 1 - distance
            distance = result.get("distance", 0.0)
            similarity = 1.0 - distance if distance is not None else 0.0

            # Filter by similarity threshold
            if similarity < self.similarity_threshold:
                continue

            formatted.append({
                "text": result.get("text", ""),
                "document_id": result.get("doc_id", ""),
                "chunk_index": result.get("chunk_index", 0),
                "similarity": round(similarity, 4),
                "metadata": result.get("metadata", {}),
                "uuid": result.get("uuid", ""),
            })

        return formatted

    def set_top_k(self, top_k: int) -> None:
        """Update top_k parameter.
        
        Args:
            top_k: New top_k value
        
        Raises:
            ValueError: If top_k <= 0
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        self.top_k = top_k

    def set_similarity_threshold(self, threshold: float) -> None:
        """Update similarity threshold.
        
        Args:
            threshold: New similarity threshold (0.0 to 1.0)
        
        Raises:
            ValueError: If threshold is out of range
        """
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        self.similarity_threshold = threshold

    def format_context(
        self,
        chunks: List[Dict[str, Any]],
        max_length: Optional[int] = None
    ) -> str:
        """Format retrieved chunks into context string for LLM.
        
        Args:
            chunks: List of retrieved chunks
            max_length: Optional maximum character length for context
        
        Returns:
            str: Formatted context string with citations
        """
        if not chunks:
            return ""

        parts: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            doc_id = chunk.get("document_id", "unknown")
            text = chunk.get("text", "")
            parts.append(f"[Source {i} - {doc_id}]\n{text}")

        context = "\n\n".join(parts)

        if max_length is not None and len(context) > max_length:
            context = context[:max_length]

        return context
