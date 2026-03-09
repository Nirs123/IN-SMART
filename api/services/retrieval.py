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

    def retrieve_with_reranking(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        rerank_top_n: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks and rerank them for better relevance.
        
        Retrieves more chunks than requested, then reranks them
        using embedding similarity for a more precise comparison.
        
        Args:
            query: User query text
            document_ids: Optional list of document IDs to filter by
            top_k: Number of final results to return
            rerank_top_n: Number of chunks to retrieve before reranking
        
        Returns:
            List[Dict[str, Any]]: List of reranked chunks (same format as retrieve)
        
        Raises:
            ValueError: If retrieval or reranking fails
        """
        k = top_k if top_k is not None else self.top_k

        # Save and temporarily override threshold to get more candidates
        original_threshold = self.similarity_threshold
        self.similarity_threshold = 0.0

        candidates = self.retrieve(query, document_ids=document_ids, top_k=rerank_top_n)

        # Restore threshold
        self.similarity_threshold = original_threshold

        # Rerank using embedding similarity between query and chunk texts
        query_embedding = self.embedding_service.generate_embedding(query)

        for chunk in candidates:
            chunk_embedding = self.embedding_service.generate_embedding(chunk["text"])
            chunk["similarity"] = round(
                self.embedding_service.compute_similarity(query_embedding, chunk_embedding),
                4,
            )

        # Sort by reranked similarity (descending), filter, and return top_k
        candidates.sort(key=lambda c: c["similarity"], reverse=True)

        reranked = [
            c for c in candidates if c["similarity"] >= self.similarity_threshold
        ]

        return reranked[:k]

    def retrieve_by_document(
        self,
        query: str,
        document_id: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks from a specific document.
        
        Args:
            query: User query text
            document_id: Document identifier
            top_k: Optional override for number of results
        
        Returns:
            List[Dict[str, Any]]: List of retrieved chunks (same format as retrieve)
        
        Raises:
            ValueError: If retrieval fails
        """
        return self.retrieve(query, document_ids=[document_id], top_k=top_k)

    def hybrid_retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining semantic and keyword search.
        
        Args:
            query: User query text
            document_ids: Optional list of document IDs to filter by
            top_k: Optional override for number of results
            keyword_weight: Weight for keyword matching (0.0 to 1.0)
        
        Returns:
            List[Dict[str, Any]]: List of retrieved chunks (same format as retrieve)
        
        Raises:
            ValueError: If retrieval fails
        """
        k = top_k if top_k is not None else self.top_k

        # Save and temporarily disable threshold to get enough candidates
        original_threshold = self.similarity_threshold
        self.similarity_threshold = 0.0

        # Get semantic results (more than needed to allow re-scoring)
        semantic_results = self.retrieve(query, document_ids=document_ids, top_k=k * 3)

        # Restore threshold
        self.similarity_threshold = original_threshold

        # Compute keyword overlap score (Jaccard similarity)
        query_tokens = set(query.lower().split())

        for chunk in semantic_results:
            chunk_tokens = set(chunk["text"].lower().split())

            if query_tokens or chunk_tokens:
                intersection = query_tokens & chunk_tokens
                union = query_tokens | chunk_tokens
                keyword_score = len(intersection) / len(union) if union else 0.0
            else:
                keyword_score = 0.0

            # Combine scores
            semantic_score = chunk["similarity"]
            combined = (1.0 - keyword_weight) * semantic_score + keyword_weight * keyword_score
            chunk["similarity"] = round(combined, 4)

        # Sort by combined score, filter, and return top_k
        semantic_results.sort(key=lambda c: c["similarity"], reverse=True)

        filtered = [
            c for c in semantic_results if c["similarity"] >= self.similarity_threshold
        ]

        return filtered[:k]

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

    def get_retrieval_statistics(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get statistics about retrieval results.
        
        Args:
            query: Original query
            retrieved_chunks: Retrieved chunks
        
        Returns:
            Dict[str, Any]: Statistics including:
                - query: Original query
                - num_results: Number of results
                - avg_similarity: Average similarity score
                - min_similarity: Minimum similarity score
                - max_similarity: Maximum similarity score
                - unique_documents: Number of unique source documents
        """
        if not retrieved_chunks:
            return {
                "query": query,
                "num_results": 0,
                "avg_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "unique_documents": 0,
            }

        similarities = [c.get("similarity", 0.0) for c in retrieved_chunks]
        unique_docs = {c.get("document_id", "") for c in retrieved_chunks}

        return {
            "query": query,
            "num_results": len(retrieved_chunks),
            "avg_similarity": round(sum(similarities) / len(similarities), 4),
            "min_similarity": round(min(similarities), 4),
            "max_similarity": round(max(similarities), 4),
            "unique_documents": len(unique_docs),
        }
