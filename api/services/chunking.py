"""Chunking service for text segmentation."""

from typing import List, Dict, Any, Optional


class ChunkingService:
    """Service for chunking text into semantically meaningful segments."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> None:
        """Initialize chunking service.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text into fixed-size segments with overlap.
        
        Args:
            text: Text content to chunk
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries with:
                - text: Chunk text content
                - chunk_index: Position of chunk (0-indexed)
                - start_char: Starting character position in original text
                - end_char: Ending character position in original text
                - metadata: Chunk metadata (merged with input metadata)
        
        Raises:
            ValueError: If text is empty or chunk_size <= 0
        """
        pass

    def chunk_text_by_sentences(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text by sentences, respecting chunk_size limits.
        
        Attempts to break at sentence boundaries while maintaining
        approximate chunk_size. Overlaps at sentence boundaries.
        
        Args:
            text: Text content to chunk
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries (same format as chunk_text)
        
        Raises:
            ValueError: If text is empty or chunk_size <= 0
        """
        pass

    def chunk_text_by_paragraphs(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text by paragraphs, respecting chunk_size limits.
        
        Attempts to break at paragraph boundaries while maintaining
        approximate chunk_size. Overlaps at paragraph boundaries.
        
        Args:
            text: Text content to chunk
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries (same format as chunk_text)
        
        Raises:
            ValueError: If text is empty or chunk_size <= 0
        """
        pass

    def chunk_with_semantic_boundaries(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text using semantic boundaries (sentences/paragraphs) when possible.
        
        Falls back to fixed-size chunking if semantic boundaries don't align
        with chunk_size constraints.
        
        Args:
            text: Text content to chunk
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries (same format as chunk_text)
        
        Raises:
            ValueError: If text is empty or chunk_size <= 0
        """
        pass

    def set_chunk_size(self, chunk_size: int) -> None:
        """Update chunk size parameter.
        
        Args:
            chunk_size: New chunk size in characters
        
        Raises:
            ValueError: If chunk_size <= 0
        """
        pass

    def set_chunk_overlap(self, chunk_overlap: int) -> None:
        """Update chunk overlap parameter.
        
        Args:
            chunk_overlap: New overlap size in characters
        
        Raises:
            ValueError: If chunk_overlap < 0 or >= chunk_size
        """
        pass

    def get_chunk_statistics(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get statistics about a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries
        
        Returns:
            Dict[str, Any]: Statistics including:
                - total_chunks: Number of chunks
                - avg_chunk_size: Average chunk size in characters
                - min_chunk_size: Minimum chunk size
                - max_chunk_size: Maximum chunk size
                - total_characters: Total characters across all chunks
        """
        pass
