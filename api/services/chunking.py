"""Chunking service for text segmentation."""
import re
from typing import Any, Optional

from api.models import Chunk


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
        metadata: Optional[dict[str, Any]] = None
    ) -> list[Chunk]:
        """Chunk text into fixed-size segments with overlap.

        Args:
            text: Text content to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            list[Chunk]: A list of chunks model.

        Raises:
            ValueError: If text is empty, chunk_size <= 0 or chunk_overlap >= chunk_size
        """
        if not text:
            raise ValueError("Text cannot be empty.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        chunks: list[Chunk] = []
        text_length = len(text)
        base_metadata = metadata or {}

        step = self.chunk_size - self.chunk_overlap

        start = 0
        chunk_index = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            chunk = Chunk(
                text=text[start:end],
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
                metadata=base_metadata.copy()
            )
            chunks.append(chunk)

            if end == text_length:
                break

            start += step
            chunk_index += 1

        return chunks

    def chunk_text_by_paragraphs(
            self,
            text: str,
            metadata: Optional[dict[str, Any]] = None
    ) -> list[Chunk]:
        """Chunk text by paragraphs, respecting chunk_size limits.

        Attempts to break at paragraph boundaries while maintaining
        approximate chunk_size. Overlaps at paragraph boundaries.

        Args:
            text: Text content to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            list[Chunk]: A list of chunks model.

        Raises:
            ValueError: If text is empty, chunk_size <= 0 or chunk_overlap >= chunk_size
        """
        if not text:
            raise ValueError("Text cannot be empty.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        chunks: list[Chunk] = []
        chunk_index = 0
        base_metadata = metadata or {}

        segments = []
        for match in re.finditer(r'(.*?)(?:(?:\r?\n){2,}|\Z)', text, flags=re.DOTALL):
            seg_text = match.group(0)
            if not seg_text:
                continue
            segments.append({
                'text': seg_text,
                'start': match.start(),
                'end': match.end()
            })

        current_segments = []
        current_len = 0

        for seg in segments:
            seg_len = len(seg['text'])

            if current_len + seg_len > self.chunk_size and current_segments:
                chunk_text = "".join(s['text'] for s in current_segments)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=current_segments[0]['start'],
                    end_char=current_segments[-1]['end'],
                    metadata=base_metadata.copy()
                ))
                chunk_index += 1

                overlap_segments = []
                overlap_len = 0

                for s in reversed(current_segments):
                    if overlap_len + len(s['text']) <= self.chunk_overlap:
                        overlap_segments.insert(0, s)
                        overlap_len += len(s['text'])
                    else:
                        break

                current_segments = overlap_segments
                current_len = overlap_len

            current_segments.append(seg)
            current_len += seg_len

        if current_segments:
            chunk_text = "".join(s['text'] for s in current_segments)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=current_segments[0]['start'],
                end_char=current_segments[-1]['end'],
                metadata=base_metadata.copy()
            ))

        return chunks

    def set_chunk_size(self, chunk_size: int) -> None:
        """Update chunk size parameter.
        
        Args:
            chunk_size: New chunk size in characters
        
        Raises:
            ValueError: If chunk_size <= 0
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        self.chunk_size = chunk_size

    def set_chunk_overlap(self, chunk_overlap: int) -> None:
        """Update chunk overlap parameter.
        
        Args:
            chunk_overlap: New overlap size in characters
        
        Raises:
            ValueError: If chunk_overlap < 0 or >= chunk_size
        """
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative.")
        if chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def get_chunk_statistics(chunks: list[Chunk]) -> dict[str, Any]:
        """Get statistics about a list of chunks.

        Args:
            chunks: list of chunk models.

        Returns:
            dict[str, Any]: Statistics including:
                - total_chunks: Number of chunks
                - avg_chunk_size: Average chunk size in characters
                - min_chunk_size: Minimum chunk size
                - max_chunk_size: Maximum chunk size
                - total_characters: Total characters across all chunks
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0.0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_characters": 0
            }

        chunk_lengths = [len(chunk.text) for chunk in chunks]

        total_chunks = len(chunks)
        total_characters = sum(chunk_lengths)

        return {
            "total_chunks": total_chunks,
            "avg_chunk_size": total_characters / total_chunks,
            "min_chunk_size": min(chunk_lengths),
            "max_chunk_size": max(chunk_lengths),
            "total_characters": total_characters
        }
