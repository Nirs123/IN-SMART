"""Embedding service for generating vector embeddings."""

import math
from typing import List, Optional
from mistralai import Mistral, MistralError


class EmbeddingService:
    """Service for generating embeddings using Mistral API."""

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-embed"
    ) -> None:
        """Initialize embedding service.
        
        Args:
            api_key: Mistral API key
            model: Embedding model name
        """
        self.api_key = api_key
        self.model = model
        self.client: Optional[Mistral] = None
        self.initialize()

    def initialize(self) -> None:
        """Initialize Mistral client.
        
        Raises:
            ValueError: If API key is invalid or client initialization fails
        """
        self.client = Mistral(self.api_key)
        if not self.client:
            raise ValueError("Failed to initialize Mistral client")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            List[float]: Embedding vector
        
        Raises:
            ValueError: If embedding generation fails
        """
        if not text:
            raise ValueError(f"Le texte ne doit pas être vide")
        try:
            embeddings_batch_response = self.client.embeddings.create(
            model=self.model,
            inputs=[text],
        )
        except MistralError as e:
            raise ValueError(f"Erreur lors de l'appel à l'API Mistral : {str(e)}")
        except Exception as e:
            raise ValueError(f"Erreur inattendue lors de l'appel à l'API : {str(e)}")

        return embeddings_batch_response.data[0].embedding

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            float: Cosine similarity score between -1 and 1
        
        Raises:
            ValueError: If embeddings have different dimensions
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Les embeddings doivent avoir la même dimension.")

        dot_product = sum(e1 * e2 for e1, e2 in zip(embedding1, embedding2))

        norm1 = math.sqrt(sum(e1 ** 2 for e1 in embedding1))
        norm2 = math.sqrt(sum(e2 ** 2 for e2 in embedding2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        return similarity
