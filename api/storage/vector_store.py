"""Vector store connector for Weaviate."""

from typing import List, Dict, Optional, Any
import json
import logging
from urllib.parse import urlparse

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5

from api.models import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Weaviate client wrapper for vector storage operations."""

    def __init__(self, url: str, class_name: str = "DocumentChunk") -> None:
        """Initialize Weaviate client.
        
        Args:
            url: Weaviate server URL (e.g., http://localhost:8080)
            class_name: Name of the Weaviate collection for document chunks
        """
        self.url = url
        self.class_name = class_name
        self.client: Optional[weaviate.WeaviateClient] = None

    def connect(self) -> None:
        """Establish connection to Weaviate server.
        
        Raises:
            ConnectionError: If connection to Weaviate fails
        """
        try:
            parsed_url = urlparse(self.url)
            host = parsed_url.hostname or "localhost"
            port = parsed_url.port or 8080
            secure = parsed_url.scheme == "https"
            
            logger.info(f"Connecting to Weaviate at {host}:{port} (secure={secure})")
            
            self.client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=secure,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=secure,
            )
            
            if not self.client.is_ready():
                raise ConnectionError("Weaviate server is not ready")
            
            logger.info("Successfully connected to Weaviate")
            
        except Exception as e:
            error_msg = f"Failed to connect to Weaviate at {self.url}: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def disconnect(self) -> None:
        """Close connection to Weaviate server."""
        if self.client is not None:
            try:
                self.client.close()
                logger.info("Disconnected from Weaviate")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self.client = None

    def _get_collection(self) -> Any:
        """Get the collection reference.
        
        Returns:
            Collection reference
        
        Raises:
            ValueError: If client is not connected or collection doesn't exist
        """
        if self.client is None:
            raise ValueError("Not connected to Weaviate. Call connect() first.")
        
        if not self.client.collections.exists(self.class_name):
            raise ValueError(f"Collection '{self.class_name}' does not exist. Call create_schema() first.")
        
        return self.client.collections.get(self.class_name)

    def create_schema(self) -> None:
        """Create Weaviate schema for document chunks.
        
        Schema includes:
        - doc_id: Reference to source document
        - text: The chunk text content
        - chunk_index: Position of chunk in document
        - start_char: Starting character position in original document
        - end_char: Ending character position in original document
        - metadata: Additional metadata (JSON string)
        
        Raises:
            ValueError: If schema creation fails
        """
        if self.client is None:
            raise ValueError("Not connected to Weaviate. Call connect() first.")
        
        try:
            # Check if collection already exists
            if self.client.collections.exists(self.class_name):
                logger.info(f"Collection '{self.class_name}' already exists")
                return
            
            logger.info(f"Creating collection '{self.class_name}'")
            
            # Create collection with properties
            self.client.collections.create(
                name=self.class_name,
                description="Document chunks for RAG system",
                properties=[
                    Property(
                        name="doc_id",
                        data_type=DataType.TEXT,
                        description="Reference to source document",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="text",
                        data_type=DataType.TEXT,
                        description="The chunk text content",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="chunk_index",
                        data_type=DataType.INT,
                        description="Position of chunk in document",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="start_char",
                        data_type=DataType.INT,
                        description="Starting character position in original document",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="end_char",
                        data_type=DataType.INT,
                        description="Ending character position in original document",
                        skip_vectorization=True,
                    ),
                    Property(
                        name="metadata",
                        data_type=DataType.TEXT,
                        description="Additional metadata as JSON string",
                        skip_vectorization=True,
                    ),
                ],
                # Use self-managed vectors (embeddings provided externally)
                vector_index_config=Configure.VectorIndex.hnsw(),
                vectorizer_config=Configure.Vectorizer.none(),
            )
            
            logger.info(f"Successfully created collection '{self.class_name}'")
            
        except Exception as e:
            error_msg = f"Failed to create schema: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def delete_schema(self) -> None:
        """Delete the Weaviate schema and all data.
        
        Warning: This will delete all stored chunks.
        
        Raises:
            ValueError: If schema deletion fails
        """
        if self.client is None:
            raise ValueError("Not connected to Weaviate. Call connect() first.")
        
        try:
            if self.client.collections.exists(self.class_name):
                logger.warning(f"Deleting collection '{self.class_name}' and all data")
                self.client.collections.delete(self.class_name)
                logger.info(f"Successfully deleted collection '{self.class_name}'")
            else:
                logger.info(f"Collection '{self.class_name}' does not exist, nothing to delete")
                
        except Exception as e:
            error_msg = f"Failed to delete schema: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def insert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        document_id: str,
    ) -> List[str]:
        """Insert document chunks with embeddings into Weaviate.
        
        Args:
            chunks: List of Chunk model instances
            embeddings: List of embedding vectors for each chunk
            document_id: Document identifier to associate with all chunks
        
        Returns:
            List[str]: List of Weaviate UUIDs for inserted chunks
        
        Raises:
            ValueError: If chunks and embeddings length mismatch or insertion fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        
        if not chunks:
            return []
        
        collection = self._get_collection()
        uuids: List[str] = []
        
        try:
            for chunk, embedding in zip(chunks, embeddings):
                # Prepare properties from Chunk model with document_id
                properties = {
                    "doc_id": document_id,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "metadata": json.dumps(chunk.metadata),
                }
                
                # Generate deterministic UUID based on document_id and chunk_index
                chunk_uuid = generate_uuid5(f"{document_id}_{chunk.chunk_index}")
                
                # Insert with vector
                collection.data.insert(
                    uuid=chunk_uuid,
                    properties=properties,
                    vector=embedding,
                )
                
                uuids.append(str(chunk_uuid))
            
            logger.info(f"Inserted {len(uuids)} chunks into '{self.class_name}' for document '{document_id}'")
            return uuids
            
        except Exception as e:
            error_msg = f"Failed to insert chunks: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results to return
            where_filter: Optional filter conditions (e.g., {"doc_id": "doc123"})
        
        Returns:
            List[Dict[str, Any]]: List of similar chunks with:
                - doc_id: Source document ID
                - text: Chunk text
                - chunk_index: Chunk position
                - start_char: Starting character position
                - end_char: Ending character position
                - metadata: Additional metadata (parsed from JSON)
                - distance: Similarity distance
                - uuid: Weaviate UUID
        
        Raises:
            ValueError: If search fails
        """
        collection = self._get_collection()
        
        try:
            # Build the query with filter if provided
            if where_filter:
                filters = self._build_filter(where_filter)
                query = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=limit,
                    return_metadata=["distance"],
                    filters=filters,
                )
            else:
                query = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=limit,
                    return_metadata=["distance"],
                )
            
            # Execute query
            results = query.objects
            
            # Format results
            formatted_results: List[Dict[str, Any]] = []
            for obj in results:
                properties = obj.properties
                metadata_str = properties.get("metadata", "{}")
                
                # Parse metadata JSON
                try:
                    metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                except json.JSONDecodeError:
                    metadata = {}
                
                formatted_results.append({
                    "doc_id": properties.get("doc_id", ""),
                    "text": properties.get("text", ""),
                    "chunk_index": properties.get("chunk_index", 0),
                    "start_char": properties.get("start_char", 0),
                    "end_char": properties.get("end_char", 0),
                    "metadata": metadata,
                    "distance": obj.metadata.distance if obj.metadata else None,
                    "uuid": str(obj.uuid),
                })
            
            logger.debug(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
            
        except Exception as e:
            error_msg = f"Failed to search similar chunks: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def _build_filter(self, where_filter: Dict[str, Any]) -> Filter:
        """Build Weaviate filter from dictionary.
        
        Args:
            where_filter: Filter dictionary (e.g., {"doc_id": "doc123"})
        
        Returns:
            Filter object
        """
        filters_list: List[Filter] = []
        
        for key, value in where_filter.items():
            if key == "doc_id":
                filters_list.append(Filter.by_property("doc_id").equal(value))
            elif key == "doc_ids" and isinstance(value, list):
                filters_list.append(Filter.by_property("doc_id").contains_any(value))
            elif key == "chunk_index_min":
                filters_list.append(Filter.by_property("chunk_index").greater_or_equal(value))
            elif key == "chunk_index_max":
                filters_list.append(Filter.by_property("chunk_index").less_or_equal(value))
        
        if not filters_list:
            # Return a filter that matches everything
            return Filter.by_property("chunk_index").greater_or_equal(0)
        
        # Combine filters with AND
        result = filters_list[0]
        for f in filters_list[1:]:
            result = result & f
        
        return result

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks associated with a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            int: Number of chunks deleted
        
        Raises:
            ValueError: If deletion fails
        """
        collection = self._get_collection()
        
        try:
            # Build filter for doc_id property
            filter_obj = Filter.by_property("doc_id").equal(document_id)
            
            # Delete with filter
            result = collection.data.delete_many(where=filter_obj)
            
            deleted_count = result.successful if hasattr(result, 'successful') else 0
            logger.info(f"Deleted {deleted_count} chunks for document '{document_id}'")
            
            return deleted_count
            
        except Exception as e:
            error_msg = f"Failed to delete chunks for document '{document_id}': {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chunk by its Weaviate UUID.
        
        Args:
            chunk_id: Weaviate UUID
        
        Returns:
            Optional[Dict[str, Any]]: Chunk data if found, None otherwise
        
        Raises:
            ValueError: If retrieval fails
        """
        collection = self._get_collection()
        
        try:
            result = collection.query.fetch_object_by_id(uuid=chunk_id)
            
            if result is None:
                return None
            
            properties = result.properties
            metadata_str = properties.get("metadata", "{}")
            
            # Parse metadata JSON
            try:
                metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            except json.JSONDecodeError:
                metadata = {}
            
            return {
                "doc_id": properties.get("doc_id", ""),
                "text": properties.get("text", ""),
                "chunk_index": properties.get("chunk_index", 0),
                "start_char": properties.get("start_char", 0),
                "end_char": properties.get("end_char", 0),
                "metadata": metadata,
                "uuid": str(result.uuid),
            }
            
        except Exception as e:
            error_msg = f"Failed to retrieve chunk '{chunk_id}': {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def count_chunks(self, document_id: Optional[str] = None) -> int:
        """Count total chunks, optionally filtered by document.
        
        Args:
            document_id: Optional document ID to filter by
        
        Returns:
            int: Total number of chunks
        
        Raises:
            ValueError: If count query fails
        """
        collection = self._get_collection()
        
        try:
            if document_id:
                filter_obj = Filter.by_property("doc_id").equal(document_id)
                result = collection.aggregate.over_all(
                    filters=filter_obj,
                    total_count=True,
                )
            else:
                result = collection.aggregate.over_all(total_count=True)
            
            count = result.total_count if hasattr(result, 'total_count') else 0
            return count
            
        except Exception as e:
            error_msg = f"Failed to count chunks: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def batch_insert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        document_id: str,
        batch_size: int = 100
    ) -> List[str]:
        """Insert chunks in batches for better performance.
        
        Args:
            chunks: List of Chunk model instances
            embeddings: List of embedding vectors
            document_id: Document identifier to associate with all chunks
            batch_size: Number of chunks per batch
        
        Returns:
            List[str]: List of Weaviate UUIDs for all inserted chunks
        
        Raises:
            ValueError: If batch insertion fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        
        if not chunks:
            return []
        
        collection = self._get_collection()
        all_uuids: List[str] = []
        
        try:
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                # Prepare batch data
                data_objects: List[Dict[str, Any]] = []
                
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    properties = {
                        "doc_id": document_id,
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "metadata": json.dumps(chunk.metadata),
                    }
                    
                    chunk_uuid = generate_uuid5(f"{document_id}_{chunk.chunk_index}")
                    
                    data_objects.append({
                        "uuid": chunk_uuid,
                        "properties": properties,
                        "vector": embedding,
                    })
                
                # Use batch insertion
                with collection.batch.dynamic() as batch:
                    for obj in data_objects:
                        batch.add_object(
                            uuid=obj["uuid"],
                            properties=obj["properties"],
                            vector=obj["vector"],
                        )
                
                # Collect UUIDs
                for obj in data_objects:
                    all_uuids.append(str(obj["uuid"]))
                
                logger.debug(f"Inserted batch {i // batch_size + 1}: {len(batch_chunks)} chunks")
            
            logger.info(f"Batch inserted {len(all_uuids)} chunks into '{self.class_name}' for document '{document_id}'")
            return all_uuids
            
        except Exception as e:
            error_msg = f"Failed to batch insert chunks: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
