"""Tests for storage modules (vector store and file store)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from api.storage.vector_store import VectorStore
from api.models import Chunk
from api.storage.file_store import FileStore
import io
import os
import pytest
from dotenv import load_dotenv
from pathlib import Path
from unittest.mock import MagicMock, patch
from minio.error import S3Error
from typing import Any

load_dotenv(Path(__file__).parent.parent.parent / ".env")

@pytest.fixture
def store_config() -> dict[str, str]:
    return {
        "endpoint": os.environ.get("MINIO_ENDPOINT"),
        "access_key": os.environ.get("MINIO_ACCESS_KEY"),
        "secret_key": os.environ.get("MINIO_SECRET_KEY"),
        "bucket_name":  os.environ.get("MINIO_BUCKET_NAME"),
    }

@pytest.fixture
def file_store(store_config: dict[str, str]) -> "FileStore":
    return FileStore(**store_config)

@pytest.fixture
def initialized_store(file_store: "FileStore") -> "FileStore":
    file_store.client = MagicMock()
    return file_store

class TestVectorStore:
    """Test cases for VectorStore."""

    @pytest.fixture
    def vector_store(self) -> VectorStore:
        """Create a VectorStore instance for testing."""
        return VectorStore(url="http://localhost:8080", class_name="TestChunk")

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        """Create sample Chunk instances for testing."""
        return [
            Chunk(
                text="This is the first chunk of text.",
                chunk_index=0,
                start_char=0,
                end_char=30,
                metadata={"page": 1},
            ),
            Chunk(
                text="This is the second chunk of text.",
                chunk_index=1,
                start_char=30,
                end_char=61,
                metadata={"page": 1},
            ),
            Chunk(
                text="This is the third chunk of text.",
                chunk_index=2,
                start_char=61,
                end_char=91,
                metadata={"page": 2},
            ),
        ]

    @pytest.fixture
    def sample_embeddings(self) -> list[list[float]]:
        """Create sample embeddings for testing."""
        return [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768,
        ]

    def test_init(self, vector_store: VectorStore) -> None:
        """Test VectorStore initialization.

        Should:
        - Store URL and class name
        - Initialize client as None
        """
        assert vector_store.url == "http://localhost:8080"
        assert vector_store.class_name == "TestChunk"
        assert vector_store.client is None

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_connect(
        self, mock_connect: Mock, vector_store: VectorStore
    ) -> None:
        """Test Weaviate connection.
        
        Should:
        - Connect to Weaviate server
        - Initialize client
        - Raise ConnectionError on failure
        """
        # Setup mock
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        # Test successful connection
        vector_store.connect()

        mock_connect.assert_called_once()
        assert vector_store.client is mock_client

        # Test connection failure
        mock_client.is_ready.return_value = False
        vector_store.client = None

        with pytest.raises(ConnectionError):
            vector_store.connect()

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_disconnect(
        self, mock_connect: Mock, vector_store: VectorStore
    ) -> None:
        """Test disconnection from Weaviate.

        Should:
        - Close client connection
        - Set client to None
        """
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        vector_store.connect()
        vector_store.disconnect()

        mock_client.close.assert_called_once()
        assert vector_store.client is None

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_create_schema(
        self, mock_connect: Mock, vector_store: VectorStore
    ) -> None:
        """Test schema creation.
        
        Should:
        - Create Weaviate collection
        - Define required properties including doc_id
        """
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_client.collections.exists.return_value = False
        mock_client.collections.create.return_value = None
        mock_connect.return_value = mock_client

        vector_store.connect()
        vector_store.create_schema()

        # Verify collection was created
        mock_client.collections.create.assert_called_once()
        call_args = mock_client.collections.create.call_args
        assert call_args.kwargs["name"] == "TestChunk"

        # Verify properties include doc_id and Chunk model fields
        properties = call_args.kwargs["properties"]
        property_names = [p.name for p in properties]
        assert "doc_id" in property_names
        assert "text" in property_names
        assert "chunk_index" in property_names
        assert "start_char" in property_names
        assert "end_char" in property_names
        assert "metadata" in property_names

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_create_schema_already_exists(
        self, mock_connect: Mock, vector_store: VectorStore
    ) -> None:
        """Test schema creation when collection already exists.

        Should:
        - Not create new collection
        - Log that collection exists
        """
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_client.collections.exists.return_value = True
        mock_connect.return_value = mock_client

        vector_store.connect()
        vector_store.create_schema()

        # Should not call create since collection exists
        mock_client.collections.create.assert_not_called()

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_delete_schema(
        self, mock_connect: Mock, vector_store: VectorStore
    ) -> None:
        """Test schema deletion.

        Should:
        - Delete the collection
        - Remove all data
        """
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_client.collections.exists.return_value = True
        mock_connect.return_value = mock_client

        vector_store.connect()
        vector_store.delete_schema()

        mock_client.collections.delete.assert_called_once_with("TestChunk")

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_insert_chunks(
        self,
        mock_connect: Mock,
        vector_store: VectorStore,
        sample_chunks: list[Chunk],
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test chunk insertion.
        
        Should:
        - Insert chunks with embeddings
        - Return UUIDs for inserted chunks
        - Store doc_id as separate property
        """
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_client.collections.exists.return_value = True

        mock_collection = MagicMock()
        mock_collection.data.insert.return_value = None
        mock_client.collections.get.return_value = mock_collection
        mock_connect.return_value = mock_client

        vector_store.connect()
        uuids = vector_store.insert_chunks(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            document_id="doc123",
        )

        assert len(uuids) == 3
        assert mock_collection.data.insert.call_count == 3

        # Verify first insert call has doc_id property
        first_call = mock_collection.data.insert.call_args_list[0]
        assert "properties" in first_call.kwargs
        assert first_call.kwargs["properties"]["doc_id"] == "doc123"
        assert first_call.kwargs["properties"]["text"] == "This is the first chunk of text."
        assert first_call.kwargs["properties"]["chunk_index"] == 0
        assert first_call.kwargs["properties"]["start_char"] == 0
        assert first_call.kwargs["properties"]["end_char"] == 30

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_insert_chunks_mismatch(
        self,
        mock_connect: Mock,
        vector_store: VectorStore,
        sample_chunks: list[Chunk],
    ) -> None:
        """Test chunk insertion with mismatched lengths.

        Should:
        - Raise ValueError when chunks and embeddings count differ
        """
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        vector_store.connect()

        with pytest.raises(ValueError, match="length mismatch"):
            vector_store.insert_chunks(
                chunks=sample_chunks,
                embeddings=[[0.1] * 768],
                document_id="doc123",
            )

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_search_similar(
        self, mock_connect: Mock, vector_store: VectorStore
    ) -> None:
        """Test similarity search.
        
        Should:
        - Search for similar chunks
        - Return results with similarity scores
        - Return doc_id as separate field
        """
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_client.collections.exists.return_value = True

        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.properties = {
            "doc_id": "doc123",
            "text": "Sample text",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 11,
            "metadata": json.dumps({"page": 1}),
        }
        mock_result.uuid = "test-uuid"
        mock_result.metadata.distance = 0.1

        mock_collection.query.near_vector.return_value = MagicMock(objects=[mock_result])
        mock_client.collections.get.return_value = mock_collection
        mock_connect.return_value = mock_client

        vector_store.connect()
        results = vector_store.search_similar([0.1] * 768, limit=5)

        assert len(results) == 1
        assert results[0]["doc_id"] == "doc123"
        assert results[0]["text"] == "Sample text"
        assert results[0]["chunk_index"] == 0
        assert results[0]["start_char"] == 0
        assert results[0]["end_char"] == 11
        assert results[0]["metadata"]["page"] == 1
        assert results[0]["distance"] == 0.1
        assert results[0]["uuid"] == "test-uuid"

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_search_similar_with_filter(
        self, mock_connect: Mock, vector_store: VectorStore
    ) -> None:
        """Test similarity search with doc_id filter.

        Should:
        - Apply filter for doc_id
        - Return filtered results
        """
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_client.collections.exists.return_value = True

        mock_collection = MagicMock()
        mock_collection.query.near_vector.return_value = MagicMock(objects=[])
        mock_client.collections.get.return_value = mock_collection
        mock_connect.return_value = mock_client

        vector_store.connect()
        results = vector_store.search_similar(
            [0.1] * 768,
            limit=5,
            where_filter={"doc_id": "doc123"},
        )

        # Verify filter was passed
        call_args = mock_collection.query.near_vector.call_args
        assert "filters" in call_args.kwargs

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_delete_by_document_id(
        self, mock_connect: Mock, vector_store: VectorStore
    ) -> None:
        """Test deletion by document ID.
        
        Should:
        - Delete all chunks for document
        - Return count of deleted chunks
        """
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_client.collections.exists.return_value = True

        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.successful = 5
        mock_collection.data.delete_many.return_value = mock_result
        mock_client.collections.get.return_value = mock_collection
        mock_connect.return_value = mock_client

        vector_store.connect()
        deleted_count = vector_store.delete_by_document_id("doc123")

        assert deleted_count == 5
        mock_collection.data.delete_many.assert_called_once()

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_get_chunk_by_id(
        self, mock_connect: Mock, vector_store: VectorStore
    ) -> None:
        """Test chunk retrieval by ID.
        
        Should:
        - Retrieve chunk by UUID
        - Return chunk data if found
        - Return None if not found
        """
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_client.collections.exists.return_value = True

        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.properties = {
            "doc_id": "doc123",
            "text": "Sample text",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 11,
            "metadata": json.dumps({"page": 1}),
        }
        mock_result.uuid = "test-uuid"
        mock_collection.query.fetch_object_by_id.return_value = mock_result
        mock_client.collections.get.return_value = mock_collection
        mock_connect.return_value = mock_client

        vector_store.connect()
        result = vector_store.get_chunk_by_id("test-uuid")

        assert result is not None
        assert result["doc_id"] == "doc123"
        assert result["text"] == "Sample text"
        assert result["metadata"]["page"] == 1

        # Test not found
        mock_collection.query.fetch_object_by_id.return_value = None
        result = vector_store.get_chunk_by_id("nonexistent")
        assert result is None

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_count_chunks(
        self, mock_connect: Mock, vector_store: VectorStore
    ) -> None:
        """Test chunk counting.
        
        Should:
        - Count total chunks
        - Filter by document_id if provided
        """
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_client.collections.exists.return_value = True

        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.total_count = 10
        mock_collection.aggregate.over_all.return_value = mock_result
        mock_client.collections.get.return_value = mock_collection
        mock_connect.return_value = mock_client

        vector_store.connect()

        # Count all
        count = vector_store.count_chunks()
        assert count == 10

        # Count with filter
        count = vector_store.count_chunks(document_id="doc123")
        call_args = mock_collection.aggregate.over_all.call_args
        assert "filters" in call_args.kwargs

    @patch("api.storage.vector_store.weaviate.connect_to_custom")
    def test_batch_insert(
            self,
            mock_connect: Mock,
            vector_store: VectorStore,
            sample_chunks: list[Chunk],
            sample_embeddings: list[list[float]],
    ) -> None:
        """Test batch insertion.

        Should:
        - Insert chunks in batches
        - Return all UUIDs
        - Store doc_id as separate property
        """
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_client.collections.exists.return_value = True

        mock_collection = MagicMock()
        mock_batch = MagicMock()
        mock_batch.__enter__ = MagicMock(return_value=mock_batch)
        mock_batch.__exit__ = MagicMock(return_value=None)
        mock_collection.batch.dynamic.return_value = mock_batch
        mock_client.collections.get.return_value = mock_collection
        mock_connect.return_value = mock_client

        vector_store.connect()
        uuids = vector_store.batch_insert(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            document_id="doc123",
            batch_size=2,
        )

        assert len(uuids) == 3

    def test_not_connected_error(self, vector_store: VectorStore) -> None:
        """Test operations fail when not connected.

        Should:
        - Raise ValueError when client is None
        """
        with pytest.raises(ValueError, match="Not connected"):
            vector_store.create_schema()

        with pytest.raises(ValueError, match="Not connected"):
            vector_store.delete_schema()


class TestFileStore:
    """Test cases for FileStore."""

    @patch("api.storage.file_store.Minio")
    def test_connect(self, mock_minio_class: MagicMock, store_config: dict[str, str]) -> None:
        """Test MinIO connection."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client

        mock_new_bucket = MagicMock()
        mock_new_bucket.name = store_config["bucket_name"]

        mock_client.list_buckets.side_effect = [
            [],
            [],
            [mock_new_bucket]
        ]

        store = FileStore(**store_config)
        store.connect()

        mock_minio_class.assert_called_once_with(
            store_config["endpoint"],
            access_key=store_config["access_key"],
            secret_key=store_config["secret_key"],
            secure=False
        )
        mock_client.make_bucket.assert_called_once_with(store_config["bucket_name"])

    def test_create_bucket(self, initialized_store: "FileStore") -> None:
        """Test bucket creation."""
        mock_bucket = MagicMock()
        mock_bucket.name = "other-bucket"

        mock_new_bucket = MagicMock()
        mock_new_bucket.name = initialized_store.bucket_name

        initialized_store.client.list_buckets.side_effect = [
            [mock_bucket],
            [mock_bucket, mock_new_bucket]
        ]

        initialized_store.create_bucket()
        initialized_store.client.make_bucket.assert_called_once_with(initialized_store.bucket_name)

    def test_upload_file(self, initialized_store: "FileStore") -> None:
        """Test file upload."""
        file_data = io.BytesIO(b"dummy data")
        object_name = "test.txt"

        mock_result = MagicMock()
        mock_result.object_name = object_name
        initialized_store.client.put_object.return_value = mock_result

        result = initialized_store.upload_file(
            file_data=file_data,
            object_name=object_name,
            content_type="text/plain"
        )

        assert result == object_name
        initialized_store.client.put_object.assert_called_once()
        kwargs = initialized_store.client.put_object.call_args.kwargs
        assert "last_modified" in kwargs["metadata"]

    @patch.object(FileStore, 'file_exists', return_value=True)
    def test_download_file(self, mock_exists: MagicMock, initialized_store: "FileStore") -> None:
        """Test file download."""
        mock_result = MagicMock()
        mock_result.data = b"file content"
        initialized_store.client.get_object.return_value = mock_result

        result = initialized_store.download_file("test.txt")

        assert result == b"file content"
        initialized_store.client.get_object.assert_called_once_with(
            initialized_store.bucket_name,
            "test.txt"
        )

    @patch.object(FileStore, 'file_exists', return_value=True)
    def test_delete_file(self, mock_exists: MagicMock, initialized_store: "FileStore") -> None:
        """Test file deletion."""
        initialized_store.delete_file("test.txt")

        initialized_store.client.remove_object.assert_called_once_with(
            initialized_store.bucket_name,
            "test.txt"
        )

    @patch.object(FileStore, 'file_exists', return_value=True)
    def test_get_file_metadata(self, mock_exists: MagicMock, initialized_store: "FileStore") -> None:
        """Test metadata retrieval."""
        mock_result = MagicMock()
        mock_result.metadata = {"custom": "data"}
        initialized_store.client.stat_object.return_value = mock_result

        result = initialized_store.get_file_metadata("test.txt")

        assert result == {"custom": "data"}

    def test_list_files(self, initialized_store: "FileStore") -> None:
        """Test file listing."""
        mock_obj1 = MagicMock()
        mock_obj1.object_name = "file1.txt"
        initialized_store.client.list_objects.return_value = [mock_obj1]

        result = initialized_store.list_files(prefix="file", recursive=True)

        assert result == [mock_obj1]
        initialized_store.client.list_objects.assert_called_once_with(
            initialized_store.bucket_name,
            prefix="file",
            recursive=True
        )

    def test_file_exists(self, initialized_store: "FileStore") -> None:
        """Test file existence check."""
        initialized_store.client.stat_object.return_value = MagicMock()
        assert initialized_store.file_exists("test.txt") is True

        initialized_store.client.stat_object.side_effect = Exception("Not found")
        with pytest.raises(S3Error):
            initialized_store.file_exists("missing.txt")