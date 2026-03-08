"""Tests for storage modules (vector store and file store)."""

import pytest
from api.storage.vector_store import VectorStore
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

    def test_connect(self) -> None:
        """Test Weaviate connection.
        
        Should:
        - Connect to Weaviate server
        - Initialize client
        """
        pass

    def test_create_schema(self) -> None:
        """Test schema creation.
        
        Should:
        - Create Weaviate schema
        - Define required properties
        """
        pass

    def test_insert_chunks(self) -> None:
        """Test chunk insertion.
        
        Should:
        - Insert chunks with embeddings
        - Return UUIDs for inserted chunks
        - Handle batch insertion
        """
        pass

    def test_search_similar(self) -> None:
        """Test similarity search.
        
        Should:
        - Search for similar chunks
        - Return results with similarity scores
        - Apply filters if provided
        """
        pass

    def test_delete_by_document_id(self) -> None:
        """Test deletion by document ID.
        
        Should:
        - Delete all chunks for document
        - Return count of deleted chunks
        """
        pass

    def test_get_chunk_by_id(self) -> None:
        """Test chunk retrieval by ID.
        
        Should:
        - Retrieve chunk by UUID
        - Return chunk data if found
        - Return None if not found
        """
        pass

    def test_count_chunks(self) -> None:
        """Test chunk counting.
        
        Should:
        - Count total chunks
        - Filter by document_id if provided
        """
        pass


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