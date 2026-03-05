"""File store connector for MinIO."""

from typing import Optional, BinaryIO, Dict, Any, List
from datetime import datetime, timedelta
from minio import Minio
from minio.error import S3Error


class FileStore:
    """MinIO client wrapper for file storage operations."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = False
    ) -> None:
        """Initialize MinIO client.
        
        Args:
            endpoint: MinIO server endpoint (host:port)
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: Default bucket name for documents
            secure: Use HTTPS if True
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.secure = secure
        self.client: Optional[Minio] = None

    def connect(self) -> None:
        """Establish connection to MinIO server and create bucket if needed.
        
        Raises:
            ConnectionError: If connection to MinIO fails
            S3Error: If bucket creation fails
        """
        self.client  = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        if self.client is not None :
            buckets_names = [bucket.name for bucket in self.client.list_buckets()]
            if self.bucket_name not in buckets_names : 
                self.create_bucket(self.bucket_name)
        else :
            raise ConnectionError("Failed to connect to MinIO server")

    def create_bucket(self, bucket_name: Optional[str] = None) -> None:
        """Create a bucket if it doesn't exist.
        
        Args:
            bucket_name: Bucket name (uses default if None)
        
        Raises:
            S3Error: If bucket creation fails
        """
        if bucket_name is None:
            bucket_name = self.bucket_name
        buckets_names = [bucket.name for bucket in self.client.list_buckets()]
        if self.bucket_name not in buckets_names : 
            print(f"Creating bucket {self.bucket_name}")
            self.client.make_bucket(self.bucket_name)
        else :
            print(f"Bucket {self.bucket_name} already exists")
        buckets_names = [bucket.name for bucket in self.client.list_buckets()]
        if self.bucket_name not in buckets_names :
            raise S3Error(f"Failed to create bucket {self.bucket_name}", 500, "BucketCreationError", "BucketCreationError", "BucketCreationError", "BucketCreationError")

    def upload_file(
        self,
        file_data: BinaryIO,
        object_name: str,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload a file to MinIO.
        
        Args:
            file_data: File-like object to upload
            object_name: Object name/path in bucket
            content_type: MIME type of the file
            metadata: Optional metadata dictionary
        
        Returns:
            str: Object name/path in bucket
        
        Raises:
            S3Error: If upload fails
        """
        if metadata is None:
            metadata = {}
        metadata["last_modified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = self.client.put_object(
            self.bucket_name,
            object_name,
            file_data,
            content_type=content_type,
            metadata=metadata,
            length= file_data.getbuffer().nbytes
        )
        if result is None:
            raise S3Error(f"Failed to upload file {object_name}", 500, "FileUploadError", "FileUploadError", "FileUploadError", "FileUploadError")
        
        return result.object_name

    def download_file(
        self,
        object_name: str,
        file_path: Optional[str] = None
    ) -> bytes:
        """Download a file from MinIO.
        
        Args:
            object_name: Object name/path in bucket
            file_path: Optional local path to save file
        
        Returns:
            bytes: File content as bytes
        
        Raises:
            S3Error: If download fails or object not found
        """
        if self.file_exists(object_name) == False :
            raise S3Error(f"File {object_name} not found", 404, "FileNotFoundError", "FileNotFoundError", "FileNotFoundError", "FileNotFoundError")
        result = self.client.get_object(
            self.bucket_name,
            object_name
        )
        if result is None:
            raise S3Error(f"Failed to download file {object_name}", 500, "FileDownloadError", "FileDownloadError", "FileDownloadError", "FileDownloadError")
        if file_path is not None:
            with open(file_path, "wb") as file:
                file.write(result.data)
        return result.data

    def delete_file(self, object_name: str) -> None:
        """Delete a file from MinIO.
        
        Args:
            object_name: Object name/path in bucket
        
        Raises:
            S3Error: If deletion fails or object not found
        """
        if self.file_exists(object_name) == False :
            raise S3Error(f"File {object_name} not found", 404, "FileNotFoundError", "FileNotFoundError", "FileNotFoundError", "FileNotFoundError")
        try :
            self.client.remove_object(
                self.bucket_name,
                object_name
            )
        except :
            raise S3Error(f"Failed to delete file {object_name}", 500, "FileDeletionError", "FileDeletionError", "FileDeletionError", "FileDeletionError")

    def get_file_metadata(self, object_name: str) -> Dict[str, Any]:
        """Get file metadata from MinIO.
        
        Args:
            object_name: Object name/path in bucket
        
        Returns:
            Dict[str, Any]: File metadata including:
                - size: File size in bytes
                - content_type: MIME type
                - last_modified: Last modification date
                - metadata: Custom metadata
        
        Raises:
            S3Error: If metadata retrieval fails or object not found
        """
        if self.file_exists(object_name) == False :
            raise S3Error(f"File {object_name} not found", 404, "FileNotFoundError", "FileNotFoundError", "FileNotFoundError", "FileNotFoundError")
        
        result = self.client.stat_object(
            self.bucket_name,
            object_name
        )
        if result is None:
            raise S3Error(f"Failed to get metadata for file {object_name}", 500, "MetadataRetrievalError", "MetadataRetrievalError", "MetadataRetrievalError", "MetadataRetrievalError")
        return result.metadata

    def list_files(
        self,
        prefix: Optional[str] = None,
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """List files in bucket, optionally filtered by prefix.
        
        Args:
            prefix: Optional prefix to filter files
            recursive: List recursively if True
        
        Returns:
            List[Dict[str, Any]]: List of file information dictionaries
        
        Raises:
            S3Error: If listing fails
        """
        result = self.client.list_objects(
            self.bucket_name,
            prefix=prefix,
            recursive=recursive
        )
        if result is None:
            raise S3Error(f"Failed to list files in bucket {self.bucket_name}", 500, "FileListingError", "FileListingError", "FileListingError", "FileListingError")

        return result

    def file_exists(self, object_name: str) -> bool:
        """Check if a file exists in MinIO.
        
        Args:
            object_name: Object name/path in bucket
        
        Returns:
            bool: True if file exists, False otherwise
        
        Raises:
            S3Error: If check fails
        """
        try :
            result = self.client.stat_object(
                self.bucket_name,
                object_name
            )
        except :
            raise S3Error(f"Failed to check if file {object_name} exists", 500, "FileExistsError", "FileExistsError", "FileExistsError", "FileExistsError")
        if result is None:
            return False
        return True

    def get_presigned_url(
        self,
        object_name: str,
        expires_seconds: int = 3600
    ) -> str:
        """Generate a presigned URL for temporary file access.
        
        Args:
            object_name: Object name/path in bucket
            expires_seconds: URL expiration time in seconds
        
        Returns:
            str: Presigned URL
        
        Raises:
            S3Error: If URL generation fails
        """
        if self.file_exists(object_name) == False :
            raise S3Error(f"File {object_name} not found", 404, "FileNotFoundError", "FileNotFoundError", "FileNotFoundError", "FileNotFoundError")
        url = self.client.presigned_get_object(
            self.bucket_name,
            object_name,
            expires=timedelta(seconds=expires_seconds)
        )
        if url is None:
            raise S3Error(f"Failed to generate presigned URL for {object_name}", 500, "PresignedUrlError", "PresignedUrlError", "PresignedUrlError", "PresignedUrlError")
        return url