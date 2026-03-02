"""Text processing module for PDF and OCR."""

from typing import Optional, Dict, Any, List
from pathlib import Path
import io
from mistralai import Mistral
import PyPDF2
import pdfplumber


class TextProcessor:
    """Text processor for PDF extraction and OCR using Mistral API."""

    def __init__(
        self,
        api_key: str,
        ocr_model: str = "pixtral-12b-2409"
    ) -> None:
        """Initialize text processor.
        
        Args:
            api_key: Mistral API key
            ocr_model: OCR model name to use
        """
        self.api_key = api_key
        self.ocr_model = ocr_model
        self.client: Optional[Mistral] = None

    def initialize(self) -> None:
        """Initialize Mistral client.
        
        Raises:
            ValueError: If API key is invalid or client initialization fails
        """
        pass

    def extract_text_from_pdf(
        self,
        pdf_file_path: str,
        method: str = "pdfplumber"
    ) -> Dict[str, Any]:
        """Extract text from PDF file.
        
        Args:
            pdf_file_path: Path to PDF file
            method: Extraction method ('pdfplumber' or 'pypdf2')
        
        Returns:
            Dict[str, Any]: Extraction result with:
                - text: Extracted text content
                - pages: List of page texts
                - page_count: Total number of pages
                - metadata: PDF metadata (title, author, etc.)
        
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If extraction fails or unsupported format
        """
        pass

    def extract_text_from_pdf_bytes(
        self,
        pdf_data: bytes,
        method: str = "pdfplumber"
    ) -> Dict[str, Any]:
        """Extract text from PDF bytes.
        
        Args:
            pdf_data: PDF file content as bytes
            method: Extraction method ('pdfplumber' or 'pypdf2')
        
        Returns:
            Dict[str, Any]: Extraction result (same format as extract_text_from_pdf)
        
        Raises:
            ValueError: If extraction fails
        """
        pass

    def extract_text_from_pdf_stream(
        self,
        pdf_stream: io.BytesIO,
        method: str = "pdfplumber"
    ) -> Dict[str, Any]:
        """Extract text from PDF stream.
        
        Args:
            pdf_stream: PDF file stream
            method: Extraction method
        
        Returns:
            Dict[str, Any]: Extraction result (same format as extract_text_from_pdf)
        
        Raises:
            ValueError: If extraction fails
        """
        pass

    def ocr_image(
        self,
        image_file_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform OCR on image file using Mistral API.
        
        Args:
            image_file_path: Path to image file
            language: Optional language code (e.g., 'fr', 'en')
        
        Returns:
            Dict[str, Any]: OCR result with:
                - text: Extracted text
                - confidence: Confidence score if available
                - language: Detected language
                - bounding_boxes: Optional text bounding box coordinates
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If OCR fails or unsupported format
        """
        pass

    def ocr_image_bytes(
        self,
        image_data: bytes,
        file_extension: str = "png",
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform OCR on image bytes using Mistral API.
        
        Args:
            image_data: Image file content as bytes
            file_extension: File extension/format (png, jpg, etc.)
            language: Optional language code
        
        Returns:
            Dict[str, Any]: OCR result (same format as ocr_image)
        
        Raises:
            ValueError: If OCR fails or unsupported format
        """
        pass

    def ocr_image_stream(
        self,
        image_stream: io.BytesIO,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform OCR on image stream using Mistral API.
        
        Args:
            image_stream: Image file stream
            language: Optional language code
        
        Returns:
            Dict[str, Any]: OCR result (same format as ocr_image)
        
        Raises:
            ValueError: If OCR fails
        """
        pass

    def get_supported_image_formats(self) -> list[str]:
        """Get list of supported image formats for OCR.
        
        Returns:
            list[str]: List of supported file extensions (e.g., ['png', 'jpg', 'jpeg'])
        """
        pass

    def validate_pdf_file(self, file_path: str) -> bool:
        """Validate if PDF file is readable.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            bool: True if file is valid, False otherwise
        """
        pass

    def validate_image_file(self, file_path: str) -> bool:
        """Validate if image file is supported for OCR.
        
        Args:
            file_path: Path to image file
        
        Returns:
            bool: True if file is valid, False otherwise
        """
        pass

    def extract_pdf_metadata(self, pdf_file_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF file.
        
        Args:
            pdf_file_path: Path to PDF file
        
        Returns:
            Dict[str, Any]: PDF metadata including:
                - title: Document title
                - author: Author name
                - subject: Document subject
                - creator: Creating application
                - producer: PDF producer
                - creation_date: Creation date
                - modification_date: Modification date
                - page_count: Number of pages
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If metadata extraction fails
        """
        pass

    def extract_text_by_page(
        self,
        pdf_file_path: str,
        page_numbers: Optional[List[int]] = None
    ) -> Dict[int, str]:
        """Extract text from specific pages of PDF.
        
        Args:
            pdf_file_path: Path to PDF file
            page_numbers: Optional list of page numbers (1-indexed)
                If None, extracts all pages
        
        Returns:
            Dict[int, str]: Dictionary mapping page numbers to text content
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If extraction fails
        """
        pass
