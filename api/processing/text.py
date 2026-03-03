"""Text processing module for PDF extraction and OCR via Mistral API."""

from typing import Optional, Dict, Any, List
from pathlib import Path
import base64
import io
from mistralai import Mistral
import pdfplumber


class TextProcessor:
    """Text processor for PDF extraction and OCR using Mistral Pixtral."""

    # Image formats accepted by Pixtral for OCR
    SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "gif", "webp"]

    def __init__(self, api_key: str, ocr_model: str = "pixtral-12b-2409") -> None:
        """Initialize the text processor.

        Args:
            api_key: Mistral API key
            ocr_model: Identifier of the Mistral model used for OCR (vision)
        """
        # Store the API key to use it when initializing the client
        self.api_key = api_key

        # Store the OCR model name (Pixtral is Mistral's vision model)
        self.ocr_model = ocr_model

        # The Mistral client is None until initialize() is called.
        # Construction and connection are separated so the key can be validated independently.
        self.client: Optional[Mistral] = None

    def initialize(self) -> None:
        """Create and validate the Mistral client.

        Must be called before any OCR operation.

        Raises:
            ValueError: If the API key is empty or client instantiation fails
        """
        # Ensure the API key is not empty before attempting a connection
        if not self.api_key:
            raise ValueError("Mistral API key is missing.")

        # Instantiate the official Mistral client with the provided API key.
        # This client will be reused for all subsequent OCR calls.
        self.client = Mistral(api_key=self.api_key)

    # -------------------------------------------------------------------------
    # PDF extraction
    # -------------------------------------------------------------------------

    def extract_text_from_pdf(
        self,
        pdf_data: bytes,
    ) -> Dict[str, Any]:
        """Extract text from a PDF given its raw bytes (main method).

        All other PDF extraction methods delegate to this one.

        Args:
            pdf_data: Raw PDF file content

        Returns:
            Dict[str, Any] containing:
                - text (str): Full extracted text (all pages concatenated)
                - pages (List[str]): Per-page text list
                - page_count (int): Total number of pages
                - metadata (Dict): PDF metadata (title, author, etc.)

        Raises:
            ValueError: If extraction fails or the data is not a valid PDF
        """
        try:
            # pdfplumber expects a file-like object (readable stream).
            # We wrap the raw bytes in io.BytesIO to simulate a file
            # without writing anything to disk.
            pdf_stream = io.BytesIO(pdf_data)

            # Open the PDF with pdfplumber. The "with" block ensures the file
            # is properly closed even if an error occurs.
            with pdfplumber.open(pdf_stream) as pdf:

                # Retrieve the metadata embedded in the PDF (title, author, etc.).
                # pdfplumber exposes this via pdf.metadata (dict or None).
                raw_meta = pdf.metadata or {}

                # Extract text from each page individually.
                # page.extract_text() returns a str or None if the page is blank/image.
                # We replace None with "" to guarantee a consistent str type.
                pages = [
                    page.extract_text() or ""  # lazy "or" so None becomes ""
                    for page in pdf.pages
                ]

            # Concatenate all pages into a single text, separated by a newline.
            # strip() removes leading/trailing whitespace and newlines.
            full_text = "\n".join(pages).strip()

            return {
                # Full document text (all pages merged)
                "text": full_text,
                # Per-page text list (useful for page-level chunking)
                "pages": pages,
                # Total number of pages in the PDF
                "page_count": len(pages),
                # Raw PDF metadata (title, author, creation date, etc.)
                "metadata": raw_meta,
            }

        except Exception as e:
            # Catch all pdfplumber errors (corrupted, encrypted PDF, etc.)
            # and re-raise with a readable message.
            raise ValueError(f"PDF extraction failed: {e}") from e

    def extract_text_from_pdf_file(
        self,
        pdf_file_path: str,
    ) -> Dict[str, Any]:
        """Extract text from a PDF on disk (wrapper around extract_text_from_pdf).

        Reads the file then delegates to extract_text_from_pdf().

        Args:
            pdf_file_path: Absolute or relative path to the PDF file

        Returns:
            Dict[str, Any]: Same format as extract_text_from_pdf()

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If extraction fails
        """
        # Convert the string path to a Path object to use cross-platform
        # existence checking via .exists()
        path = Path(pdf_file_path)

        # Verify the file exists before attempting to read it.
        # Raises FileNotFoundError with a clear message instead of a generic error.
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")

        # Read the entire file as bytes.
        # read_bytes() opens in binary mode, which is required for PDF files.
        pdf_data = path.read_bytes()

        # Delegate all processing to the core method.
        # This wrapper has a single responsibility: read the file.
        return self.extract_text_from_pdf(pdf_data)

    def extract_text_by_page(
        self,
        pdf_data: bytes,
        page_numbers: Optional[List[int]] = None,
    ) -> Dict[int, str]:
        """Extract text from specific pages of a PDF.

        Args:
            pdf_data: Raw PDF file content
            page_numbers: List of page numbers to extract (1-indexed).
                          If None, all pages are extracted.

        Returns:
            Dict[int, str]: Dictionary mapping {page_number: extracted_text}

        Raises:
            ValueError: If a page number is out of bounds or extraction fails
        """
        try:
            # Wrap bytes in a stream so pdfplumber can read them without disk I/O
            # (same pattern as in extract_text_from_pdf)
            pdf_stream = io.BytesIO(pdf_data)

            with pdfplumber.open(pdf_stream) as pdf:
                # Total page count, used to validate requested indices
                total_pages = len(pdf.pages)

                # If no pages are specified, extract all pages.
                # Build the full list [1, 2, ..., total_pages] using 1-based indexing
                # to match the user-facing convention (page 1 = first page).
                targets = page_numbers if page_numbers is not None else list(range(1, total_pages + 1))

                result: Dict[int, str] = {}

                for page_num in targets:
                    # Validate that the requested page number is within bounds.
                    # The user provides 1-based numbers (page 1 = first page).
                    if page_num < 1 or page_num > total_pages:
                        raise ValueError(
                            f"Page {page_num} out of range (the PDF has {total_pages} page(s))."
                        )

                    # pdfplumber uses 0-based indexing: page 1 → index 0
                    page = pdf.pages[page_num - 1]

                    # Extract text from the page; returns None if the page is an image.
                    # Replace None with "" to guarantee a str type in all cases.
                    result[page_num] = page.extract_text() or ""

            return result

        except ValueError:
            # Let our own ValueErrors propagate (page out of bounds)
            raise
        except Exception as e:
            # Catch unexpected pdfplumber errors (corrupted, encrypted PDF, etc.)
            raise ValueError(f"Per-page extraction failed: {e}") from e

    def extract_pdf_metadata(
        self,
        pdf_data: bytes,
    ) -> Dict[str, Any]:
        """Extract metadata from a PDF.

        Args:
            pdf_data: Raw PDF file content

        Returns:
            Dict[str, Any] containing:
                - title (str): Document title
                - author (str): Author name
                - subject (str): Document subject
                - creator (str): Creating application
                - creation_date (str): Creation date
                - modification_date (str): Last modification date
                - page_count (int): Number of pages

        Raises:
            ValueError: If metadata extraction fails
        """
        try:
            # Wrap bytes in a readable stream without disk I/O
            pdf_stream = io.BytesIO(pdf_data)

            with pdfplumber.open(pdf_stream) as pdf:
                # pdfplumber exposes built-in PDF metadata via pdf.metadata.
                # Keys follow the PDF standard (Author, Title, etc.).
                # Fall back to an empty dict if the PDF has no metadata,
                # to avoid KeyError on subsequent .get() calls.
                raw = pdf.metadata or {}

                return {
                    # .get(key, "") returns "" if the metadata field is absent,
                    # preventing None values in the output dict.
                    "title":             raw.get("Title", ""),
                    "author":            raw.get("Author", ""),
                    "subject":           raw.get("Subject", ""),
                    # Creator = application that produced the source document (e.g. Word, LaTeX)
                    "creator":           raw.get("Creator", ""),
                    # CreationDate and ModDate are strings in PDF format (e.g. "D:20240101")
                    "creation_date":     raw.get("CreationDate", ""),
                    "modification_date": raw.get("ModDate", ""),
                    # Page count read directly from pdfplumber, not from metadata
                    "page_count":        len(pdf.pages),
                }

        except Exception as e:
            # Catch pdfplumber errors (unreadable, encrypted, or corrupted PDF)
            raise ValueError(f"Metadata extraction failed: {e}") from e

    # -------------------------------------------------------------------------
    # Image OCR
    # -------------------------------------------------------------------------

    def ocr_image(
        self,
        image_data: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """Perform OCR on an image via Mistral Pixtral (main method).

        All other OCR methods delegate to this one.

        Args:
            image_data: Raw image file content
            filename: File name with its extension (e.g. "note.png").
                      Used to determine the MIME type for the base64 data URI.

        Returns:
            Dict[str, Any] containing:
                - text (str): Text extracted from the image
                - confidence (Optional[float]): Confidence score if available

        Raises:
            ValueError: If the format is not supported or OCR fails
        """
        # Ensure initialize() has been called before attempting to use the client.
        # Without a client, any API call would fail with an unclear AttributeError.
        if self.client is None:
            raise ValueError("Mistral client is not initialized. Call initialize() first.")

        # Extract the extension from the filename and normalize it to lowercase.
        # Path(filename).suffix returns ".png"; lstrip(".") removes the dot → "png"
        extension = Path(filename).suffix.lstrip(".").lower()

        # Check that the image format is supported by Pixtral
        if extension not in self.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"Image format '{extension}' is not supported. "
                f"Accepted formats: {self.SUPPORTED_IMAGE_FORMATS}"
            )

        # Mapping from file extension to standard MIME type.
        # Required to build the data URI sent to the API.
        # Both jpg and jpeg map to "image/jpeg" (same format, two extensions).
        mime_types = {
            "png":  "image/png",
            "jpg":  "image/jpeg",
            "jpeg": "image/jpeg",
            "gif":  "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_types[extension]

        # Encode the image bytes as base64 (ASCII string).
        # base64.b64encode() returns bytes; .decode("utf-8") converts them to str.
        # Base64 is the only way to embed binary data inside a JSON payload / URL.
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        # Build the data URI: standard format for embedding an image in a URL.
        # Example: "data:image/png;base64,iVBORw0KGgo..."
        # This is the format the Mistral API expects for inline images in messages.
        data_uri = f"data:{mime_type};base64,{image_b64}"

        try:
            # Call the Mistral API with the Pixtral model (multimodal vision model).
            # We send a "user" message with two parts:
            #   1. The image encoded as a base64 data URI
            #   2. A text instruction asking to extract the visible text
            response = self.client.chat.complete(
                model=self.ocr_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                # "image_url" type tells the API this content is an image
                                "type": "image_url",
                                "image_url": {"url": data_uri},
                            },
                            {
                                # "text" type tells the API this content is a text prompt.
                                # We explicitly ask for raw extracted text only,
                                # with no reformulation or commentary.
                                "type": "text",
                                "text": (
                                    "Extract all visible text from this image. "
                                    "Return only the text, with no commentary or reformulation."
                                ),
                            },
                        ],
                    }
                ],
            )

            # Retrieve the generated text from the response.
            # response.choices is a list of completions; we take the first one ([0]).
            # .message.content holds the text produced by Pixtral.
            # .strip() removes leading/trailing whitespace and newlines.
            extracted_text = response.choices[0].message.content.strip()

            return {
                # Text extracted from the image by Pixtral
                "text": extracted_text,
                # Pixtral does not return a confidence score: always None
                "confidence": None,
            }

        except Exception as e:
            # Catch all network or Mistral API errors
            # and re-raise with a readable message
            raise ValueError(f"OCR failed: {e}") from e

    def ocr_image_file(
        self,
        image_file_path: str,
    ) -> Dict[str, Any]:
        """Perform OCR on an image from disk (wrapper around ocr_image).

        Reads the file then delegates to ocr_image().

        Args:
            image_file_path: Absolute or relative path to the image file

        Returns:
            Dict[str, Any]: Same format as ocr_image()

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the format is not supported or OCR fails
        """
        # Convert the string path to a Path object to access .exists() and .name
        path = Path(image_file_path)

        # Verify the file exists before attempting to read it.
        # Raises FileNotFoundError with a clear message instead of a generic error.
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_file_path}")

        # Read the entire file as bytes.
        # read_bytes() opens in binary mode, which is required for images.
        image_data = path.read_bytes()

        # Delegate to the core method, passing the filename (with extension)
        # so ocr_image() can infer the format and build the correct MIME type.
        return self.ocr_image(image_data, filename=path.name)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def get_supported_image_formats(self) -> list[str]:
        """Return the list of image extensions supported for OCR.

        Returns:
            list[str]: Supported extensions (e.g. ['png', 'jpg', 'jpeg', ...])
        """
        # Return a copy of the class constant so that an external caller
        # cannot accidentally modify the original list (e.g. via .append())
        return list(self.SUPPORTED_IMAGE_FORMATS)

    def validate_pdf_file(self, file_path: str) -> bool:
        """Check whether a PDF file is readable.

        Args:
            file_path: Path to the PDF file

        Returns:
            bool: True if the file is a valid, readable PDF; False otherwise
        """
        try:
            path = Path(file_path)

            # Check that the file physically exists on disk
            if not path.exists():
                return False

            # Check that the extension is .pdf (case-insensitive)
            if path.suffix.lower() != ".pdf":
                return False

            # Attempt to open the PDF with pdfplumber to verify it is readable.
            # A file can have the .pdf extension without being a valid PDF
            # (corrupted, encrypted, or simply renamed). This real check is necessary.
            # We use io.BytesIO to avoid keeping the file handle open after the check.
            with pdfplumber.open(io.BytesIO(path.read_bytes())):
                # If pdfplumber opens without error, the PDF is valid and readable
                pass

            return True

        except Exception:
            # Any exception (FileNotFoundError, PDFSyntaxError, etc.)
            # means the file is not a usable PDF → return False
            return False

    def validate_image_file(self, file_path: str) -> bool:
        """Check whether an image file is in a supported format for OCR.

        Args:
            file_path: Path to the image file

        Returns:
            bool: True if the format is supported and the file is readable; False otherwise
        """
        try:
            path = Path(file_path)

            # Check that the file physically exists on disk
            if not path.exists():
                return False

            # Extract and normalize the extension (e.g. ".PNG" → "png")
            extension = path.suffix.lstrip(".").lower()

            # Check that the extension is in the list of formats supported by Pixtral
            if extension not in self.SUPPORTED_IMAGE_FORMATS:
                return False

            # Check that the file is non-empty.
            # stat().st_size returns the size in bytes; a 0-byte file
            # cannot be a valid image.
            if path.stat().st_size == 0:
                return False

            return True

        except Exception:
            # Any exception (permission error, invalid path, etc.)
            # means the file is not usable → return False
            return False
