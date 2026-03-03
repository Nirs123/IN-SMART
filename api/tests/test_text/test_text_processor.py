"""
pytest tests for TextProcessor.

Run from the api/ directory:
    uv run pytest tests/test_text/test_text_processor.py -v

Mistral API calls made: 1 (only test_ocr_image_api)
All PDF tests use pdfplumber locally, no network required.
"""

import base64
import pytest
from pathlib import Path
from processing.text import TextProcessor

# ── Mistral API key ────────────────────────────────────────────────────────────
API_KEY = "votre_clé_api_mistral_ici"  # Replace with your actual API key

# ── Paths relative to this test file's directory ──────────────────────────────
# __file__ = tests/test_text/test_text_processor.py
# .parent   = tests/test_text/
TEST_DIR = Path(__file__).parent
PDF_PATH = TEST_DIR / "benev.pdf"

# ── Output .txt file generated automatically for manual verification ──────────
OUTPUT_TXT = TEST_DIR / "benev_extracted.txt"

# ── Minimal PNG image (1×1 white pixel) encoded in base64 ─────────────────────
# Allows testing OCR without requiring an image file on disk.
MINIMAL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhf"
    "DwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def processor():
    """Create and initialize a TextProcessor shared across all tests in this module.

    scope="module": the object is created once for the entire test file,
    avoiding re-creating the Mistral client for every individual test.
    """
    p = TextProcessor(api_key=API_KEY)
    p.initialize()
    return p


@pytest.fixture(scope="module")
def pdf_bytes():
    """Read benev.pdf as bytes once for the entire module."""
    return PDF_PATH.read_bytes()


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Initialization (0 API calls)
# ─────────────────────────────────────────────────────────────────────────────

def test_init_client_is_none_before_initialize():
    """The client must be None until initialize() has been called."""
    p = TextProcessor(api_key=API_KEY)
    assert p.client is None


def test_initialize_creates_client():
    """initialize() must instantiate the Mistral client."""
    p = TextProcessor(api_key=API_KEY)
    p.initialize()
    assert p.client is not None


def test_initialize_raises_on_empty_key():
    """initialize() must raise ValueError if the API key is empty."""
    p = TextProcessor(api_key="")
    with pytest.raises(ValueError):
        p.initialize()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Full PDF extraction (0 API calls)
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_text_from_pdf_returns_expected_keys(processor, pdf_bytes):
    """extract_text_from_pdf must return the keys text, pages, page_count, metadata."""
    result = processor.extract_text_from_pdf(pdf_bytes)
    assert isinstance(result["text"], str)
    assert isinstance(result["pages"], list)
    assert isinstance(result["page_count"], int)
    assert isinstance(result["metadata"], dict)


def test_extract_text_from_pdf_page_count_matches_pages(processor, pdf_bytes):
    """The number of items in 'pages' must match 'page_count'."""
    result = processor.extract_text_from_pdf(pdf_bytes)
    assert result["page_count"] > 0
    assert len(result["pages"]) == result["page_count"]


def test_extract_text_from_pdf_saves_txt(processor, pdf_bytes):
    """Extract text from benev.pdf and save it to benev_extracted.txt.

    This file allows manual verification of the extracted content.
    It is written to the same directory as the PDF: tests/test_text/
    """
    result = processor.extract_text_from_pdf(pdf_bytes)

    # Build an informational header before the extracted text
    header = (
        f"Source file: {PDF_PATH.name}\n"
        f"Pages extracted: {result['page_count']}\n"
        f"Total characters: {len(result['text'])}\n"
        f"{'─' * 60}\n\n"
    )

    # Write the header followed by the full text to the output file.
    # encoding="utf-8" ensures accented characters and special symbols are preserved.
    OUTPUT_TXT.write_text(header + result["text"], encoding="utf-8")

    # Verify the file was created and is not empty
    assert OUTPUT_TXT.exists()
    assert OUTPUT_TXT.stat().st_size > 0


def test_extract_text_from_pdf_file_matches_bytes(processor, pdf_bytes):
    """The file wrapper must return the same text as the core method."""
    result_bytes = processor.extract_text_from_pdf(pdf_bytes)
    result_file = processor.extract_text_from_pdf_file(str(PDF_PATH))
    assert result_file["text"] == result_bytes["text"]


def test_extract_text_from_pdf_file_not_found(processor):
    """extract_text_from_pdf_file must raise FileNotFoundError if the file does not exist."""
    with pytest.raises(FileNotFoundError):
        processor.extract_text_from_pdf_file("nonexistent_file.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Per-page extraction (0 API calls)
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_text_by_page_all_pages(processor, pdf_bytes):
    """Without page_numbers, all pages must be returned."""
    result = processor.extract_text_from_pdf(pdf_bytes)
    pages = processor.extract_text_by_page(pdf_bytes)
    assert len(pages) == result["page_count"]
    # All keys are int, all values are str
    assert all(isinstance(k, int) for k in pages)
    assert all(isinstance(v, str) for v in pages.values())


def test_extract_text_by_page_single_page(processor, pdf_bytes):
    """extract_text_by_page([1]) must return only page 1."""
    all_pages = processor.extract_text_by_page(pdf_bytes)
    page1 = processor.extract_text_by_page(pdf_bytes, page_numbers=[1])
    assert 1 in page1
    assert page1[1] == all_pages[1]


def test_extract_text_by_page_out_of_bounds(processor, pdf_bytes):
    """A page number out of range must raise ValueError."""
    with pytest.raises(ValueError):
        processor.extract_text_by_page(pdf_bytes, page_numbers=[9999])


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — PDF metadata (0 API calls)
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_pdf_metadata_keys(processor, pdf_bytes):
    """The metadata dict must contain exactly the expected keys."""
    meta = processor.extract_pdf_metadata(pdf_bytes)
    expected = {"title", "author", "subject", "creator", "creation_date", "modification_date", "page_count"}
    assert set(meta.keys()) == expected


def test_extract_pdf_metadata_page_count(processor, pdf_bytes):
    """page_count in the metadata must be > 0."""
    meta = processor.extract_pdf_metadata(pdf_bytes)
    assert isinstance(meta["page_count"], int)
    assert meta["page_count"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Validation (0 API calls)
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_pdf_file_valid(processor):
    """validate_pdf_file must return True for benev.pdf."""
    assert processor.validate_pdf_file(str(PDF_PATH)) is True


def test_validate_pdf_file_not_found(processor):
    """validate_pdf_file must return False if the file does not exist."""
    assert processor.validate_pdf_file("nonexistent.pdf") is False


def test_validate_image_file_valid_png(processor, tmp_path):
    """validate_image_file must return True for a valid PNG.

    tmp_path is a built-in pytest fixture that provides a temporary directory
    automatically cleaned up after the test.
    """
    png_file = tmp_path / "test.png"
    png_file.write_bytes(MINIMAL_PNG_BYTES)
    assert processor.validate_image_file(str(png_file)) is True


def test_validate_image_file_unsupported_format(processor):
    """validate_image_file must return False for an unsupported format."""
    assert processor.validate_image_file("image.bmp") is False


def test_get_supported_image_formats(processor):
    """get_supported_image_formats must return a list containing at least png and jpg."""
    formats = processor.get_supported_image_formats()
    assert isinstance(formats, list)
    assert "png" in formats
    assert "jpg" in formats


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — OCR (1 API call)
# ─────────────────────────────────────────────────────────────────────────────

def test_ocr_image_api(processor):
    """ocr_image must return a dict with 'text' and 'confidence' after the API call.

    [1 Mistral API call] — 1×1 white pixel PNG image, no text content.
    We only verify that the API responds with the correct format.
    """
    result = processor.ocr_image(MINIMAL_PNG_BYTES, filename="test.png")
    assert isinstance(result, dict)
    assert isinstance(result["text"], str)
    assert result["confidence"] is None  # Pixtral does not return a confidence score


def test_ocr_image_unsupported_format(processor):
    """ocr_image must raise ValueError for an unsupported format."""
    with pytest.raises(ValueError):
        processor.ocr_image(b"data", filename="image.bmp")


def test_ocr_image_not_initialized():
    """ocr_image must raise ValueError if initialize() has not been called."""
    p = TextProcessor(api_key=API_KEY)  # no initialize() call
    with pytest.raises(ValueError):
        p.ocr_image(MINIMAL_PNG_BYTES, filename="test.png")
