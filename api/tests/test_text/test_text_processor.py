"""
Tests pytest pour TextProcessor.

Lancer depuis le dossier api/ :
    uv run pytest tests/test_text/test_text_processor.py -v

Appels API Mistral effectués : 1 (uniquement test_ocr_image_api)
Tous les tests PDF utilisent pdfplumber en local, sans réseau.
"""

import base64
import pytest
from pathlib import Path
from processing.text import TextProcessor

# ── Clé API Mistral ────────────────────────────────────────────────────────────
API_KEY = "e52FrUF2mPF1z68U7ob3ocZC0Kt1MZ5W"

# ── Chemins relatifs au dossier du fichier de test ────────────────────────────
# __file__ = tests/test_text/test_text_processor.py
# .parent   = tests/test_text/
TEST_DIR = Path(__file__).parent
PDF_PATH = TEST_DIR / "benev.pdf"

# ── Fichier .txt généré automatiquement pour vérification manuelle ────────────
OUTPUT_TXT = TEST_DIR / "benev_extracted.txt"

# ── Image PNG minimale (1×1 pixel blanc) encodée en base64 ────────────────────
# Permet de tester l'OCR sans fichier image sur le disque.
MINIMAL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhf"
    "DwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def processor():
    """Crée et initialise un TextProcessor réutilisé par tous les tests du module.

    scope="module" : l'objet est créé une seule fois pour tout le fichier de test,
    ce qui évite de recréer le client Mistral à chaque test.
    """
    p = TextProcessor(api_key=API_KEY)
    p.initialize()
    return p


@pytest.fixture(scope="module")
def pdf_bytes():
    """Lit benev.pdf en bytes une seule fois pour tout le module."""
    return PDF_PATH.read_bytes()


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Initialisation (0 appel API)
# ─────────────────────────────────────────────────────────────────────────────

def test_init_client_is_none_before_initialize():
    """Le client doit être None tant que initialize() n'a pas été appelé."""
    p = TextProcessor(api_key=API_KEY)
    assert p.client is None


def test_initialize_creates_client():
    """initialize() doit instancier le client Mistral."""
    p = TextProcessor(api_key=API_KEY)
    p.initialize()
    assert p.client is not None


def test_initialize_raises_on_empty_key():
    """initialize() doit lever ValueError si la clé API est vide."""
    p = TextProcessor(api_key="")
    with pytest.raises(ValueError):
        p.initialize()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Extraction PDF complète (0 appel API)
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_text_from_pdf_returns_expected_keys(processor, pdf_bytes):
    """extract_text_from_pdf doit retourner les clés text, pages, page_count, metadata."""
    result = processor.extract_text_from_pdf(pdf_bytes)
    assert isinstance(result["text"], str)
    assert isinstance(result["pages"], list)
    assert isinstance(result["page_count"], int)
    assert isinstance(result["metadata"], dict)


def test_extract_text_from_pdf_page_count_matches_pages(processor, pdf_bytes):
    """Le nombre de pages dans 'pages' doit correspondre à 'page_count'."""
    result = processor.extract_text_from_pdf(pdf_bytes)
    assert result["page_count"] > 0
    assert len(result["pages"]) == result["page_count"]


def test_extract_text_from_pdf_saves_txt(processor, pdf_bytes):
    """Extrait le texte de benev.pdf et le sauvegarde dans benev_extracted.txt.

    Ce fichier permet une vérification manuelle du contenu extrait.
    Il est écrit dans le même dossier que le PDF : tests/test_text/
    """
    result = processor.extract_text_from_pdf(pdf_bytes)

    # Construit un en-tête informatif avant le texte extrait
    header = (
        f"Fichier source : {PDF_PATH.name}\n"
        f"Pages extraites : {result['page_count']}\n"
        f"Caractères totaux : {len(result['text'])}\n"
        f"{'─' * 60}\n\n"
    )

    # Écrit l'en-tête suivi du texte complet dans le fichier de sortie.
    # encoding="utf-8" pour supporter les accents et caractères spéciaux.
    OUTPUT_TXT.write_text(header + result["text"], encoding="utf-8")

    # Vérifie que le fichier a bien été créé et n'est pas vide
    assert OUTPUT_TXT.exists()
    assert OUTPUT_TXT.stat().st_size > 0


def test_extract_text_from_pdf_file_matches_bytes(processor, pdf_bytes):
    """Le wrapper fichier doit retourner le même texte que la méthode core."""
    result_bytes = processor.extract_text_from_pdf(pdf_bytes)
    result_file = processor.extract_text_from_pdf_file(str(PDF_PATH))
    assert result_file["text"] == result_bytes["text"]


def test_extract_text_from_pdf_file_not_found(processor):
    """extract_text_from_pdf_file doit lever FileNotFoundError si le fichier n'existe pas."""
    with pytest.raises(FileNotFoundError):
        processor.extract_text_from_pdf_file("fichier_inexistant.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Extraction par page (0 appel API)
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_text_by_page_all_pages(processor, pdf_bytes):
    """Sans page_numbers, toutes les pages doivent être retournées."""
    result = processor.extract_text_from_pdf(pdf_bytes)
    pages = processor.extract_text_by_page(pdf_bytes)
    assert len(pages) == result["page_count"]
    assert all(isinstance(k, int) for k in pages)
    assert all(isinstance(v, str) for v in pages.values())


def test_extract_text_by_page_single_page(processor, pdf_bytes):
    """extract_text_by_page([1]) doit retourner uniquement la page 1."""
    all_pages = processor.extract_text_by_page(pdf_bytes)
    page1 = processor.extract_text_by_page(pdf_bytes, page_numbers=[1])
    assert 1 in page1
    assert page1[1] == all_pages[1]


def test_extract_text_by_page_out_of_bounds(processor, pdf_bytes):
    """Un numéro de page hors limites doit lever ValueError."""
    with pytest.raises(ValueError):
        processor.extract_text_by_page(pdf_bytes, page_numbers=[9999])


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Métadonnées PDF (0 appel API)
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_pdf_metadata_keys(processor, pdf_bytes):
    """Les métadonnées doivent contenir exactement les clés attendues."""
    meta = processor.extract_pdf_metadata(pdf_bytes)
    expected = {"title", "author", "subject", "creator", "creation_date", "modification_date", "page_count"}
    assert set(meta.keys()) == expected


def test_extract_pdf_metadata_page_count(processor, pdf_bytes):
    """page_count dans les métadonnées doit être > 0."""
    meta = processor.extract_pdf_metadata(pdf_bytes)
    assert isinstance(meta["page_count"], int)
    assert meta["page_count"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Validation (0 appel API)
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_pdf_file_valid(processor):
    """validate_pdf_file doit retourner True pour benev.pdf."""
    assert processor.validate_pdf_file(str(PDF_PATH)) is True


def test_validate_pdf_file_not_found(processor):
    """validate_pdf_file doit retourner False si le fichier n'existe pas."""
    assert processor.validate_pdf_file("inexistant.pdf") is False


def test_validate_image_file_valid_png(processor, tmp_path):
    """validate_image_file doit retourner True pour un PNG valide.

    tmp_path est un fixture pytest qui fournit un dossier temporaire
    automatiquement nettoyé après le test.
    """
    png_file = tmp_path / "test.png"
    png_file.write_bytes(MINIMAL_PNG_BYTES)
    assert processor.validate_image_file(str(png_file)) is True


def test_validate_image_file_unsupported_format(processor):
    """validate_image_file doit retourner False pour un format non supporté."""
    assert processor.validate_image_file("image.bmp") is False


def test_get_supported_image_formats(processor):
    """get_supported_image_formats doit retourner une liste contenant au moins png et jpg."""
    formats = processor.get_supported_image_formats()
    assert isinstance(formats, list)
    assert "png" in formats
    assert "jpg" in formats


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — OCR (1 seul appel API)
# ─────────────────────────────────────────────────────────────────────────────

def test_ocr_image_api(processor):
    """ocr_image doit retourner un dict avec 'text' et 'confidence' après appel API.

    [1 appel API Mistral] — image PNG 1x1 pixel blanc, sans texte.
    On vérifie uniquement que l'API répond avec le bon format.
    """
    result = processor.ocr_image(MINIMAL_PNG_BYTES, filename="test.png")
    assert isinstance(result, dict)
    assert isinstance(result["text"], str)
    assert result["confidence"] is None  # Pixtral ne retourne pas de score


def test_ocr_image_unsupported_format(processor):
    """ocr_image doit lever ValueError pour un format non supporté."""
    with pytest.raises(ValueError):
        processor.ocr_image(b"data", filename="image.bmp")


def test_ocr_image_not_initialized():
    """ocr_image doit lever ValueError si initialize() n'a pas été appelé."""
    p = TextProcessor(api_key=API_KEY)  # pas d'initialize()
    with pytest.raises(ValueError):
        p.ocr_image(MINIMAL_PNG_BYTES, filename="test.png")
