# Tests — TextProcessor

## Commandes

Toutes les commandes se lancent depuis le dossier **`api/`**.

### Installer les dépendances de test (première fois)
```bash
uv sync --extra dev
```

### Lancer tous les tests
```bash
uv run pytest tests/test_text/test_text_processor.py -v
```

### Lancer une section spécifique
```bash
# Initialisation uniquement
uv run pytest tests/test_text/test_text_processor.py -v -k "initialize"

# Tests PDF uniquement (0 appel API)
uv run pytest tests/test_text/test_text_processor.py -v -k "pdf"

# Test OCR uniquement (1 appel API)
uv run pytest tests/test_text/test_text_processor.py -v -k "ocr"
```

## Fichiers générés

Après exécution des tests, le fichier suivant est créé automatiquement :

- `benev_extracted.txt` — texte extrait de `benev.pdf` pour vérification manuelle

## Appels API Mistral

| Section | Appels API |
|---|---|
| 1. Initialisation | 0 |
| 2. Extraction PDF | 0 |
| 3. Extraction par page | 0 |
| 4. Métadonnées PDF | 0 |
| 5. Validation | 0 |
| 6. OCR (`test_ocr_image_api`) | **1** |
| **Total** | **1** |

## Résultats

**Tous les tests ont été passés avec succès le 03 mars 2026.**

```
tests/test_text/test_text_processor.py::test_init_client_is_none_before_initialize  PASSED
tests/test_text/test_text_processor.py::test_initialize_creates_client              PASSED
tests/test_text/test_text_processor.py::test_initialize_raises_on_empty_key         PASSED
tests/test_text/test_text_processor.py::test_extract_text_from_pdf_returns_expected_keys  PASSED
tests/test_text/test_text_processor.py::test_extract_text_from_pdf_page_count_matches_pages  PASSED
tests/test_text/test_text_processor.py::test_extract_text_from_pdf_saves_txt        PASSED
tests/test_text/test_text_processor.py::test_extract_text_from_pdf_file_matches_bytes  PASSED
tests/test_text/test_text_processor.py::test_extract_text_from_pdf_file_not_found   PASSED
tests/test_text/test_text_processor.py::test_extract_text_by_page_all_pages         PASSED
tests/test_text/test_text_processor.py::test_extract_text_by_page_single_page       PASSED
tests/test_text/test_text_processor.py::test_extract_text_by_page_out_of_bounds     PASSED
tests/test_text/test_text_processor.py::test_extract_pdf_metadata_keys              PASSED
tests/test_text/test_text_processor.py::test_extract_pdf_metadata_page_count        PASSED
tests/test_text/test_text_processor.py::test_validate_pdf_file_valid                PASSED
tests/test_text/test_text_processor.py::test_validate_pdf_file_not_found            PASSED
tests/test_text/test_text_processor.py::test_validate_image_file_valid_png          PASSED
tests/test_text/test_text_processor.py::test_validate_image_file_unsupported_format PASSED
tests/test_text/test_text_processor.py::test_get_supported_image_formats            PASSED
tests/test_text/test_text_processor.py::test_ocr_image_api                          PASSED
tests/test_text/test_text_processor.py::test_ocr_image_unsupported_format           PASSED
tests/test_text/test_text_processor.py::test_ocr_image_not_initialized              PASSED

21 passed in 1.82s
```
