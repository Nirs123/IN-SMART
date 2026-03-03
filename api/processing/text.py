"""Text processing module for PDF extraction and OCR via Mistral API."""

from typing import Optional, Dict, Any, List
from pathlib import Path
import base64
import io
from mistralai import Mistral
import pdfplumber


class TextProcessor:
    """Text processor for PDF extraction and OCR using Mistral Pixtral."""

    # Formats image acceptés par Pixtral pour l'OCR
    SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "gif", "webp"]

    def __init__(self, api_key: str, ocr_model: str = "pixtral-12b-2409") -> None:
        """Initialise le processeur de texte.

        Args:
            api_key: Clé API Mistral
            ocr_model: Identifiant du modèle Mistral utilisé pour l'OCR (vision)
        """
        # Stocke la clé API pour l'utiliser lors de l'initialisation du client
        self.api_key = api_key

        # Stocke le nom du modèle OCR (Pixtral est le modèle vision de Mistral)
        self.ocr_model = ocr_model

        # Le client Mistral est None jusqu'à ce que initialize() soit appelé.
        # On sépare construction et connexion pour pouvoir valider la clé séparément.
        self.client: Optional[Mistral] = None

    def initialize(self) -> None:
        """Crée et valide le client Mistral.

        Doit être appelé avant toute opération d'OCR.

        Raises:
            ValueError: Si la clé API est vide ou si l'instanciation du client échoue
        """
        # Vérifie que la clé API n'est pas vide avant de tenter une connexion
        if not self.api_key:
            raise ValueError("La clé API Mistral est manquante.")

        # Instancie le client officiel Mistral avec la clé API fournie.
        # Ce client sera réutilisé pour tous les appels OCR suivants.
        self.client = Mistral(api_key=self.api_key)

    # -------------------------------------------------------------------------
    # Extraction PDF
    # -------------------------------------------------------------------------

    def extract_text_from_pdf(
        self,
        pdf_data: bytes,
    ) -> Dict[str, Any]:
        """Extrait le texte d'un PDF depuis ses bytes bruts (méthode principale).

        Toutes les autres méthodes d'extraction PDF délèguent ici.

        Args:
            pdf_data: Contenu brut du fichier PDF

        Returns:
            Dict[str, Any] contenant :
                - text (str): Texte complet extrait (toutes pages concaténées)
                - pages (List[str]): Liste du texte par page
                - page_count (int): Nombre total de pages
                - metadata (Dict): Métadonnées du PDF (titre, auteur, etc.)

        Raises:
            ValueError: Si l'extraction échoue ou si le format n'est pas un PDF valide
        """
        try:
            # pdfplumber attend un objet file-like (flux lisible).
            # On encapsule les bytes bruts dans un io.BytesIO pour simuler un fichier
            # sans avoir à écrire quoi que ce soit sur le disque.
            pdf_stream = io.BytesIO(pdf_data)

            # Ouvre le PDF via pdfplumber. Le bloc "with" garantit que le fichier
            # sera fermé proprement même en cas d'erreur.
            with pdfplumber.open(pdf_stream) as pdf:

                # Récupère les métadonnées intégrées au PDF (titre, auteur, etc.).
                # pdfplumber expose ces infos via pdf.metadata (dict ou None).
                raw_meta = pdf.metadata or {}

                # Extrait le texte de chaque page individuellement.
                # page.extract_text() retourne une str ou None si la page est vide/image.
                # On remplace None par "" pour garantir un type str cohérent.
                pages = [
                    page.extract_text() or "" # ou paresseux odnc bonne comparaison
                    for page in pdf.pages
                ]

            # Concatène toutes les pages en un seul texte, séparées par un saut de ligne.
            # strip() supprime les espaces/sauts de ligne superflus en début et fin.
            full_text = "\n".join(pages).strip()

            return {
                # Texte complet du document (toutes pages fusionnées)
                "text": full_text,
                # Liste du texte page par page (utile pour le chunking par page)
                "pages": pages,
                # Nombre total de pages du PDF
                "page_count": len(pages),
                # Métadonnées brutes du PDF (titre, auteur, date de création, etc.)
                "metadata": raw_meta,
            }

        except Exception as e:
            # On capture toutes les erreurs pdfplumber (PDF corrompu, chiffré, etc.)
            # et on les reformule avec un message lisible.
            raise ValueError(f"Échec de l'extraction PDF : {e}") from e

    def extract_text_from_pdf_file(
        self,
        pdf_file_path: str,
    ) -> Dict[str, Any]:
        """Extrait le texte d'un PDF depuis le disque (wrapper de extract_text_from_pdf).

        Lit le fichier puis délègue à extract_text_from_pdf().

        Args:
            pdf_file_path: Chemin absolu ou relatif vers le fichier PDF

        Returns:
            Dict[str, Any]: Même format que extract_text_from_pdf()

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si l'extraction échoue
        """
        # Convertit le chemin string en objet Path pour utiliser les méthodes
        # de vérification d'existence (.exists()) de façon cross-platform.
        path = Path(pdf_file_path)

        # Vérifie que le fichier existe bien avant de tenter de le lire.
        # Lève FileNotFoundError avec un message clair plutôt qu'une erreur générique.
        if not path.exists():
            raise FileNotFoundError(f"Fichier PDF introuvable : {pdf_file_path}")

        # Lit l'intégralité du fichier en bytes.
        # "rb" = read binary, obligatoire pour les fichiers PDF.
        pdf_data = path.read_bytes()

        # Délègue tout le traitement à la méthode core.
        # Ce wrapper n'a qu'une seule responsabilité : lire le fichier.
        return self.extract_text_from_pdf(pdf_data)

    def extract_text_by_page(
        self,
        pdf_data: bytes,
        page_numbers: Optional[List[int]] = None,
    ) -> Dict[int, str]:
        """Extrait le texte de pages spécifiques d'un PDF.

        Args:
            pdf_data: Contenu brut du fichier PDF
            page_numbers: Liste de numéros de pages à extraire (indexés à partir de 1).
                          Si None, toutes les pages sont extraites.

        Returns:
            Dict[int, str]: Dictionnaire {numéro_de_page: texte_extrait}

        Raises:
            ValueError: Si un numéro de page est hors limites ou l'extraction échoue
        """
        try:
            # Encapsule les bytes dans un flux pour que pdfplumber puisse les lire
            # sans écriture disque (même principe que dans extract_text_from_pdf)
            pdf_stream = io.BytesIO(pdf_data)

            with pdfplumber.open(pdf_stream) as pdf:
                # Nombre total de pages du document, utilisé pour valider les indices
                total_pages = len(pdf.pages)

                # Si aucune page n'est spécifiée, on extrait toutes les pages.
                # On génère la liste complète [1, 2, ..., total_pages] (indexation 1-based)
                # pour rester cohérent avec la convention utilisateur (page 1 = première page)
                targets = page_numbers if page_numbers is not None else list(range(1, total_pages + 1))

                result: Dict[int, str] = {}

                for page_num in targets:
                    # Vérifie que le numéro de page demandé est dans les bornes valides.
                    # L'utilisateur donne des numéros 1-based (page 1 = première page).
                    if page_num < 1 or page_num > total_pages:
                        raise ValueError(
                            f"Page {page_num} hors limites (le PDF a {total_pages} page(s))."
                        )

                    # pdfplumber indexe les pages en 0-based : page 1 → index 0
                    page = pdf.pages[page_num - 1]

                    # Extrait le texte de la page ; retourne None si la page est une image.
                    # On remplace None par "" pour garantir un type str dans tous les cas.
                    result[page_num] = page.extract_text() or ""

            return result

        except ValueError:
            # Laisse remonter les ValueError qu'on a levées nous-mêmes (page hors limites)
            raise
        except Exception as e:
            # Capture les erreurs inattendues de pdfplumber (PDF corrompu, chiffré, etc.)
            raise ValueError(f"Échec de l'extraction par page : {e}") from e

    def extract_pdf_metadata(
        self,
        pdf_data: bytes,
    ) -> Dict[str, Any]:
        """Extrait les métadonnées d'un PDF.

        Args:
            pdf_data: Contenu brut du fichier PDF

        Returns:
            Dict[str, Any] contenant :
                - title (str): Titre du document
                - author (str): Auteur
                - subject (str): Sujet
                - creator (str): Application créatrice
                - creation_date (str): Date de création
                - modification_date (str): Date de modification
                - page_count (int): Nombre de pages

        Raises:
            ValueError: Si l'extraction des métadonnées échoue
        """
        try:
            # Encapsule les bytes dans un flux lisible sans écriture disque
            pdf_stream = io.BytesIO(pdf_data)

            with pdfplumber.open(pdf_stream) as pdf:
                # pdfplumber expose les métadonnées intégrées au PDF via pdf.metadata.
                # C'est un dict dont les clés suivent le standard PDF (Author, Title, etc.).
                # Si le PDF n'a pas de métadonnées, on utilise un dict vide pour éviter les KeyError.
                raw = pdf.metadata or {}

                return {
                    # .get(clé, "") retourne "" si la métadonnée est absente,
                    # ce qui évite des None dans le résultat final.
                    "title":             raw.get("Title", ""),
                    "author":            raw.get("Author", ""),
                    "subject":           raw.get("Subject", ""),
                    # Creator = application qui a créé le document source (ex: Word, LaTeX)
                    "creator":           raw.get("Creator", ""),
                    # CreationDate et ModDate sont des chaînes au format PDF (ex: "D:20240101")
                    "creation_date":     raw.get("CreationDate", ""),
                    "modification_date": raw.get("ModDate", ""),
                    # Nombre de pages compté directement depuis pdfplumber
                    "page_count":        len(pdf.pages),
                }

        except Exception as e:
            # Capture les erreurs pdfplumber (PDF illisible, chiffré, corrompu)
            raise ValueError(f"Échec de l'extraction des métadonnées : {e}") from e

    # -------------------------------------------------------------------------
    # OCR image
    # -------------------------------------------------------------------------

    def ocr_image(
        self,
        image_data: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """Effectue l'OCR sur une image via Mistral Pixtral (méthode principale).

        Toutes les autres méthodes OCR délèguent ici.

        Args:
            image_data: Contenu brut de l'image
            filename: Nom du fichier avec son extension (ex: "note.png").
                      Utilisé pour encoder correctement l'image en base64 data URI.

        Returns:
            Dict[str, Any] contenant :
                - text (str): Texte extrait de l'image
                - confidence (Optional[float]): Score de confiance si disponible

        Raises:
            ValueError: Si le format n'est pas supporté ou si l'OCR échoue
        """
        # Vérifie que initialize() a bien été appelé avant d'essayer d'utiliser le client.
        # Sans client, tout appel API échouerait avec une AttributeError peu claire.
        if self.client is None:
            raise ValueError("Le client Mistral n'est pas initialisé. Appelez initialize() d'abord.")

        # Extrait l'extension depuis le nom de fichier et la normalise en minuscules.
        # Path(filename).suffix retourne ".png", on retire le point avec [1:] → "png"
        extension = Path(filename).suffix.lstrip(".").lower()

        # Vérifie que le format de l'image est supporté par Pixtral
        if extension not in self.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"Format image '{extension}' non supporté. "
                f"Formats acceptés : {self.SUPPORTED_IMAGE_FORMATS}"
            )

        # Table de correspondance extension → type MIME standard.
        # Le type MIME est requis pour construire le data URI envoyé à l'API.
        # jpg et jpeg pointent tous les deux vers "image/jpeg" (même format, deux extensions)
        mime_types = {
            "png":  "image/png",
            "jpg":  "image/jpeg",
            "jpeg": "image/jpeg",
            "gif":  "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_types[extension]

        # Encode les bytes de l'image en base64 (chaîne ASCII).
        # base64.b64encode() retourne des bytes ; .decode("utf-8") les convertit en str.
        # Le base64 est le seul moyen d'insérer des données binaires dans un JSON / URL.
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        # Construit le data URI : format standard pour embarquer une image dans une URL.
        # Exemple : "data:image/png;base64,iVBORw0KGgo..."
        # C'est ce format qu'attend l'API Mistral pour les images envoyées en message.
        data_uri = f"data:{mime_type};base64,{image_b64}"

        try:
            # Appel à l'API Mistral avec le modèle Pixtral (modèle vision multimodal).
            # On envoie un message "user" contenant deux parties :
            #   1. L'image encodée en base64 sous forme de data URI
            #   2. L'instruction texte demandant d'extraire le texte visible
            response = self.client.chat.complete(
                model=self.ocr_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                # Type "image_url" indique à l'API que ce contenu est une image
                                "type": "image_url",
                                "image_url": {"url": data_uri},
                            },
                            {
                                # Type "text" indique à l'API que ce contenu est un prompt texte.
                                # On demande explicitement de retourner uniquement le texte extrait,
                                # sans reformulation ni commentaire, pour un résultat brut et propre.
                                "type": "text",
                                "text": (
                                    "Extrais tout le texte visible dans cette image. "
                                    "Retourne uniquement le texte, sans commentaire ni reformulation."
                                ),
                            },
                        ],
                    }
                ],
            )

            # Récupère le texte généré par le modèle depuis la réponse.
            # response.choices est une liste de choix ; on prend le premier ([0]).
            # .message.content contient le texte produit par Pixtral.
            # .strip() supprime les espaces et sauts de ligne superflus en début/fin.
            extracted_text = response.choices[0].message.content.strip()

            return {
                # Texte extrait de l'image par Pixtral
                "text": extracted_text,
                # Pixtral ne retourne pas de score de confiance : toujours None
                "confidence": None,
            }

        except Exception as e:
            # Capture toutes les erreurs réseau ou API Mistral
            # et les reformule avec un message lisible
            raise ValueError(f"Échec de l'OCR : {e}") from e

    def ocr_image_file(
        self,
        image_file_path: str,
    ) -> Dict[str, Any]:
        """Effectue l'OCR sur une image depuis le disque (wrapper de ocr_image).

        Lit le fichier puis délègue à ocr_image().

        Args:
            image_file_path: Chemin absolu ou relatif vers le fichier image

        Returns:
            Dict[str, Any]: Même format que ocr_image()

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le format n'est pas supporté ou l'OCR échoue
        """
        # Convertit le chemin string en objet Path pour accéder à .exists() et .name
        path = Path(image_file_path)

        # Vérifie que le fichier existe avant de tenter de le lire.
        # Lève FileNotFoundError avec un message clair plutôt qu'une erreur générique.
        if not path.exists():
            raise FileNotFoundError(f"Fichier image introuvable : {image_file_path}")

        # Lit l'intégralité du fichier en bytes.
        # "rb" (read binary) est obligatoire pour les images.
        image_data = path.read_bytes()

        # Délègue à la méthode core en passant le nom du fichier (avec son extension)
        # pour que ocr_image() puisse déduire le format et construire le bon MIME type.
        return self.ocr_image(image_data, filename=path.name)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def get_supported_image_formats(self) -> list[str]:
        """Retourne la liste des extensions image supportées pour l'OCR.

        Returns:
            list[str]: Extensions supportées (ex: ['png', 'jpg', 'jpeg', ...])
        """
        # Retourne une copie de la constante de classe pour éviter qu'un appelant
        # externe puisse modifier la liste originale par accident (ex: .append())
        return list(self.SUPPORTED_IMAGE_FORMATS)

    def validate_pdf_file(self, file_path: str) -> bool:
        """Vérifie qu'un fichier PDF est lisible.

        Args:
            file_path: Chemin vers le fichier PDF

        Returns:
            bool: True si le fichier est un PDF valide et lisible, False sinon
        """
        try:
            path = Path(file_path)

            # Vérifie que le fichier existe physiquement sur le disque
            if not path.exists():
                return False

            # Vérifie que l'extension est bien .pdf (insensible à la casse)
            if path.suffix.lower() != ".pdf":
                return False

            # Tente d'ouvrir le PDF avec pdfplumber pour vérifier qu'il est lisible.
            # Un fichier peut avoir l'extension .pdf sans être un PDF valide (corrompu,
            # chiffré, ou simplement renommé). Cette vérification réelle est nécessaire.
            # On utilise io.BytesIO pour ne pas garder le fichier ouvert après la vérif.
            with pdfplumber.open(io.BytesIO(path.read_bytes())):
                # Si pdfplumber ouvre sans erreur, le PDF est valide et lisible
                pass

            return True

        except Exception:
            # Toute exception (FileNotFoundError, pdfplumber.PDFSyntaxError, etc.)
            # signifie que le fichier n'est pas un PDF utilisable → on retourne False
            return False

    def validate_image_file(self, file_path: str) -> bool:
        """Vérifie qu'un fichier image est dans un format supporté pour l'OCR.

        Args:
            file_path: Chemin vers le fichier image

        Returns:
            bool: True si le format est supporté et le fichier lisible, False sinon
        """
        try:
            path = Path(file_path)

            # Vérifie que le fichier existe physiquement sur le disque
            if not path.exists():
                return False

            # Extrait et normalise l'extension (ex: ".PNG" → "png")
            extension = path.suffix.lstrip(".").lower()

            # Vérifie que l'extension figure dans la liste des formats supportés par Pixtral
            if extension not in self.SUPPORTED_IMAGE_FORMATS:
                return False

            # Vérifie que le fichier est lisible et non vide.
            # stat().st_size retourne la taille en octets ; un fichier de 0 octet
            # ne peut pas être une image valide.
            if path.stat().st_size == 0:
                return False

            return True

        except Exception:
            # Toute exception (permissions, chemin invalide, etc.)
            # signifie que le fichier n'est pas utilisable → on retourne False
            return False
