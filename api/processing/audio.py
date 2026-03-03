"""Audio processing module for Speech-to-Text (STT)."""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

from mistralai import Mistral


class AudioProcessor:
    """Audio processor using Mistral API for STT."""

    SUPPORTED_FORMATS: List[str] = ["mp3", "wav", "flac", "ogg", "m4a", "webm"]

    def __init__(self, api_key: str, model: str = "voxtral-mini-latest") -> None:
        """Initialize audio processor.

        Args:
            api_key: Mistral API key
            model: STT model name to use
        """
        self.api_key = api_key
        self.model = model
        self.client: Optional[Mistral] = None

    def initialize(self) -> None:
        """Initialize Mistral client.

        Raises:
            ValueError: If API key is invalid or client initialization fails
        """
        try:
            self.client = Mistral(api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Mistral client: {e}")

    def transcribe_file(
        self, audio_file_path: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe audio file to text using STT.

        Args:
            audio_file_path: Path to audio file
            language: Optional language code (e.g., 'fr', 'en')

        Returns:
            Dict[str, Any]: Transcription result with:
                - text: Transcribed text
                - language: Detected language
                - duration: Audio duration in seconds
                - confidence: Optional confidence score

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If transcription fails or unsupported format
        """
        if not self.client:
            raise ValueError("Mistral client not initialized. Call initialize() first.")

        path = Path(audio_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        if not self.validate_audio_file(audio_file_path):
            raise ValueError(f"Unsupported or invalid audio file: {audio_file_path}")

        file_name = path.name

        try:
            with open(audio_file_path, "rb") as f:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "file": {
                        "content": f,
                        "file_name": file_name,
                    },
                }
                if language:
                    kwargs["language"] = language

                response = self.client.audio.transcriptions.complete(**kwargs)

            return {
                "text": getattr(response, "text", ""),
                "language": getattr(response, "language", language),
                "duration": getattr(response, "duration", None),
                "confidence": getattr(response, "confidence", None),
            }
        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Transcription failed: {e}")

    def transcribe_bytes(
        self,
        audio_data: bytes,
        file_extension: str = "mp3",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio from bytes using STT.

        Args:
            audio_data: Audio file content as bytes
            file_extension: File extension/format (mp3, wav, etc.)
            language: Optional language code

        Returns:
            Dict[str, Any]: Transcription result (same format as transcribe_file)

        Raises:
            ValueError: If transcription fails or unsupported format
        """
        if not self.client:
            raise ValueError("Mistral client not initialized. Call initialize() first.")

        ext = file_extension.lstrip(".").lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {ext}")

        if not audio_data:
            raise ValueError("Audio data is empty")

        file_name = f"audio.{ext}"

        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "file": {
                    "content": audio_data,
                    "file_name": file_name,
                },
            }
            if language:
                kwargs["language"] = language

            response = self.client.audio.transcriptions.complete(**kwargs)

            return {
                "text": getattr(response, "text", ""),
                "language": getattr(response, "language", language),
                "duration": getattr(response, "duration", None),
                "confidence": getattr(response, "confidence", None),
            }
        except ValueError as e:
            print(e)
            raise
        except Exception as e:
            raise ValueError(f"Transcription from bytes failed: {e}")

    def transcribe_stream(
        self, audio_stream: io.BytesIO, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe audio from stream using STT.

        Args:
            audio_stream: Audio file stream
            language: Optional language code

        Returns:
            Dict[str, Any]: Transcription result (same format as transcribe_file)

        Raises:
            ValueError: If transcription fails
        """
        if not self.client:
            raise ValueError("Mistral client not initialized. Call initialize() first.")

        try:
            audio_stream.seek(0)
            audio_data = audio_stream.read()
        except Exception as e:
            raise ValueError(f"Failed to read audio stream: {e}")

        if not audio_data:
            raise ValueError("Audio stream is empty")

        return self.transcribe_bytes(audio_data, file_extension="wav", language=language)

    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats.

        Returns:
            list[str]: List of supported file extensions (e.g., ['mp3', 'wav', 'm4a'])
        """
        return list(self.SUPPORTED_FORMATS)

    def validate_audio_file(self, file_path: str) -> bool:
        """Validate if audio file is supported and readable.

        Args:
            file_path: Path to audio file

        Returns:
            bool: True if file is valid, False otherwise
        """
        path = Path(file_path)

        if not path.is_file():
            return False

        ext = path.suffix.lstrip(".").lower()
        if ext not in self.SUPPORTED_FORMATS:
            return False

        if path.stat().st_size == 0:
            return False

        return True

    def get_audio_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Dict[str, Any]: Metadata including:
                - duration: Duration in seconds
                - format: Audio format
                - sample_rate: Sample rate in Hz
                - channels: Number of audio channels
                - bitrate: Bitrate if available

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If metadata extraction fails
        """
        file = Path(file_path)
        if not file.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            import mutagen
            from mutagen.flac import FLAC
            from mutagen.mp3 import MP3
            from mutagen.mp4 import MP4
            from mutagen.oggvorbis import OggVorbis
            from mutagen.wave import WAVE

            suffix = file.suffix.lower()

            handler_map = {
                ".mp3": MP3,
                ".flac": FLAC,
                ".ogg": OggVorbis,
                ".wav": WAVE,
                ".m4a": MP4,
                ".mp4": MP4,
                ".aac": MP4,
            }

            handler = handler_map.get(suffix)
            if handler:
                audio = handler(str(file))
            else:
                audio = mutagen.File(str(file))

            if audio is None:
                raise ValueError(f"Unable to read metadata from: {file_path}")

            metadata: Dict[str, Any] = {
                "duration": None,
                "format": suffix.lstrip("."),
                "sample_rate": None,
                "channels": None,
                "bitrate": None,
            }

            if hasattr(audio, "info") and audio.info is not None:
                info = audio.info
                if hasattr(info, "length"):
                    metadata["duration"] = round(info.length, 2)
                metadata["sample_rate"] = getattr(info, "sample_rate", None)
                metadata["channels"] = getattr(info, "channels", None)

                bitrate = getattr(info, "bitrate", None)
                if bitrate is not None:
                    # mutagen returns bitrate in bps, convert to kbps
                    metadata["bitrate"] = round(bitrate / 1000) if bitrate > 1000 else bitrate

            return metadata

        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to extract metadata from '{file_path}': {e}")
