"""Audio processing module for Speech-to-Text (STT)."""

from typing import Optional, Dict, Any
from pathlib import Path
import io
from mistralai import Mistral


class AudioProcessor:
    """Audio processor using Mistral API for STT."""

    def __init__(self, api_key: str, model: str = "whisper-large-v3") -> None:
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
        pass

    def transcribe_file(
        self,
        audio_file_path: str,
        language: Optional[str] = None
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
        pass

    def transcribe_bytes(
        self,
        audio_data: bytes,
        file_extension: str = "mp3",
        language: Optional[str] = None
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
        pass

    def transcribe_stream(
        self,
        audio_stream: io.BytesIO,
        language: Optional[str] = None
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
        pass

    def get_supported_formats(self) -> list[str]:
        """Get list of supported audio formats.
        
        Returns:
            list[str]: List of supported file extensions (e.g., ['mp3', 'wav', 'm4a'])
        """
        pass

    def validate_audio_file(self, file_path: str) -> bool:
        """Validate if audio file is supported and readable.
        
        Args:
            file_path: Path to audio file
        
        Returns:
            bool: True if file is valid, False otherwise
        """
        pass

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
        pass
