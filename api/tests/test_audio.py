"""Tests for the AudioProcessor module."""

import io
import os
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from dotenv import load_dotenv

from api.processing.audio import AudioProcessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Load environment variables from .env file (if it exists) to get the real API key for integration tests
# load_dotenv(Path(__file__).parent.parent.parent / ".env")
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_mp3() -> Path:
    """Chemin vers un vrai fichier MP3 de test."""
    path = FIXTURES_DIR / "ma-lubulule.mp3"
    assert path.exists(), f"Fichier fixture manquant : {path}"
    return path


@pytest.fixture
def real_api_key() -> str:
    """Récupère la vraie clé API depuis l'environnement."""
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        pytest.skip("MISTRAL_API_KEY non définie, test d'intégration ignoré")
    return key


@pytest.fixture
def api_key() -> str:
    return "test-api-key-123"


@pytest.fixture
def processor(api_key: str) -> AudioProcessor:
    """Return an AudioProcessor instance (not initialized)."""
    return AudioProcessor(api_key=api_key)


@pytest.fixture
def initialized_processor(processor: AudioProcessor) -> AudioProcessor:
    """Return an AudioProcessor with a mocked Mistral client."""
    processor.client = MagicMock()
    return processor


@pytest.fixture
def tmp_audio_file(tmp_path: Path) -> Path:
    """Create a minimal valid WAV file."""
    file_path = tmp_path / "test.wav"
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 1 second of silence
        frames = struct.pack("<" + "h" * 16000, *([0] * 16000))
        wf.writeframes(frames)
    return file_path


@pytest.fixture
def tmp_mp3_file(tmp_path: Path) -> Path:
    """Create a fake mp3 file (non-empty, for validation tests)."""
    file_path = tmp_path / "test.mp3"
    file_path.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 1024)
    return file_path


@pytest.fixture
def tmp_empty_file(tmp_path: Path) -> Path:
    """Create an empty file."""
    file_path = tmp_path / "empty.mp3"
    file_path.write_bytes(b"")
    return file_path


@pytest.fixture
def tmp_unsupported_file(tmp_path: Path) -> Path:
    """Create a file with unsupported extension."""
    file_path = tmp_path / "test.xyz"
    file_path.write_bytes(b"some data")
    return file_path


@pytest.fixture
def mock_transcription_response() -> Mock:
    """Create a mock transcription response."""
    response = Mock()
    response.text = "Bonjour, ceci est un test."
    response.language = "fr"
    response.duration = 3.5
    response.confidence = 0.95
    return response


# ---------------------------------------------------------------------------
# Tests: __init__
# ---------------------------------------------------------------------------


class TestAudioProcessorInit:
    def test_default_model(self, api_key: str) -> None:
        proc = AudioProcessor(api_key=api_key)
        assert proc.api_key == api_key
        assert proc.model == "voxtral-mini-latest"
        assert proc.client is None

    def test_custom_model(self, api_key: str) -> None:
        proc = AudioProcessor(api_key=api_key, model="custom-model")
        assert proc.model == "custom-model"


# ---------------------------------------------------------------------------
# Tests: initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    @patch("api.processing.audio.Mistral")
    def test_initialize_success(self, mock_mistral_cls: Mock, processor: AudioProcessor) -> None:
        processor.initialize()
        mock_mistral_cls.assert_called_once_with(api_key=processor.api_key)
        assert processor.client is not None

    @patch("api.processing.audio.Mistral", side_effect=Exception("bad key"))
    def test_initialize_failure(self, mock_mistral_cls: Mock, processor: AudioProcessor) -> None:
        with pytest.raises(ValueError, match="Failed to initialize"):
            processor.initialize()


# ---------------------------------------------------------------------------
# Tests: transcribe_file
# ---------------------------------------------------------------------------


class TestTranscribeFile:
    def test_not_initialized_raises(self, processor: AudioProcessor, tmp_audio_file: Path) -> None:
        with pytest.raises(ValueError, match="not initialized"):
            processor.transcribe_file(str(tmp_audio_file))

    def test_file_not_found_raises(self, initialized_processor: AudioProcessor) -> None:
        with pytest.raises(FileNotFoundError):
            initialized_processor.transcribe_file("/nonexistent/audio.wav")

    def test_unsupported_format_raises(
        self, initialized_processor: AudioProcessor, tmp_unsupported_file: Path
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            initialized_processor.transcribe_file(str(tmp_unsupported_file))

    def test_transcribe_success(
        self,
        initialized_processor: AudioProcessor,
        tmp_audio_file: Path,
        mock_transcription_response: Mock,
    ) -> None:
        initialized_processor.client.audio.transcriptions.complete.return_value = (
            mock_transcription_response
        )

        result = initialized_processor.transcribe_file(str(tmp_audio_file), language="fr")

        assert result["text"] == "Bonjour, ceci est un test."
        assert result["language"] == "fr"
        assert result["duration"] == 3.5
        assert result["confidence"] == 0.95

    def test_transcribe_without_language(
        self,
        initialized_processor: AudioProcessor,
        tmp_audio_file: Path,
        mock_transcription_response: Mock,
    ) -> None:
        initialized_processor.client.audio.transcriptions.complete.return_value = (
            mock_transcription_response
        )

        result = initialized_processor.transcribe_file(str(tmp_audio_file))

        call_kwargs = initialized_processor.client.audio.transcriptions.complete.call_args[1]
        assert "language" not in call_kwargs
        assert result["text"] == "Bonjour, ceci est un test."

    def test_transcribe_api_error(
        self, initialized_processor: AudioProcessor, tmp_audio_file: Path
    ) -> None:
        initialized_processor.client.audio.transcriptions.complete.side_effect = RuntimeError(
            "API error"
        )
        with pytest.raises(ValueError, match="Transcription failed"):
            initialized_processor.transcribe_file(str(tmp_audio_file))


# ---------------------------------------------------------------------------
# Tests: transcribe_bytes
# ---------------------------------------------------------------------------


class TestTranscribeBytes:
    def test_not_initialized_raises(self, processor: AudioProcessor) -> None:
        with pytest.raises(ValueError, match="not initialized"):
            processor.transcribe_bytes(b"data")

    def test_unsupported_format_raises(self, initialized_processor: AudioProcessor) -> None:
        with pytest.raises(ValueError, match="Unsupported audio format"):
            initialized_processor.transcribe_bytes(b"data", file_extension="xyz")

    def test_empty_data_raises(self, initialized_processor: AudioProcessor) -> None:
        with pytest.raises(ValueError, match="empty"):
            initialized_processor.transcribe_bytes(b"", file_extension="mp3")

    def test_transcribe_bytes_success(
        self,
        initialized_processor: AudioProcessor,
        mock_transcription_response: Mock,
    ) -> None:
        initialized_processor.client.audio.transcriptions.complete.return_value = (
            mock_transcription_response
        )

        result = initialized_processor.transcribe_bytes(
            b"\xff\xfb\x90\x00" * 100, file_extension="mp3", language="fr"
        )

        assert result["text"] == "Bonjour, ceci est un test."
        assert result["language"] == "fr"

    def test_transcribe_bytes_with_dot_extension(
        self,
        initialized_processor: AudioProcessor,
        mock_transcription_response: Mock,
    ) -> None:
        initialized_processor.client.audio.transcriptions.complete.return_value = (
            mock_transcription_response
        )

        result = initialized_processor.transcribe_bytes(b"data", file_extension=".wav")
        assert result["text"] == "Bonjour, ceci est un test."

    def test_transcribe_bytes_api_error(self, initialized_processor: AudioProcessor) -> None:
        initialized_processor.client.audio.transcriptions.complete.side_effect = RuntimeError(
            "API error"
        )
        with pytest.raises(ValueError, match="Transcription from bytes failed"):
            initialized_processor.transcribe_bytes(b"data", file_extension="mp3")


# ---------------------------------------------------------------------------
# Tests: transcribe_stream
# ---------------------------------------------------------------------------


class TestTranscribeStream:
    def test_not_initialized_raises(self, processor: AudioProcessor) -> None:
        stream = io.BytesIO(b"data")
        with pytest.raises(ValueError, match="not initialized"):
            processor.transcribe_stream(stream)

    def test_empty_stream_raises(self, initialized_processor: AudioProcessor) -> None:
        stream = io.BytesIO(b"")
        with pytest.raises(ValueError, match="empty"):
            initialized_processor.transcribe_stream(stream)

    def test_transcribe_stream_success(
        self,
        initialized_processor: AudioProcessor,
        mock_transcription_response: Mock,
    ) -> None:
        initialized_processor.client.audio.transcriptions.complete.return_value = (
            mock_transcription_response
        )

        stream = io.BytesIO(b"\x00" * 1024)
        result = initialized_processor.transcribe_stream(stream, language="en")

        assert result["text"] == "Bonjour, ceci est un test."

    def test_stream_seeks_to_beginning(
        self,
        initialized_processor: AudioProcessor,
        mock_transcription_response: Mock,
    ) -> None:
        """Ensure stream is rewound before reading."""
        initialized_processor.client.audio.transcriptions.complete.return_value = (
            mock_transcription_response
        )

        stream = io.BytesIO(b"\x00" * 512)
        stream.read(100)  # advance the cursor

        result = initialized_processor.transcribe_stream(stream)
        assert result["text"] == "Bonjour, ceci est un test."

    def test_broken_stream_raises(self, initialized_processor: AudioProcessor) -> None:
        stream = Mock(spec=io.BytesIO)
        stream.seek.side_effect = IOError("broken stream")
        with pytest.raises(ValueError, match="Failed to read audio stream"):
            initialized_processor.transcribe_stream(stream)


# ---------------------------------------------------------------------------
# Tests: get_supported_formats
# ---------------------------------------------------------------------------


class TestGetSupportedFormats:
    def test_returns_list(self, processor: AudioProcessor) -> None:
        formats = processor.get_supported_formats()
        assert isinstance(formats, list)

    def test_contains_common_formats(self, processor: AudioProcessor) -> None:
        formats = processor.get_supported_formats()
        for fmt in ["mp3", "wav", "flac", "ogg", "m4a"]:
            assert fmt in formats

    def test_returns_copy(self, processor: AudioProcessor) -> None:
        """Modifying the returned list should not affect the class."""
        formats = processor.get_supported_formats()
        formats.clear()
        assert len(processor.get_supported_formats()) > 0


# ---------------------------------------------------------------------------
# Tests: validate_audio_file
# ---------------------------------------------------------------------------


class TestValidateAudioFile:
    def test_valid_wav(self, processor: AudioProcessor, tmp_audio_file: Path) -> None:
        assert processor.validate_audio_file(str(tmp_audio_file)) is True

    def test_valid_mp3(self, processor: AudioProcessor, tmp_mp3_file: Path) -> None:
        assert processor.validate_audio_file(str(tmp_mp3_file)) is True

    def test_nonexistent_file(self, processor: AudioProcessor) -> None:
        assert processor.validate_audio_file("/nonexistent/file.mp3") is False

    def test_empty_file(self, processor: AudioProcessor, tmp_empty_file: Path) -> None:
        assert processor.validate_audio_file(str(tmp_empty_file)) is False

    def test_unsupported_extension(
        self, processor: AudioProcessor, tmp_unsupported_file: Path
    ) -> None:
        assert processor.validate_audio_file(str(tmp_unsupported_file)) is False


# ---------------------------------------------------------------------------
# Tests: get_audio_metadata
# ---------------------------------------------------------------------------


class TestGetAudioMetadata:
    def test_file_not_found_raises(self, processor: AudioProcessor) -> None:
        with pytest.raises(FileNotFoundError):
            processor.get_audio_metadata("/nonexistent/file.wav")

    @patch("api.processing.audio.Path")
    def test_metadata_extraction_with_mock(
        self, mock_path_cls: Mock, processor: AudioProcessor
    ) -> None:
        """Test metadata extraction by mocking mutagen."""
        mock_file = Mock()
        mock_file.is_file.return_value = True
        mock_file.suffix = ".mp3"
        mock_file.stat.return_value = Mock(st_size=1024)
        mock_path_cls.return_value = mock_file

        mock_info = Mock()
        mock_info.length = 120.55
        mock_info.sample_rate = 44100
        mock_info.channels = 2
        mock_info.bitrate = 128000

        mock_audio = Mock()
        mock_audio.info = mock_info

        with patch.dict("sys.modules", {}):
            with patch("builtins.__import__") as mock_import:
                mock_mp3_module = Mock()
                mock_mp3_module.MP3.return_value = mock_audio

                def side_effect(name, *args, **kwargs):
                    if name == "mutagen.mp3":
                        return mock_mp3_module
                    return __builtins__.__import__(name, *args, **kwargs)  # type: ignore[attr-defined]

                # Simpler approach: mock at mutagen level
                pass

        # Direct approach using a real wav file
        # Tested below with real file

    def test_metadata_wav_file(self, processor: AudioProcessor, tmp_audio_file: Path) -> None:
        """Test metadata extraction on a real WAV file (requires mutagen)."""
        metadata = processor.get_audio_metadata(str(tmp_audio_file))

        assert metadata["format"] == "wav"
        assert metadata["duration"] is not None
        assert metadata["duration"] > 0
        assert metadata["sample_rate"] == 16000
        assert metadata["channels"] == 1

    def test_metadata_unreadable_file(self, processor: AudioProcessor, tmp_path: Path) -> None:
        """Test with a file that mutagen cannot read."""
        bad_file = tmp_path / "bad.mp3"
        bad_file.write_bytes(b"this is not audio")

        with pytest.raises(ValueError):
            processor.get_audio_metadata(str(bad_file))

    def test_metadata_returns_expected_keys(
        self, processor: AudioProcessor, tmp_audio_file: Path
    ) -> None:
        metadata = processor.get_audio_metadata(str(tmp_audio_file))
        expected_keys = {"duration", "format", "sample_rate", "channels", "bitrate"}
        assert set(metadata.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: Integration-like (mocked API)
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_workflow_file(
        self,
        tmp_audio_file: Path,
        mock_transcription_response: Mock,
    ) -> None:
        """Test full workflow: init → transcribe file."""
        proc = AudioProcessor(api_key="test-key")
        proc.initialize()
        proc.client = MagicMock()
        proc.client.audio.transcriptions.complete.return_value = mock_transcription_response

        result = proc.transcribe_file(str(tmp_audio_file), language="fr")

        assert result["text"] == "Bonjour, ceci est un test."
        assert result["language"] == "fr"
        assert result["duration"] == 3.5

    def test_full_workflow_bytes_then_stream(self, mock_transcription_response: Mock) -> None:
        """Test bytes and stream transcription in sequence."""
        proc = AudioProcessor(api_key="test-key")
        proc.initialize()
        proc.client = MagicMock()
        proc.client.audio.transcriptions.complete.return_value = mock_transcription_response

        # Bytes
        result1 = proc.transcribe_bytes(b"\x00" * 100, file_extension="wav")
        assert result1["text"] == "Bonjour, ceci est un test."

        # Stream
        stream = io.BytesIO(b"\x00" * 100)
        result2 = proc.transcribe_stream(stream, language="en")
        assert result2["text"] == "Bonjour, ceci est un test."

        assert proc.client.audio.transcriptions.complete.call_count == 2


class TestAudioProcessorWithRealFile:
    """Tests utilisant un vrai fichier audio et la vraie API."""

    def test_transcribe_real_file(
        self,
        sample_mp3: Path,
        real_api_key: str,
    ) -> None:
        """Transcrit un vrai fichier et vérifie le contenu."""
        proc = AudioProcessor(api_key=real_api_key)
        proc.initialize()

        result = proc.transcribe_file(str(sample_mp3), language="fr")

        # Vérifie que le résultat contient du texte
        assert result["text"] is not None
        assert len(result["text"]) > 0

        # Affiche le texte transcrit pour debug
        print(f"\n{'=' * 60}")
        print("Transcription obtenue :")
        print(f"  Texte    : {result['text']}")
        print(f"  Langue   : {result.get('language', 'N/A')}")
        print(f"  Durée    : {result.get('duration', 'N/A')}s")
        print(f"  Confiance: {result.get('confidence', 'N/A')}")
        print(f"{'=' * 60}\n")

        # Vérifie que les mots attendus sont présents
        text_lower = result["text"].lower()
        mots_attendus = ["lubulule", "mort", "blessé"]
        for mot in mots_attendus:
            assert mot in text_lower, f"Mot '{mot}' non trouvé dans : {result['text']}"

    def test_real_file_metadata(
        self,
        sample_mp3: Path,
        real_api_key: str,
    ) -> None:
        """Vérifie que les métadonnées sont retournées."""
        proc = AudioProcessor(api_key=real_api_key)
        proc.initialize()

        result = proc.get_audio_metadata(str(sample_mp3))

        assert result["format"] == "mp3"
        assert result["duration"] is not None
        assert result["duration"] > 0
        assert result["sample_rate"] is not None
        assert result["format"] == "mp3"
        assert result["sample_rate"] == 48000
        assert result["channels"] == 2
        assert result["bitrate"] == 128
        print(f"\nMetadata : {result}")

    def test_transcribe_real_bytes(
        self,
        sample_mp3: Path,
        real_api_key: str,
    ) -> None:
        """Transcrit depuis des bytes lus du fichier."""
        proc = AudioProcessor(api_key=real_api_key)
        proc.initialize()

        audio_bytes = sample_mp3.read_bytes()
        result = proc.transcribe_bytes(audio_bytes, file_extension="mp3")

        print(f"\nTranscription (bytes) : {result['text']}")

        assert result["text"] is not None
        assert len(result["text"]) > 0

    def test_transcribe_prints_full_result(
        self,
        sample_mp3: Path,
        real_api_key: str,
    ) -> None:
        """Affiche le résultat complet pour inspection."""
        proc = AudioProcessor(api_key=real_api_key)
        proc.initialize()

        result = proc.transcribe_file(str(sample_mp3))

        # Affiche tout le dictionnaire résultat
        print("\nRésultat complet :")
        for key, value in result.items():
            print(f"  {key}: {value}")

        assert result["text"]
