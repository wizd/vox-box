from abc import ABC, abstractmethod
import logging
from typing import Dict, Generator, Optional
from vox_box.config.config import Config
from vox_box.utils.log import log_method

logger = logging.getLogger(__name__)


class TTSBackend(ABC):
    def __init__(
        self,
        cfg: Config,
    ):
        pass

    @abstractmethod
    def load() -> bool:
        pass

    @abstractmethod
    def is_load() -> bool:
        pass

    @abstractmethod
    def model_info() -> Dict:
        pass

    @log_method
    def speech(
        self,
        input: str,
        voice: Optional[str],
        speed: float = 1,
        reponse_format: str = "mp3",
        **kwargs
    ):
        pass

    def speech_stream(
        self,
        input: str,
        voice: Optional[str],
        speed: float = 1,
        **kwargs,
    ) -> Generator[bytes, None, None]:
        """Stream raw PCM audio chunks (16-bit signed, mono).
        Override in subclass to enable streaming TTS.
        Yields bytes of PCM audio data.
        """
        raise NotImplementedError("Streaming not supported by this backend")

    @property
    def stream_sample_rate(self) -> int:
        """Return the sample rate for streaming audio. Override in subclass."""
        return 22050
