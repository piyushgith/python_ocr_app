from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class OcrBlock:
    """One recognized text line. `bbox` is (x_min, y_min, x_max, y_max) in pixels of the input image."""

    text: str
    confidence: float | None
    bbox: tuple[float, float, float, float]


class OcrEngine(ABC):
    name: str
    capabilities: list[str]

    @property
    @abstractmethod
    def supported_languages(self) -> frozenset[str]:
        """Application language codes (en, fr, ar) this engine can currently use."""

    @abstractmethod
    def problems(self) -> list[str]:
        """Reasons the engine cannot run, with remediation. Empty when it is ready. Must not load models."""

    def notes(self) -> list[str]:
        """Informational messages shown next to the engine, such as first-run downloads."""
        return []

    @abstractmethod
    def recognize(self, image: Image.Image, languages: list[str]) -> list[OcrBlock]:
        """Runs OCR on an RGB image. Raises AppError when the engine or its models are unusable."""
