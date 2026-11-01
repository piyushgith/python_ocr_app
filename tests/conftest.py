from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.extractors.base import OcrBlock, OcrEngine
from app.main import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class FakeEngine(OcrEngine):
    """Deterministic engine for unit tests: returns one fixed line per page and records its calls."""

    name = "paddleocr"
    capabilities = ["ocr"]

    def __init__(self, text: str = "FAKE OCR TEXT", languages=frozenset({"en", "fr", "ar"}), problems=None, delay=0.0):
        self.text = text
        self._languages = languages
        self._problems = problems or []
        self.delay = delay
        self.calls: list[tuple[tuple[int, int], list[str]]] = []

    @property
    def supported_languages(self):
        return self._languages

    def problems(self):
        return self._problems

    def recognize(self, image: Image.Image, languages: list[str]):
        import time

        time.sleep(self.delay)
        self.calls.append((image.size, list(languages)))
        return [OcrBlock(text=self.text, confidence=0.9, bbox=(10, 20, 110, 70))]


class UnavailableTesseract(FakeEngine):
    name = "tesseract"

    def __init__(self):
        super().__init__(languages=frozenset(), problems=["Tesseract binary 'tesseract' not found."])


def build_client(tmp_path, engine=None, tesseract=None, **setting_overrides):
    settings = Settings(model_cache_dir=tmp_path / "models", **setting_overrides)
    engines = {"paddleocr": engine or FakeEngine(), "tesseract": tesseract or UnavailableTesseract()}
    return TestClient(create_app(settings, engines))


@pytest.fixture
def fake_engine():
    return FakeEngine()


@pytest.fixture
def client(tmp_path, fake_engine):
    with build_client(tmp_path, engine=fake_engine) as test_client:
        yield test_client


@pytest.fixture
def samples() -> Path:
    return SAMPLES
