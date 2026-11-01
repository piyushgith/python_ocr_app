import os
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import AppError
from app.extractors.paddleocr_extractor import PaddleOcrEngine
from app.extractors.tesseract_extractor import TesseractEngine
from app.services.document_service import DocumentProcessor

CER_REPORT: list[tuple[str, float]] = []

MODEL_DIR = Path(os.environ.get("OCR_MODEL_CACHE_DIR", "models"))


@pytest.fixture(scope="session")
def paddle_engine():
    engine = PaddleOcrEngine(MODEL_DIR)
    if engine.problems():
        pytest.skip(f"PaddleOCR not installed: {engine.problems()[0]}")
    from PIL import Image

    try:
        # Loads every model group once so later tests measure recognition, not downloads.
        for languages in (["en"], ["fr"], ["ar"]):
            engine.recognize(Image.new("RGB", (64, 64), "white"), languages)
    except AppError as error:
        pytest.skip(f"PaddleOCR models unavailable: {error.message}")
    return engine


@pytest.fixture(scope="session")
def tesseract_engine():
    engine = TesseractEngine(os.environ.get("OCR_TESSERACT_CMD", "tesseract"))
    if engine.problems():
        pytest.skip(f"Tesseract not usable: {engine.problems()[0]}")
    return engine


@pytest.fixture
def real_processor(paddle_engine, tesseract_engine_or_none):
    engines = {"paddleocr": paddle_engine, "tesseract": tesseract_engine_or_none}
    return DocumentProcessor(Settings(model_cache_dir=MODEL_DIR), engines)


@pytest.fixture
def tesseract_engine_or_none():
    return TesseractEngine(os.environ.get("OCR_TESSERACT_CMD", "tesseract"))


@pytest.fixture
def record_cer():
    def record(label: str, value: float) -> None:
        CER_REPORT.append((label, value))

    return record


def pytest_terminal_summary(terminalreporter):
    if CER_REPORT:
        terminalreporter.section("Character error rate (lower is better)")
        for label, value in CER_REPORT:
            terminalreporter.write_line(f"{label:<45} CER={value:.4f}")
