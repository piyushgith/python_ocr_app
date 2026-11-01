"""Real OCR with downloaded models. Skips (with a reason) when models or Tesseract are missing.

Run: pytest -m integration -rs   (a skip is NOT a pass: check the skip reasons in the summary)
"""

import json
import re
from pathlib import Path

import pytest

from app.schemas.ocr import EngineChoice, ExtractionOptions, Mode
from tests.integration.metrics import character_error_rate

pytestmark = pytest.mark.integration

SAMPLES = Path(__file__).resolve().parent.parent.parent / "samples"
TRUTH = json.loads((SAMPLES / "ground_truth.json").read_text(encoding="utf-8"))["texts"]
ARABIC_LETTERS = re.compile(r"[؀-ۿ]")

# Thresholds sit just above the CER measured on these clean synthetic renderings.
LATIN_MAX_CER = 0.02
ARABIC_MAX_CER = 0.05
MIXED_SCRIPT_MIN_TOKEN_RECALL = 0.7


def extract(processor, filename, languages, engine=EngineChoice.paddleocr, mode=Mode.auto):
    options = ExtractionOptions(engine=engine, languages=languages, mode=mode)
    return processor.process(SAMPLES / filename, filename, options)


def test_english_image(real_processor, record_cer):
    result = extract(real_processor, "english.png", ["en"])

    cer = character_error_rate("\n".join(TRUTH["english"]), result.pages[0].text)
    record_cer("paddleocr english.png [en]", cer)
    assert cer <= LATIN_MAX_CER, result.pages[0].text
    assert all(block.confidence is not None and 0 < block.confidence <= 1 for block in result.pages[0].blocks)


def test_french_image_preserves_accents(real_processor, record_cer):
    result = extract(real_processor, "french.png", ["fr"])

    text = result.pages[0].text
    cer = character_error_rate("\n".join(TRUTH["french"]), text)
    record_cer("paddleocr french.png [fr]", cer)
    assert cer <= LATIN_MAX_CER, text
    for accented in ("Relevé", "Échéance", "février", "Café", "crème", "à", "Noël", "où", "€"):
        assert accented in text


def test_english_and_french_together(real_processor, record_cer):
    result = extract(real_processor, "french.png", ["en", "fr"])

    cer = character_error_rate("\n".join(TRUTH["french"]), result.pages[0].text)
    record_cer("paddleocr french.png [en,fr]", cer)
    assert cer <= LATIN_MAX_CER


def test_arabic_image_is_logical_order_unicode(real_processor, record_cer):
    result = extract(real_processor, "arabic.png", ["ar"])

    text = result.pages[0].text
    cer = character_error_rate("\n".join(TRUTH["arabic"]), text)
    record_cer("paddleocr arabic.png [ar]", cer)
    assert cer <= ARABIC_MAX_CER, text
    assert ARABIC_LETTERS.search(text)
    assert text.splitlines()[0] == TRUTH["arabic"][0]
    assert "123456789" in text


def test_arabic_model_also_reads_latin_text(real_processor, record_cer):
    result = extract(real_processor, "english.png", ["en", "ar"])

    cer = character_error_rate("\n".join(TRUTH["english"]), result.pages[0].text)
    record_cer("paddleocr english.png [en,ar]", cer)
    assert cer <= LATIN_MAX_CER


def test_mixed_arabic_latin_is_best_effort(real_processor, record_cer):
    """Mixed-script lines are not reliable with PaddleOCR (measured). This guards against a total regression only."""
    result = extract(real_processor, "mixed_arabic_latin.png", ["ar", "en"])

    expected_tokens = " ".join(TRUTH["mixed_arabic_latin"]).replace(":", "").split()
    actual_tokens = result.pages[0].text.replace(":", "").split()
    recall = sum(token in actual_tokens for token in expected_tokens) / len(expected_tokens)
    record_cer("paddleocr mixed_arabic_latin.png (1 - token recall)", 1 - recall)
    assert recall >= MIXED_SCRIPT_MIN_TOKEN_RECALL, result.pages[0].text


@pytest.mark.xfail(
    reason="PaddleOCR returns mixed Arabic/Latin lines in non-logical order (measured). The app does not reorder them.",
    strict=False,
)
def test_mixed_arabic_latin_logical_order(real_processor):
    result = extract(real_processor, "mixed_arabic_latin.png", ["ar", "en"])

    assert result.pages[0].text == "\n".join(TRUTH["mixed_arabic_latin"])


def test_scanned_pdf_is_ocred_page_by_page(real_processor, record_cer):
    result = extract(real_processor, "scanned.pdf", ["en", "fr"])

    assert [page.method for page in result.pages] == ["ocr", "ocr"]
    first = character_error_rate("\n".join(TRUTH["english"]), result.pages[0].text)
    second = character_error_rate("\n".join(TRUTH["french"]), result.pages[1].text)
    record_cer("paddleocr scanned.pdf page 1 [en,fr]", first)
    record_cer("paddleocr scanned.pdf page 2 [en,fr]", second)
    assert first <= LATIN_MAX_CER and second <= LATIN_MAX_CER


def test_mixed_pdf_uses_native_then_ocr(real_processor, record_cer):
    result = extract(real_processor, "mixed.pdf", ["en"])

    assert [page.method for page in result.pages] == ["native", "ocr"]
    assert result.pages[0].text == "Native text page: Relevé de compte"
    cer = character_error_rate("\n".join(TRUTH["english"]), result.pages[1].text)
    record_cer("paddleocr mixed.pdf page 2 [en]", cer)
    assert cer <= LATIN_MAX_CER


def test_searchable_pdf_is_native_even_with_real_engine(real_processor):
    result = extract(real_processor, "searchable.pdf", ["fr"])

    assert result.extraction.mode == "native"
    assert result.extraction.engine is None
    assert "Relevé de compte bancaire" in result.pages[0].text


def test_forced_ocr_on_searchable_pdf_keeps_native_text(real_processor):
    result = extract(real_processor, "searchable.pdf", ["fr"], mode=Mode.ocr)

    assert result.pages[0].method == "ocr"
    assert "Relevé de compte bancaire" in result.pages[0].native_text
    assert "Relevé" in result.pages[0].text


def test_end_to_end_job_over_http_with_real_engine(paddle_engine, tmp_path):
    import time

    from fastapi.testclient import TestClient

    from app.core.config import Settings
    from app.extractors.tesseract_extractor import TesseractEngine
    from app.main import create_app
    from tests.integration.conftest import MODEL_DIR

    app = create_app(
        Settings(model_cache_dir=MODEL_DIR), {"paddleocr": paddle_engine, "tesseract": TesseractEngine("missing-tesseract")}
    )
    with TestClient(app) as client:
        submitted = client.post(
            "/api/v1/ocr/jobs",
            files={"file": ("arabic.png", (SAMPLES / "arabic.png").read_bytes())},
            data={"languages": "ar"},
        )
        job_id = submitted.json()["job_id"]
        for _ in range(100):
            job = client.get(f"/api/v1/ocr/jobs/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(0.2)

    assert job["status"] == "completed", job
    assert job["result"]["pages"][0]["text"].splitlines()[0] == TRUTH["arabic"][0]


# Tesseract 5.5.4 skips the middle line of the Arabic sample entirely (measured CER 0.29); the lines it returns are exact.
@pytest.mark.parametrize(
    "filename, languages, truth_key, max_cer",
    [
        ("english.png", ["en"], "english", LATIN_MAX_CER),
        ("french.png", ["fr"], "french", LATIN_MAX_CER),
        ("arabic.png", ["ar"], "arabic", 0.35),
    ],
)
def test_tesseract_engine(real_processor, tesseract_engine, record_cer, filename, languages, truth_key, max_cer):
    result = extract(real_processor, filename, languages, engine=EngineChoice.tesseract)

    text = result.pages[0].text
    cer = character_error_rate("\n".join(TRUTH[truth_key]), text)
    record_cer(f"tesseract {filename} {languages}", cer)
    assert result.extraction.engine == "tesseract"
    assert cer <= max_cer
    assert all(line in TRUTH[truth_key] for line in text.splitlines()), text
