import logging
import os
import threading
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.errors import AppError
from app.extractors.base import OcrBlock, OcrEngine

logger = logging.getLogger(__name__)

SUPPORTED = frozenset({"en", "fr", "ar"})

MODEL_NAMES = {
    "ar": {
        "text_detection_model_name": "PP-OCRv6_medium_det",
        "text_recognition_model_name": "arabic_PP-OCRv5_mobile_rec",
    }
}


class PaddleOcrEngine(OcrEngine):
    """PaddleOCR 3.x on CPU. One pipeline per PaddleOCR `lang`, created on first use and reused.

    PaddleOCR selects a single recognition model per `lang`:
    en/fr -> PP-OCRv6 medium (Latin script), ar -> arabic PP-OCRv5 mobile (also reads Latin text and digits).
    Any request containing `ar` uses the Arabic model. Arabic is paired with the PP-OCRv6 medium detector because
    the default PP-OCRv5 server detector needed about 4x the memory on a dense A4 page.
    """

    name = "paddleocr"
    capabilities = ["ocr", "bounding_boxes", "confidence_scores"]

    def __init__(self, model_cache_dir: Path, det_max_side: int = 1280):
        self._model_cache_dir = model_cache_dir
        self._det_max_side = det_max_side
        self._pipelines: dict[str, object] = {}
        self._lock = threading.Lock()

    @property
    def supported_languages(self) -> frozenset[str]:
        return SUPPORTED if not self.problems() else frozenset()

    def problems(self) -> list[str]:
        missing = [package for package in ("paddle", "paddleocr") if find_spec(package) is None]
        if missing:
            return [
                "PaddleOCR is not installed (missing: " + ", ".join(missing) + "). "
                'Run: pip install paddlepaddle==3.3.1 paddleocr==3.7.0'
            ]
        return []

    def notes(self) -> list[str]:
        models_dir = self._model_cache_dir / "official_models"
        if models_dir.is_dir() and any(models_dir.iterdir()):
            return []
        return [f"No models cached in '{self._model_cache_dir}'. The first OCR request downloads them."]

    def recognize(self, image: Image.Image, languages: list[str]) -> list[OcrBlock]:
        pixels = np.asarray(image.convert("RGB"))
        with self._lock:
            pipeline = self._pipeline(self._paddle_lang(languages))
            try:
                results = pipeline.predict(pixels)
            except Exception as error:
                raise AppError("OCR_FAILED", "PaddleOCR failed while recognizing a page.", 500) from error
        return self._to_blocks(results)

    @staticmethod
    def _paddle_lang(languages: list[str]) -> str:
        if "ar" in languages:
            return "ar"
        return "fr" if "fr" in languages else "en"

    def _pipeline(self, paddle_lang: str):
        if paddle_lang in self._pipelines:
            return self._pipelines[paddle_lang]
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(self._model_cache_dir.resolve()))
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        logger.info("Loading PaddleOCR models for lang=%s (first use downloads them if not cached)", paddle_lang)
        try:
            from paddleocr import PaddleOCR

            pipeline = PaddleOCR(
                **MODEL_NAMES.get(paddle_lang, {"lang": paddle_lang}),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_type="max",
                text_det_limit_side_len=self._det_max_side,
            )
        except Exception as error:
            logger.exception("PaddleOCR model initialization failed for lang=%s", paddle_lang)
            raise AppError(
                "MODEL_UNAVAILABLE",
                f"PaddleOCR models for '{paddle_lang}' could not be loaded. On first use they are downloaded into "
                f"'{self._model_cache_dir}': check network access, or copy the models into that directory.",
                503,
            ) from error
        self._pipelines[paddle_lang] = pipeline
        return pipeline

    @staticmethod
    def _to_blocks(results) -> list[OcrBlock]:
        blocks: list[OcrBlock] = []
        for result in results:
            texts = result["rec_texts"]
            scores = result["rec_scores"]
            boxes = result["rec_boxes"]
            for text, score, box in zip(texts, scores, boxes):
                if not text.strip():
                    continue
                x_min, y_min, x_max, y_max = (float(value) for value in box)
                blocks.append(OcrBlock(text=text, confidence=float(score), bbox=(x_min, y_min, x_max, y_max)))
        return blocks
