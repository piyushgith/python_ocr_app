import logging
import shutil
import subprocess
import tempfile
from functools import cached_property
from pathlib import Path

from PIL import Image

from app.core.errors import AppError
from app.extractors.base import OcrBlock, OcrEngine

logger = logging.getLogger(__name__)

TESSERACT_CODES = {"en": "eng", "fr": "fra", "ar": "ara"}
RUN_TIMEOUT_SECONDS = 120


class TesseractEngine(OcrEngine):
    """Optional engine that calls the `tesseract` command line tool. Needs the binary and language data."""

    name = "tesseract"
    capabilities = ["ocr", "bounding_boxes", "confidence_scores"]

    def __init__(self, command: str = "tesseract"):
        self._command = command

    @cached_property
    def _installed_codes(self) -> frozenset[str]:
        if shutil.which(self._command) is None:
            return frozenset()
        try:
            completed = subprocess.run(
                [self._command, "--list-langs"], capture_output=True, text=True, timeout=30, check=True
            )
        except (subprocess.SubprocessError, OSError):
            return frozenset()
        return frozenset(line.strip() for line in completed.stdout.splitlines()[1:] if line.strip())

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset(code for code, tesseract_code in TESSERACT_CODES.items() if tesseract_code in self._installed_codes)

    def problems(self) -> list[str]:
        if shutil.which(self._command) is None:
            return [
                f"Tesseract binary '{self._command}' not found. Install it (macOS: brew install tesseract, "
                "Debian/Ubuntu: apt install tesseract-ocr) or set OCR_TESSERACT_CMD."
            ]
        if not self.supported_languages:
            return [
                "Tesseract has none of the language data needed (eng, fra, ara). "
                "Install traineddata files (Debian/Ubuntu: apt install tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-ara)."
            ]
        return []

    def recognize(self, image: Image.Image, languages: list[str]) -> list[OcrBlock]:
        codes = "+".join(TESSERACT_CODES[language] for language in languages)
        with tempfile.TemporaryDirectory(prefix="ocr-tess-") as workdir:
            image_path = Path(workdir) / "page.png"
            image.convert("RGB").save(image_path)
            try:
                completed = subprocess.run(
                    [self._command, str(image_path), "stdout", "-l", codes, "tsv"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=RUN_TIMEOUT_SECONDS,
                    check=True,
                )
            except subprocess.TimeoutExpired as error:
                raise AppError("TIMEOUT", "Tesseract timed out while recognizing a page.", 504) from error
            except (subprocess.CalledProcessError, OSError) as error:
                logger.error("Tesseract failed: %s", getattr(error, "stderr", error))
                raise AppError("OCR_FAILED", "Tesseract failed while recognizing a page.", 500) from error
        return parse_tsv(completed.stdout)


def parse_tsv(tsv: str) -> list[OcrBlock]:
    """Groups Tesseract word rows (level 5) into one block per line. Confidence is the mean word confidence / 100."""
    lines: dict[tuple[str, str, str], list[tuple[str, float, int, int, int, int]]] = {}
    for row in tsv.splitlines()[1:]:
        columns = row.split("\t")
        if len(columns) < 12 or columns[0] != "5":
            continue
        text = columns[11].strip()
        confidence = float(columns[10])
        if not text or confidence < 0:
            continue
        left, top, width, height = (int(value) for value in columns[6:10])
        lines.setdefault((columns[2], columns[3], columns[4]), []).append((text, confidence, left, top, width, height))

    blocks = []
    for words in lines.values():
        blocks.append(
            OcrBlock(
                text=" ".join(word[0] for word in words),
                confidence=sum(word[1] for word in words) / len(words) / 100,
                bbox=(
                    float(min(word[2] for word in words)),
                    float(min(word[3] for word in words)),
                    float(max(word[2] + word[4] for word in words)),
                    float(max(word[3] + word[5] for word in words)),
                ),
            )
        )
    return blocks
