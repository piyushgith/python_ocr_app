import logging
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

from PIL import Image

from app.core.config import Settings
from app.core.errors import AppError
from app.extractors.base import OcrBlock, OcrEngine
from app.schemas.ocr import (
    Block,
    DocumentInfo,
    EngineChoice,
    EngineInfo,
    EnginesResponse,
    ExtractionInfo,
    ExtractionOptions,
    ExtractionResult,
    Mode,
    Page,
    Table,
)
from app.services import image_service, office_service, pdf_service
from app.services.language_service import detect_languages, language_letter_counts
from app.utils.files import IMAGE_FORMATS, MIME_TYPES, SUPPORTED_FORMAT_LABELS, DocFormat, detect_format

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ("en", "fr", "ar")
AUTO_ENGINE_ORDER = ("paddleocr", "tesseract")
FATAL_PAGE_ERROR_CODES = {"MODEL_UNAVAILABLE", "TIMEOUT"}

ProgressCallback = Callable[[int, int], None]


def normalize_languages(requested: list[str]) -> list[str]:
    """Lower-cases, splits comma-separated values, removes duplicates. Empty means detect the language."""
    languages: list[str] = []
    for value in requested:
        for code in value.split(","):
            code = code.strip().lower()
            if code and code not in languages:
                languages.append(code)
    unknown = [code for code in languages if code not in SUPPORTED_LANGUAGES]
    if unknown:
        raise AppError(
            "UNSUPPORTED_LANGUAGE",
            f"Unsupported language(s): {', '.join(unknown)}. Supported: {', '.join(SUPPORTED_LANGUAGES)}.",
            422,
        )
    return languages


def render_text(result: ExtractionResult) -> str:
    """Plain text of a result. Multi-page documents get a '--- Page N ---' line before each page."""
    if len(result.pages) == 1:
        return result.pages[0].text
    return "\n\n".join(f"--- Page {page.page_number} ---\n{page.text}" for page in result.pages)


class _OcrSession:
    """Resolves the OCR engine on first need, picks the OCR languages and counts page failures for one document.

    With no requested languages, pages are read with the engine's detection languages. When the detected language
    needs another model, that page is re-read with it and later pages use it directly.
    """

    def __init__(
        self,
        processor: "DocumentProcessor",
        options: ExtractionOptions,
        languages: list[str],
        warnings: list[str],
        deadline: float,
    ):
        self._processor = processor
        self._options = options
        self._languages = languages
        self._warnings = warnings
        self._deadline = deadline
        self._engine: OcrEngine | None = None
        self.pages_attempted = 0
        self.pages_failed = 0
        self.last_error: AppError | None = None

    @property
    def engine(self) -> OcrEngine:
        if self._engine is None:
            self._engine = self._processor.resolve_engine(self._options.engine, self._languages, self._warnings)
        return self._engine

    def ensure_engine(self) -> None:
        """Resolves the engine now so an unusable engine fails before any page work."""
        _ = self.engine

    @property
    def engine_name(self) -> str | None:
        return self._engine.name if self._engine else None

    def recognize_page(self, image: Image.Image, page_number: int) -> Page:
        self.pages_attempted += 1
        try:
            raw_blocks = self._recognize(image)
        except AppError as error:
            if error.code in FATAL_PAGE_ERROR_CODES or error.code == "ENGINE_UNAVAILABLE":
                raise
            self.pages_failed += 1
            self.last_error = error
            self._warnings.append(f"Page {page_number}: {error.message}")
            return Page(page_number=page_number, method="ocr", text="", blocks=[], tables=[])
        blocks = [_normalized_block(block, image.size) for block in raw_blocks]
        return Page(
            page_number=page_number,
            method="ocr",
            text="\n".join(block.text for block in blocks),
            blocks=blocks,
            tables=[],
        )

    def _recognize(self, image: Image.Image) -> list[OcrBlock]:
        if self._languages:
            return self._run_engine(image, self._languages)
        probe_languages = self.engine.detection_languages()
        raw_blocks = self._run_engine(image, probe_languages)
        detected = detect_languages("\n".join(block.text for block in raw_blocks))
        usable = [language for language in detected if language in self.engine.supported_languages]
        if usable and self.engine.needs_reread(probe_languages, usable):
            self._languages = usable
            raw_blocks = self._run_engine(image, usable)
        return raw_blocks

    def _run_engine(self, image: Image.Image, languages: list[str]) -> list[OcrBlock]:
        """Runs OCR on a worker thread so a slow page or model download cannot outlive the processing deadline."""
        future = self._processor.page_workers.submit(self.engine.recognize, image, languages)
        try:
            return future.result(timeout=max(self._deadline - time.monotonic(), 0))
        except FuturesTimeoutError:
            future.cancel()
            raise AppError("TIMEOUT", "Processing exceeded the configured time limit.", 504) from None


def _clamp(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 4)


def _normalized_block(block: OcrBlock, image_size: tuple[int, int]) -> Block:
    width, height = image_size
    x_min, y_min, x_max, y_max = block.bbox
    bbox = [_clamp(x_min / width), _clamp(y_min / height), _clamp(x_max / width), _clamp(y_max / height)]
    confidence = None if block.confidence is None else round(block.confidence, 4)
    return Block(text=block.text, confidence=confidence, bbox=bbox)


class DocumentProcessor:
    """Turns an uploaded file into an ExtractionResult, choosing native extraction or OCR per page."""

    def __init__(self, settings: Settings, engines: dict[str, OcrEngine]):
        self._settings = settings
        self._engines = engines
        # One worker per synchronous request plus the job worker; a timed-out page keeps its thread until it ends.
        self.page_workers = ThreadPoolExecutor(max_workers=settings.sync_max_active + 2, thread_name_prefix="ocr-page")

    def engines_info(self) -> EnginesResponse:
        infos = []
        for engine in self._engines.values():
            problems = engine.problems()
            supported = sorted(engine.supported_languages)
            warnings = list(problems) + engine.notes()
            missing = [language for language in SUPPORTED_LANGUAGES if language not in supported]
            if not problems and missing:
                warnings.append(f"Not available for: {', '.join(missing)}.")
            infos.append(
                EngineInfo(
                    name=engine.name,
                    available=not problems,
                    supported_languages=supported,
                    capabilities=engine.capabilities,
                    warnings=warnings,
                )
            )
        return EnginesResponse(
            engines=infos,
            language_detection=True,
            native_extraction=["pdf", "docx", "txt"],
            supported_formats=SUPPORTED_FORMAT_LABELS,
        )

    def resolve_engine(self, choice: EngineChoice, languages: list[str], warnings: list[str]) -> OcrEngine:
        if choice != EngineChoice.auto:
            engine = self._engines[choice.value]
            reason = self._unusable_reason(engine, languages)
            if reason:
                code = "UNSUPPORTED_LANGUAGE" if not engine.problems() else "ENGINE_UNAVAILABLE"
                raise AppError(code, f"Engine '{engine.name}' cannot be used: {reason}", 422 if code == "UNSUPPORTED_LANGUAGE" else 503)
            return engine

        skipped = []
        for name in AUTO_ENGINE_ORDER:
            engine = self._engines[name]
            reason = self._unusable_reason(engine, languages)
            if reason:
                skipped.append(f"{name}: {reason}")
                continue
            if skipped:
                warnings.append(f"Engine 'auto' used '{name}' because " + " | ".join(skipped))
            return engine
        raise AppError("ENGINE_UNAVAILABLE", "No OCR engine can process this request. " + " | ".join(skipped), 503)

    @staticmethod
    def _unusable_reason(engine: OcrEngine, languages: list[str]) -> str | None:
        problems = engine.problems()
        if problems:
            return problems[0]
        unsupported = [language for language in languages if language not in engine.supported_languages]
        if unsupported:
            return f"language(s) {', '.join(unsupported)} not available. Available: {', '.join(sorted(engine.supported_languages))}."
        return None

    def process(
        self,
        path: Path,
        filename: str,
        options: ExtractionOptions,
        progress: ProgressCallback | None = None,
    ) -> ExtractionResult:
        started = time.monotonic()
        languages = normalize_languages(options.languages)
        doc_format = detect_format(path, self._settings.max_docx_uncompressed_mb * 1024 * 1024)
        deadline = started + self._settings.processing_timeout_seconds
        warnings: list[str] = []
        session = _OcrSession(self, options, languages, warnings, deadline)

        if doc_format in IMAGE_FORMATS:
            pages = self._process_image(path, options, session, deadline, progress)
        elif doc_format == DocFormat.pdf:
            pages = self._process_pdf(path, options, session, warnings, deadline, progress)
        else:
            pages = self._process_editable(path, doc_format, options, warnings, progress)

        if session.pages_attempted and session.pages_failed == session.pages_attempted:
            raise AppError("OCR_FAILED", f"OCR failed on every page. {session.last_error.message}", 500)

        methods = {page.method for page in pages}
        detected_languages = None if languages else _document_languages(pages)
        result = ExtractionResult(
            document=DocumentInfo(
                filename=filename,
                content_type=MIME_TYPES[doc_format],
                size_bytes=path.stat().st_size,
                page_count=len(pages),
            ),
            extraction=ExtractionInfo(
                engine=session.engine_name,
                requested_engine=options.engine,
                requested_mode=options.mode,
                mode=methods.pop() if len(methods) == 1 else "mixed",
                requested_languages=languages,
                detected_languages=detected_languages,
            ),
            pages=pages,
            warnings=warnings,
        )
        logger.info(
            "Processed document: format=%s pages=%d mode=%s engine=%s detected_languages=%s seconds=%.1f",
            doc_format.value,
            len(pages),
            result.extraction.mode,
            result.extraction.engine,
            detected_languages,
            time.monotonic() - started,
        )
        return result

    def _process_image(self, path, options, session, deadline, progress) -> list[Page]:
        if options.mode == Mode.native:
            raise AppError(
                "UNSUPPORTED_MODE",
                "Mode 'native' needs embedded or editable text, which images do not have. Use mode 'auto' or 'ocr'.",
                422,
            )
        session.ensure_engine()
        image = image_service.open_image(path, self._settings)
        try:
            frames = image_service.frame_count(image)
            if frames > self._settings.max_pages:
                raise AppError(
                    "PAGE_LIMIT_EXCEEDED", f"The image has {frames} frames; the limit is {self._settings.max_pages}.", 413
                )
            pages = []
            for index in range(frames):
                self._check_deadline(deadline)
                _report(progress, index, frames)
                frame = image_service.load_frame(image, index, self._settings)
                pages.append(session.recognize_page(frame, index + 1))
            _report(progress, frames, frames)
            return pages
        finally:
            image.close()

    def _process_pdf(self, path, options, session, warnings, deadline, progress) -> list[Page]:
        document = pdf_service.open_pdf(path, self._settings)
        try:
            total = document.page_count
            native_pages = [pdf_service.native_page_blocks(document, index) for index in range(total)]
            native_texts = ["\n".join(text for text, _ in blocks) for blocks in native_pages]
            needs_ocr = [
                options.mode == Mode.ocr
                or (options.mode == Mode.auto and len(text.strip()) < self._settings.native_min_chars)
                for text in native_texts
            ]
            if any(needs_ocr):
                session.ensure_engine()
            pages = []
            for index in range(total):
                self._check_deadline(deadline)
                _report(progress, index, total)
                page_number = index + 1
                native_blocks = native_pages[index]
                native_text = native_texts[index]

                if not needs_ocr[index]:
                    if not native_text:
                        warnings.append(f"Page {page_number} has no embedded text. Use mode 'auto' or 'ocr' to read it.")
                    pages.append(
                        Page(
                            page_number=page_number,
                            method="native",
                            text=native_text,
                            blocks=[Block(text=text, confidence=None, bbox=[_clamp(v) for v in bbox]) for text, bbox in native_blocks],
                            tables=[],
                        )
                    )
                    continue

                image = pdf_service.render_page(document, index, self._settings)
                page = session.recognize_page(image, page_number)
                page.native_text = native_text or None
                pages.append(page)
            _report(progress, total, total)
            return pages
        finally:
            document.close()

    def _process_editable(self, path, doc_format, options, warnings, progress) -> list[Page]:
        if options.mode == Mode.ocr:
            warnings.append(
                f"Mode 'ocr' does not apply to {doc_format.value.upper()} files; their text was extracted natively."
            )
        _report(progress, 0, 1)
        if doc_format == DocFormat.docx:
            text, tables = office_service.extract_docx(path)
        else:
            text, tables = office_service.read_text_file(path), []
        _report(progress, 1, 1)
        return [Page(page_number=1, method="native", text=text, blocks=[], tables=tables)]

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        if time.monotonic() > deadline:
            raise AppError("TIMEOUT", "Processing exceeded the configured time limit.", 504)


def _document_languages(pages: list[Page]) -> list[str]:
    """Languages detected page by page, the language covering the most letters first."""
    totals: Counter[str] = Counter()
    for page in pages:
        totals.update(language_letter_counts(page.text))
    return [code for code, _ in totals.most_common()]


def _report(progress: ProgressCallback | None, done: int, total: int) -> None:
    if progress:
        progress(done, total)
