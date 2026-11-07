import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import health, ocr
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.extractors.base import OcrEngine
from app.extractors.paddleocr_extractor import PaddleOcrEngine
from app.extractors.tesseract_extractor import TesseractEngine
from app.services.document_service import DocumentProcessor
from app.services.job_service import JobStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MULTIPART_OVERHEAD_BYTES = 1024 * 1024

DESCRIPTION = """
Extracts text from images, PDFs and DOCX files in English, French and Arabic. CPU only.

Use `POST /api/v1/ocr/jobs` then poll `GET /api/v1/ocr/jobs/{job_id}` for long documents,
or `POST /api/v1/ocr/extract` for a synchronous call.
Bounding boxes are `[x_min, y_min, x_max, y_max]` fractions (0..1) of the page, origin top-left. Page numbers start at 1.
"""


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def create_app(settings: Settings | None = None, engines: dict[str, OcrEngine] | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engines = engines or {
        "paddleocr": PaddleOcrEngine(settings.model_cache_dir, settings.paddle_det_max_side),
        "tesseract": TesseractEngine(settings.tesseract_cmd),
    }
    processor = DocumentProcessor(settings, engines)
    jobs = JobStore(processor, settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(title="OCR Platform", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)
    app.state.settings = settings
    app.state.processor = processor
    app.state.jobs = jobs
    app.state.sync_slots = threading.BoundedSemaphore(settings.sync_max_active)

    @app.middleware("http")
    async def reject_oversized_uploads(request: Request, call_next):
        if request.method == "POST" and request.url.path.startswith("/api/"):
            files_allowed = settings.batch_max_files if request.url.path.endswith("/batch") else 1
            limit = settings.max_upload_bytes * files_allowed + MULTIPART_OVERHEAD_BYTES
            declared = request.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > limit:
                return _error_response(413, "FILE_TOO_LARGE", f"Request exceeds the {settings.max_upload_mb} MB upload limit.")
        return await call_next(request)

    # Added last so it is the outermost middleware and its headers also reach the 413 above.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
        )

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, error: AppError):
        return _error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, error: RequestValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'][1:])}: {item['msg']}" for item in error.errors()
        )
        return _error_response(422, "INVALID_REQUEST", details)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_request: Request, error: StarletteHTTPException):
        return _error_response(error.status_code, "HTTP_ERROR", str(error.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, error: Exception):
        logger.exception("Unhandled error")
        return _error_response(500, "INTERNAL_ERROR", "Unexpected server error.")

    app.include_router(health.router)
    app.include_router(ocr.router)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
