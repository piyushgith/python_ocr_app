from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.routes.ocr import get_processor, get_settings
from app.core.config import Settings
from app.services.document_service import DocumentProcessor

router = APIRouter(tags=["system"])

TRACKED_PACKAGES = ["paddlepaddle", "paddleocr", "pymupdf", "python-docx", "pillow"]


class HealthResponse(BaseModel):
    status: str
    version: str
    dependencies: dict[str, str | None]
    ready_ocr_engines: list[str]
    limits: dict[str, int]


@router.get("/health", response_model=HealthResponse, summary="Service status and dependency readiness")
def health(
    processor: DocumentProcessor = Depends(get_processor), settings: Settings = Depends(get_settings)
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=version("ocr-platform"),
        dependencies={package: _installed_version(package) for package in TRACKED_PACKAGES},
        ready_ocr_engines=[engine.name for engine in processor.engines_info().engines if engine.available],
        limits={
            "max_upload_mb": settings.max_upload_mb,
            "max_pages": settings.max_pages,
            "batch_max_files": settings.batch_max_files,
            "processing_timeout_seconds": settings.processing_timeout_seconds,
        },
    )


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
