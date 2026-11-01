import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse

from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.jobs import JobStatus, JobSubmitted
from app.schemas.ocr import (
    BatchItem,
    BatchResponse,
    EngineChoice,
    EnginesResponse,
    ErrorBody,
    ErrorResponse,
    ExtractionOptions,
    ExtractionResult,
    Mode,
    OutputFormat,
)
from app.services.document_service import DocumentProcessor, normalize_languages, render_text
from app.services.job_service import JobStore
from app.utils.files import detect_format, sanitize_filename, save_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ocr", tags=["ocr"])

ERROR_RESPONSES = {
    code: {"model": ErrorResponse, "description": description}
    for code, description in {
        400: "Empty file or invalid request",
        404: "Job not found",
        409: "Job not completed",
        413: "File, page count or image size over the configured limit",
        415: "Unsupported file type",
        422: "Corrupted document, unsupported language or mode, invalid options",
        429: "Too many active jobs",
        500: "OCR failed",
        503: "OCR engine or model unavailable",
        504: "Processing timeout",
    }.items()
}

MODE_HELP = (
    "auto: native text where a PDF page has enough embedded text, OCR otherwise. "
    "native: embedded/editable text only, never OCR. "
    "ocr: always OCR images and PDF pages (a PDF page's embedded text is kept in `native_text`)."
)
LANGUAGES_HELP = "Language codes en, fr, ar. Repeat the field or comma-separate (en,fr). Default: en."


def get_processor(request: Request) -> DocumentProcessor:
    return request.app.state.processor


def get_jobs(request: Request) -> JobStore:
    return request.app.state.jobs


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def extraction_options(
    engine: EngineChoice = Form(EngineChoice.auto, description="OCR engine. auto = paddleocr, else tesseract if installed."),
    languages: list[str] = Form(["en"], description=LANGUAGES_HELP),
    mode: Mode = Form(Mode.auto, description=MODE_HELP),
) -> ExtractionOptions:
    return ExtractionOptions(engine=engine, languages=languages, mode=mode)


@router.get("/engines", response_model=EnginesResponse, summary="Installed OCR engines and the languages they can use")
def list_engines(processor: DocumentProcessor = Depends(get_processor)) -> EnginesResponse:
    return processor.engines_info()


@router.post(
    "/extract",
    response_model=ExtractionResult,
    responses={**ERROR_RESPONSES, 200: {"content": {"text/plain": {}}, "description": "JSON result, or text/plain"}},
    summary="Extract one document synchronously",
)
async def extract(
    file: UploadFile = File(..., description="PNG, JPEG, TIFF, BMP, WEBP, PDF, DOCX or TXT"),
    output_format: OutputFormat = Form(OutputFormat.json, description="json: structured result. text: plain text."),
    options: ExtractionOptions = Depends(extraction_options),
    processor: DocumentProcessor = Depends(get_processor),
    settings: Settings = Depends(get_settings),
):
    with tempfile.TemporaryDirectory(prefix="ocr-") as workdir:
        path = Path(workdir) / "upload.bin"
        await save_upload(file, path, settings.max_upload_bytes)
        result = await run_in_threadpool(processor.process, path, sanitize_filename(file.filename), options)
    if output_format == OutputFormat.text:
        return PlainTextResponse(
            render_text(result),
            headers={"X-OCR-Engine": result.extraction.engine or "none", "X-OCR-Warning-Count": str(len(result.warnings))},
        )
    return result


@router.post(
    "/batch",
    response_model=BatchResponse,
    responses=ERROR_RESPONSES,
    summary="Extract several documents; each file succeeds or fails on its own",
)
async def batch(
    files: list[UploadFile] = File(..., description="Up to OCR_BATCH_MAX_FILES files"),
    options: ExtractionOptions = Depends(extraction_options),
    processor: DocumentProcessor = Depends(get_processor),
    settings: Settings = Depends(get_settings),
) -> BatchResponse:
    if len(files) > settings.batch_max_files:
        raise AppError("BATCH_TOO_LARGE", f"At most {settings.batch_max_files} files per batch request.", 413)
    normalize_languages(options.languages)

    items = []
    for upload in files:
        filename = sanitize_filename(upload.filename)
        with tempfile.TemporaryDirectory(prefix="ocr-") as workdir:
            path = Path(workdir) / "upload.bin"
            try:
                await save_upload(upload, path, settings.max_upload_bytes)
                result = await run_in_threadpool(processor.process, path, filename, options)
                items.append(BatchItem(filename=filename, status="completed", result=result))
            except AppError as error:
                items.append(
                    BatchItem(filename=filename, status="failed", error=ErrorBody(code=error.code, message=error.message))
                )
            except Exception:
                logger.exception("Batch item failed unexpectedly")
                items.append(
                    BatchItem(
                        filename=filename,
                        status="failed",
                        error=ErrorBody(code="INTERNAL_ERROR", message="Unexpected error while processing the document."),
                    )
                )
    return BatchResponse(items=items)


@router.post(
    "/jobs",
    status_code=202,
    response_model=JobSubmitted,
    responses=ERROR_RESPONSES,
    summary="Submit a document for asynchronous extraction and get a job id (transaction UUID)",
)
async def submit_job(
    request: Request,
    file: UploadFile = File(..., description="PNG, JPEG, TIFF, BMP, WEBP, PDF, DOCX or TXT"),
    options: ExtractionOptions = Depends(extraction_options),
    jobs: JobStore = Depends(get_jobs),
    settings: Settings = Depends(get_settings),
) -> JobSubmitted:
    workdir = Path(tempfile.mkdtemp(prefix="ocr-job-"))
    try:
        path = workdir / "upload.bin"
        await save_upload(file, path, settings.max_upload_bytes)
        detect_format(path, settings.max_docx_uncompressed_mb * 1024 * 1024)
        normalize_languages(options.languages)
        job_id = jobs.submit(workdir, sanitize_filename(file.filename), options)
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    return JobSubmitted(job_id=job_id, status="queued", status_url=f"{request.url.path}/{job_id}")


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    responses=ERROR_RESPONSES,
    summary="Poll a job. `result` is set when status is completed, `error` when failed",
)
def get_job(job_id: str, jobs: JobStore = Depends(get_jobs)) -> JobStatus:
    return jobs.status(jobs.get(job_id))


@router.get(
    "/jobs/{job_id}/text",
    response_class=PlainTextResponse,
    responses=ERROR_RESPONSES,
    summary="Plain text of a completed job",
)
def get_job_text(job_id: str, jobs: JobStore = Depends(get_jobs)) -> PlainTextResponse:
    status = jobs.status(jobs.get(job_id))
    if status.status != "completed":
        raise AppError("JOB_NOT_COMPLETED", f"Job status is '{status.status}'; text is available once completed.", 409)
    return PlainTextResponse(render_text(status.result))
