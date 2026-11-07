import logging
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.jobs import JobStatus
from app.schemas.ocr import ErrorBody, ExtractionOptions, ExtractionResult
from app.services.document_service import DocumentProcessor

logger = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str
    workdir: Path
    filename: str
    options: ExtractionOptions
    created_at: datetime
    status: str = "queued"
    finished_at: datetime | None = None
    pages_done: int = 0
    total_pages: int | None = None
    result: ExtractionResult | None = None
    error: ErrorBody | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class JobStore:
    """Runs extraction jobs one at a time on a background thread. Jobs live in memory and expire after job_ttl_seconds.

    At most job_max_stored jobs are kept; when full, the oldest finished job is dropped to make room.
    """

    def __init__(self, processor: DocumentProcessor, settings: Settings):
        self._processor = processor
        self._settings = settings
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-job")

    def submit(self, workdir: Path, filename: str, options: ExtractionOptions) -> str:
        """Queues the file stored under `workdir` (as upload.bin). The job deletes `workdir` when it finishes."""
        with self._lock:
            self._purge_expired()
            active = sum(1 for job in self._jobs.values() if job.status in ("queued", "processing"))
            if active >= self._settings.job_max_active:
                raise AppError("TOO_MANY_JOBS", "Too many jobs are queued. Retry later.", 429)
            if len(self._jobs) >= self._settings.job_max_stored:
                self._drop_oldest_finished()
            job = Job(
                job_id=str(uuid.uuid4()),
                workdir=workdir,
                filename=filename,
                options=options,
                created_at=datetime.now(timezone.utc),
            )
            self._executor.submit(self._run, job)
            self._jobs[job.job_id] = job
        return job.job_id

    def get(self, job_id: str) -> Job:
        with self._lock:
            self._purge_expired()
            job = self._jobs.get(job_id)
        if job is None:
            raise AppError("JOB_NOT_FOUND", "Job not found. It may have expired.", 404)
        return job

    def status(self, job: Job) -> JobStatus:
        with job._lock:
            return JobStatus(
                job_id=job.job_id,
                status=job.status,
                created_at=job.created_at,
                finished_at=job.finished_at,
                pages_done=job.pages_done,
                total_pages=job.total_pages,
                result=job.result,
                error=job.error,
            )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            for job in self._jobs.values():
                shutil.rmtree(job.workdir, ignore_errors=True)
            self._jobs.clear()

    def _run(self, job: Job) -> None:
        with job._lock:
            job.status = "processing"

        def report_progress(done: int, total: int) -> None:
            with job._lock:
                job.pages_done, job.total_pages = done, total

        outcome = {
            "status": "failed",
            "error": ErrorBody(code="INTERNAL_ERROR", message="Unexpected error while processing the document."),
        }
        try:
            result = self._processor.process(job.workdir / "upload.bin", job.filename, job.options, report_progress)
            outcome = {"status": "completed", "result": result}
        except AppError as error:
            outcome = {"status": "failed", "error": ErrorBody(code=error.code, message=error.message)}
        except Exception:
            logger.exception("Job %s failed unexpectedly", job.job_id)
        finally:
            shutil.rmtree(job.workdir, ignore_errors=True)
            with job._lock:
                job.finished_at = datetime.now(timezone.utc)
                for key, value in outcome.items():
                    setattr(job, key, value)
            with self._lock:
                self._purge_expired()

    def _drop_oldest_finished(self) -> None:
        finished = [job for job in self._jobs.values() if job.finished_at]
        if finished:
            oldest = min(finished, key=lambda job: job.finished_at)
            del self._jobs[oldest.job_id]

    def _purge_expired(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._settings.job_ttl_seconds)
        expired = [job_id for job_id, job in self._jobs.items() if job.finished_at and job.finished_at < cutoff]
        for job_id in expired:
            del self._jobs[job_id]
