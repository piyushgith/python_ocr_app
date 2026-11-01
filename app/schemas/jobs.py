from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.schemas.ocr import ErrorBody, ExtractionResult


class JobSubmitted(BaseModel):
    job_id: str
    status: Literal["queued"]
    status_url: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    created_at: datetime
    finished_at: datetime | None = None
    pages_done: int = 0
    total_pages: int | None = None
    result: ExtractionResult | None = None
    error: ErrorBody | None = None
