from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime limits and paths. Every field can be set with an OCR_-prefixed environment variable."""

    model_config = SettingsConfigDict(env_prefix="OCR_", env_file=".env", extra="ignore")

    max_upload_mb: int = 25
    max_pages: int = 50
    render_dpi: int = 200
    max_image_side: int = 4000
    max_image_pixels: int = 100_000_000
    max_docx_uncompressed_mb: int = 100
    native_min_chars: int = 20
    paddle_det_max_side: int = 1280
    processing_timeout_seconds: int = 600
    batch_max_files: int = 5
    job_ttl_seconds: int = 3600
    job_max_active: int = 20
    job_max_stored: int = 100
    sync_max_active: int = 10
    model_cache_dir: Path = Path("models")
    tesseract_cmd: str = "tesseract"
    cors_origins: list[str] = []
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
