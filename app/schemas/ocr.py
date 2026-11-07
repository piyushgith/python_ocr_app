from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Mode(str, Enum):
    auto = "auto"
    native = "native"
    ocr = "ocr"


class EngineChoice(str, Enum):
    auto = "auto"
    paddleocr = "paddleocr"
    tesseract = "tesseract"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class ExtractionOptions(BaseModel):
    engine: EngineChoice = EngineChoice.auto
    languages: list[str] = []
    mode: Mode = Mode.auto


class DocumentInfo(BaseModel):
    filename: str
    content_type: str = Field(description="MIME type detected from the file content, not the upload header")
    size_bytes: int
    page_count: int = Field(description="Number of pages or frames. DOCX and TXT count as one page")


class ExtractionInfo(BaseModel):
    engine: str | None = Field(description="OCR engine actually used. null when no page needed OCR")
    requested_engine: EngineChoice
    requested_mode: Mode
    mode: Literal["native", "ocr", "mixed"] = Field(description="How the pages were actually extracted")
    requested_languages: list[str] = Field(description="Languages named in the request. Empty when left to detection")
    detected_languages: list[str] | None = Field(
        default=None,
        description="Languages detected in the extracted text, most used first, when the request named none. "
        "Empty when the text was too short to tell. null when languages were requested",
    )


class Block(BaseModel):
    text: str
    confidence: float | None = Field(description="OCR confidence 0..1. null for native text")
    bbox: list[float] | None = Field(
        description="[x_min, y_min, x_max, y_max] as fractions 0..1 of page width/height, origin top-left"
    )


class Table(BaseModel):
    position: int = Field(description="Zero-based index of the table among the document body elements")
    rows: list[list[str]]


class Page(BaseModel):
    page_number: int = Field(description="One-based")
    method: Literal["native", "ocr"]
    text: str
    native_text: str | None = Field(
        default=None, description="Embedded text layer of a PDF page that was OCR-ed. null when none existed"
    )
    blocks: list[Block]
    tables: list[Table]


class ExtractionResult(BaseModel):
    document: DocumentInfo
    extraction: ExtractionInfo
    pages: list[Page]
    warnings: list[str]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class BatchItem(BaseModel):
    filename: str
    status: Literal["completed", "failed"]
    result: ExtractionResult | None = None
    error: ErrorBody | None = None


class BatchResponse(BaseModel):
    items: list[BatchItem]


class EngineInfo(BaseModel):
    name: str
    available: bool = Field(description="Installed and able to run. Models may still download on first use")
    supported_languages: list[str]
    capabilities: list[str]
    warnings: list[str]


class EnginesResponse(BaseModel):
    engines: list[EngineInfo]
    language_detection: bool
    native_extraction: list[str] = Field(description="Formats with embedded or editable text")
    supported_formats: list[str]
