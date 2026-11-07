import codecs
import re
import zipfile
from enum import Enum
from pathlib import Path

from fastapi import UploadFile

from app.core.errors import AppError, corrupted_document, unsupported_format

READ_CHUNK_BYTES = 1024 * 1024


class DocFormat(str, Enum):
    png = "png"
    jpeg = "jpeg"
    tiff = "tiff"
    bmp = "bmp"
    webp = "webp"
    pdf = "pdf"
    docx = "docx"
    txt = "txt"


MIME_TYPES = {
    DocFormat.png: "image/png",
    DocFormat.jpeg: "image/jpeg",
    DocFormat.tiff: "image/tiff",
    DocFormat.bmp: "image/bmp",
    DocFormat.webp: "image/webp",
    DocFormat.pdf: "application/pdf",
    DocFormat.docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    DocFormat.txt: "text/plain",
}

IMAGE_FORMATS = {DocFormat.png, DocFormat.jpeg, DocFormat.tiff, DocFormat.bmp, DocFormat.webp}

SUPPORTED_FORMAT_LABELS = ["png", "jpg/jpeg", "tiff", "bmp", "webp", "pdf", "docx", "txt"]

BMP_DIB_HEADER_SIZES = {12, 40, 52, 56, 64, 108, 124}

_UNSAFE_FILENAME_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_filename(raw_name: str | None) -> str:
    """Returns the base name only, without path components or control characters."""
    name = (raw_name or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = _UNSAFE_FILENAME_CHARS.sub("", name).strip()
    return name[:255] or "document"


async def save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    """Streams an upload to `destination`, rejecting empty files and files over `max_bytes`."""
    written = 0
    with destination.open("wb") as target:
        while chunk := await upload.read(READ_CHUNK_BYTES):
            written += len(chunk)
            if written > max_bytes:
                raise AppError(
                    "FILE_TOO_LARGE", f"File exceeds the maximum upload size of {max_bytes // (1024 * 1024)} MB.", 413
                )
            target.write(chunk)
    if written == 0:
        raise AppError("EMPTY_FILE", "The uploaded file is empty.", 400)
    return written


def detect_format(path: Path, max_docx_uncompressed_bytes: int) -> DocFormat:
    """Identifies the format from file content. The extension and client content type are ignored."""
    with path.open("rb") as source:
        header = source.read(1024)

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return DocFormat.png
    if header.startswith(b"\xff\xd8\xff"):
        return DocFormat.jpeg
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return DocFormat.tiff
    if header.startswith(b"BM") and int.from_bytes(header[14:18], "little") in BMP_DIB_HEADER_SIZES:
        return DocFormat.bmp
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return DocFormat.webp
    if header.removeprefix(b"\xef\xbb\xbf").lstrip().startswith(b"%PDF-"):
        return DocFormat.pdf
    if header.startswith(b"PK\x03\x04"):
        return _check_docx(path, max_docx_uncompressed_bytes)
    if _is_utf8_text(path, header):
        return DocFormat.txt
    # Readers accept a PDF header anywhere in the first 1 KB; text that merely quotes it was matched above.
    if b"%PDF-" in header:
        return DocFormat.pdf
    raise unsupported_format(
        "Unsupported file type. Supported: " + ", ".join(SUPPORTED_FORMAT_LABELS) + "."
    )


def _check_docx(path: Path, max_uncompressed_bytes: int) -> DocFormat:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = {entry.filename for entry in entries}
            if "word/document.xml" not in names:
                raise unsupported_format("ZIP-based file is not a DOCX document.")
            if sum(entry.file_size for entry in entries) > max_uncompressed_bytes:
                raise AppError("FILE_TOO_LARGE", "DOCX content is too large once decompressed.", 413)
    except zipfile.BadZipFile as error:
        raise corrupted_document("The DOCX file is corrupted.") from error
    return DocFormat.docx


def _is_utf8_text(path: Path, header: bytes) -> bool:
    if b"\x00" in header:
        return False
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    try:
        with path.open("rb") as source:
            while chunk := source.read(READ_CHUNK_BYTES):
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True
