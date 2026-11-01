from pathlib import Path

import pymupdf
from PIL import Image

from app.core.config import Settings
from app.core.errors import AppError, corrupted_document

PDF_POINTS_PER_INCH = 72


def open_pdf(path: Path, settings: Settings) -> pymupdf.Document:
    """Opens a PDF and enforces the page limit. Caller closes it."""
    try:
        document = pymupdf.open(path)
    except Exception as error:
        raise corrupted_document("The PDF file is corrupted or cannot be read.") from error
    if document.needs_pass:
        document.close()
        raise AppError("ENCRYPTED_DOCUMENT", "The PDF is password protected.", 422)
    if document.page_count == 0:
        document.close()
        raise corrupted_document("The PDF contains no pages.")
    if document.page_count > settings.max_pages:
        page_count = document.page_count
        document.close()
        raise AppError(
            "PAGE_LIMIT_EXCEEDED",
            f"The PDF has {page_count} pages; the limit is {settings.max_pages}.",
            413,
        )
    return document


def native_page_blocks(document: pymupdf.Document, page_index: int) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Text blocks of the embedded text layer as (text, normalized bbox), in reading order."""
    try:
        page = document.load_page(page_index)
        rect = page.rect
        raw_blocks = page.get_text("blocks", sort=True)
    except Exception as error:
        raise corrupted_document(f"PDF page {page_index + 1} could not be read.") from error

    blocks = []
    for x_min, y_min, x_max, y_max, text, _block_number, block_type in raw_blocks:
        text = text.strip()
        if block_type != 0 or not text:
            continue
        bbox = (
            (x_min - rect.x0) / rect.width,
            (y_min - rect.y0) / rect.height,
            (x_max - rect.x0) / rect.width,
            (y_max - rect.y0) / rect.height,
        )
        blocks.append((text, bbox))
    return blocks


def render_page(document: pymupdf.Document, page_index: int, settings: Settings) -> Image.Image:
    """Renders one page to RGB at render_dpi, shrunk so the longest side is at most max_image_side."""
    try:
        page = document.load_page(page_index)
        scale = settings.render_dpi / PDF_POINTS_PER_INCH
        longest_side = max(page.rect.width, page.rect.height) * scale
        if longest_side > settings.max_image_side:
            scale *= settings.max_image_side / longest_side
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    except Exception as error:
        raise corrupted_document(f"PDF page {page_index + 1} could not be rendered.") from error
