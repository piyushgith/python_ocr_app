"""Generate the sample documents used by the tests and by the README.

Usage: python scripts/make_fixtures.py
Needs the dev extras (arabic-reshaper, python-bidi) and a TTF font with Arabic + Latin glyphs.
Set OCR_FIXTURE_FONT to a font path to override the font search.
"""

import json
import os
import sys
from pathlib import Path

import arabic_reshaper
import docx
import pymupdf
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]

TEXTS = {
    "english": [
        "Loan application number 2024-00123",
        "Applicant: John Smith",
        "Total amount due: $12,500.00",
    ],
    "french": [
        "Relevé de compte bancaire",
        "Échéance le 15 février : 1 250,75 €",
        "Café crème à Noël, où êtes-vous ?",
    ],
    "arabic": [
        "مرحبا بكم في البنك الوطني",
        "كشف حساب شهر يناير",
        "رقم الحساب 123456789",
    ],
    "mixed_arabic_latin": [
        "Invoice رقم 2024 مبلغ 1500 USD",
        "العميل: Ahmed Hassan",
    ],
}

RENDER_AS_ARABIC = {"arabic", "mixed_arabic_latin"}


def find_font() -> str:
    override = os.environ.get("OCR_FIXTURE_FONT")
    candidates = [override] if override else FONT_CANDIDATES
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    sys.exit("No usable font found. Set OCR_FIXTURE_FONT to a TTF with Arabic and Latin glyphs.")


def visual_line(line: str, language: str) -> str:
    """Pillow without libraqm does not shape Arabic, so shape and reorder before drawing."""
    if language in RENDER_AS_ARABIC:
        return get_display(arabic_reshaper.reshape(line))
    return line


def render_lines(lines: list[str], language: str, font_path: str, size: int = 44) -> Image.Image:
    font = ImageFont.truetype(font_path, size)
    line_height = int(size * 1.9)
    image = Image.new("RGB", (1300, line_height * len(lines) + 80), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        text = visual_line(line, language)
        if language in RENDER_AS_ARABIC:
            width = draw.textlength(text, font=font)
            x = image.width - 60 - width
        else:
            x = 60
        draw.text((x, 40 + index * line_height), text, font=font, fill="black")
    return image


def save_scanned_pdf(images: list[Image.Image], path: Path) -> None:
    """Image-only PDF: every page is a picture with no text layer."""
    document = pymupdf.open()
    for image in images:
        page = document.new_page(width=612, height=792)
        image_path = SAMPLES / "_tmp_page.png"
        image.save(image_path)
        page.insert_image(pymupdf.Rect(36, 36, 576, 36 + 540 * image.height / image.width), filename=str(image_path))
        image_path.unlink()
    document.save(path)
    document.close()


def save_searchable_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    y = 100
    for line in TEXTS["french"] + TEXTS["english"]:
        page.insert_text((72, y), line, fontname="helv", fontsize=14)
        y += 28
    document.save(path)
    document.close()


def save_mixed_pdf(images: list[Image.Image], path: Path) -> None:
    """Page 1 has a text layer, page 2 is image-only."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "Native text page: Relevé de compte", fontname="helv", fontsize=14)
    scanned_page = document.new_page(width=612, height=792)
    image_path = SAMPLES / "_tmp_page.png"
    images[0].save(image_path)
    scanned_page.insert_image(
        pymupdf.Rect(36, 36, 576, 36 + 540 * images[0].height / images[0].width), filename=str(image_path)
    )
    image_path.unlink()
    document.save(path)
    document.close()


def save_docx(path: Path) -> None:
    document = docx.Document()
    document.add_paragraph("First paragraph of the loan agreement.")
    table = document.add_table(rows=3, cols=2)
    rows = [("Item", "Amount"), ("Principal", "10000.00"), ("Fee", "250.00")]
    for row, values in zip(table.rows, rows):
        for cell, value in zip(row.cells, values):
            cell.text = value
    document.add_paragraph("Paragraph after the table: Relevé et échéance.")
    document.save(path)


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    font_path = find_font()
    images = {language: render_lines(lines, language, font_path) for language, lines in TEXTS.items()}
    for language, image in images.items():
        image.save(SAMPLES / f"{language}.png")
    save_scanned_pdf([images["english"], images["french"]], SAMPLES / "scanned.pdf")
    save_searchable_pdf(SAMPLES / "searchable.pdf")
    save_mixed_pdf([images["english"]], SAMPLES / "mixed.pdf")
    save_docx(SAMPLES / "document.docx")
    ground_truth = {"texts": TEXTS}
    (SAMPLES / "ground_truth.json").write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Fixtures written to {SAMPLES} using font {font_path}")


if __name__ == "__main__":
    main()
