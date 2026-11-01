import zipfile

import pytest

from app.core.errors import AppError
from app.extractors.paddleocr_extractor import PaddleOcrEngine
from app.extractors.tesseract_extractor import TesseractEngine, parse_tsv
from app.services.document_service import normalize_languages
from app.utils.files import DocFormat, detect_format, sanitize_filename

LIMIT = 100 * 1024 * 1024


@pytest.mark.parametrize(
    "sample, expected",
    [
        ("english.png", DocFormat.png),
        ("scanned.pdf", DocFormat.pdf),
        ("document.docx", DocFormat.docx),
    ],
)
def test_detects_format_from_content(samples, sample, expected):
    assert detect_format(samples / sample, LIMIT) == expected


def test_extension_is_ignored(tmp_path, samples):
    disguised = tmp_path / "invoice.pdf"
    disguised.write_bytes((samples / "english.png").read_bytes())

    assert detect_format(disguised, LIMIT) == DocFormat.png


@pytest.mark.parametrize(
    "header, expected",
    [
        (b"\xff\xd8\xff\xe0" + b"0" * 20, DocFormat.jpeg),
        (b"II*\x00" + b"0" * 20, DocFormat.tiff),
        (b"MM\x00*" + b"0" * 20, DocFormat.tiff),
        (b"BM" + b"0" * 20, DocFormat.bmp),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", DocFormat.webp),
    ],
)
def test_detects_other_image_signatures(tmp_path, header, expected):
    path = tmp_path / "f.bin"
    path.write_bytes(header)

    assert detect_format(path, LIMIT) == expected


def test_binary_garbage_is_unsupported(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"\x00\x01\x02garbage")

    with pytest.raises(AppError) as error:
        detect_format(path, LIMIT)

    assert error.value.code == "UNSUPPORTED_FORMAT"


def test_non_docx_zip_is_unsupported(tmp_path):
    path = tmp_path / "sheet.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", "<x/>")

    with pytest.raises(AppError) as error:
        detect_format(path, LIMIT)

    assert error.value.code == "UNSUPPORTED_FORMAT"


def test_docx_that_decompresses_too_large_is_rejected(tmp_path):
    path = tmp_path / "bomb.docx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "a" * 10_000)

    with pytest.raises(AppError) as error:
        detect_format(path, 1000)

    assert error.value.code == "FILE_TOO_LARGE"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\statement.pdf", "statement.pdf"),
        ("bad\x00name\n.png", "badname.png"),
        ("", "document"),
        (None, "document"),
        ("كشف حساب.pdf", "كشف حساب.pdf"),
    ],
)
def test_filename_is_reduced_to_a_safe_base_name(raw, expected):
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize(
    "requested, expected",
    [
        ([], ["en"]),
        (["en,fr"], ["en", "fr"]),
        (["FR", "fr", " ar "], ["fr", "ar"]),
    ],
)
def test_language_normalization(requested, expected):
    assert normalize_languages(requested) == expected


def test_unknown_language_is_rejected():
    with pytest.raises(AppError) as error:
        normalize_languages(["en", "de"])

    assert error.value.code == "UNSUPPORTED_LANGUAGE"


@pytest.mark.parametrize(
    "languages, expected",
    [
        (["en"], "en"),
        (["fr"], "fr"),
        (["en", "fr"], "fr"),
        (["ar"], "ar"),
        (["en", "ar"], "ar"),
        (["fr", "en", "ar"], "ar"),
    ],
)
def test_paddle_language_selection(languages, expected):
    assert PaddleOcrEngine._paddle_lang(languages) == expected


def test_paddle_engine_reports_missing_models_without_loading_them(tmp_path):
    engine = PaddleOcrEngine(tmp_path / "empty-models")

    assert "first OCR request downloads them" in engine.notes()[0]
    assert engine._pipelines == {}


def test_tesseract_tsv_is_grouped_into_lines():
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = [
        "4\t1\t1\t1\t1\t0\t10\t20\t200\t30\t-1\t",
        "5\t1\t1\t1\t1\t1\t10\t20\t80\t30\t96.0\tHello",
        "5\t1\t1\t1\t1\t2\t100\t22\t110\t28\t90.0\tWorld",
        "5\t1\t1\t1\t2\t1\t10\t60\t50\t30\t-1\t",
        "5\t1\t1\t1\t2\t2\t70\t60\t90\t30\t80.0\tBonjour",
    ]

    blocks = parse_tsv("\n".join([header, *rows]))

    assert [block.text for block in blocks] == ["Hello World", "Bonjour"]
    assert blocks[0].confidence == pytest.approx(0.93)
    assert blocks[0].bbox == (10.0, 20.0, 210.0, 50.0)


def test_missing_tesseract_binary_gives_install_instructions():
    engine = TesseractEngine("tesseract-binary-that-does-not-exist")

    assert "apt install tesseract-ocr" in engine.problems()[0]
    assert engine.supported_languages == frozenset()
