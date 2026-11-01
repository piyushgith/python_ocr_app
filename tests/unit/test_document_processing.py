from pathlib import Path

import docx
import pymupdf
import pytest

from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.ocr import EngineChoice, ExtractionOptions, Mode
from app.services.document_service import DocumentProcessor, render_text
from tests.conftest import FakeEngine, UnavailableTesseract


def make_processor(tmp_path, engine=None, tesseract=None, **overrides) -> tuple[DocumentProcessor, FakeEngine]:
    engine = engine or FakeEngine()
    settings = Settings(model_cache_dir=tmp_path / "models", **overrides)
    return DocumentProcessor(settings, {"paddleocr": engine, "tesseract": tesseract or UnavailableTesseract()}), engine


def options(mode=Mode.auto, engine=EngineChoice.auto, languages=("en",)) -> ExtractionOptions:
    return ExtractionOptions(mode=mode, engine=engine, languages=list(languages))


def text_pdf(path: Path, page_texts: list[str]) -> Path:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 100), text, fontname="helv", fontsize=12)
    document.save(path)
    return path


class TestPdf:
    def test_searchable_pdf_uses_native_text_without_ocr(self, tmp_path):
        processor, engine = make_processor(tmp_path)
        pdf = text_pdf(tmp_path / "a.pdf", ["Relevé de compte: échéance 15 février 2025"])

        result = processor.process(pdf, "a.pdf", options())

        assert engine.calls == []
        assert result.extraction.mode == "native"
        assert result.extraction.engine is None
        assert "Relevé de compte: échéance 15 février 2025" in result.pages[0].text
        assert result.pages[0].blocks[0].confidence is None

    def test_scanned_pdf_is_ocred(self, tmp_path, samples):
        processor, engine = make_processor(tmp_path)

        result = processor.process(samples / "scanned.pdf", "scanned.pdf", options())

        assert len(engine.calls) == 2
        assert result.extraction.mode == "ocr"
        assert result.extraction.engine == "paddleocr"
        assert result.pages[0].text == "FAKE OCR TEXT"
        assert result.pages[0].blocks[0].bbox == pytest.approx([10 / engine.calls[0][0][0], 20 / engine.calls[0][0][1], 110 / engine.calls[0][0][0], 70 / engine.calls[0][0][1]], abs=1e-4)

    def test_mixed_pdf_picks_method_per_page(self, tmp_path, samples):
        processor, engine = make_processor(tmp_path)

        result = processor.process(samples / "mixed.pdf", "mixed.pdf", options())

        assert [page.method for page in result.pages] == ["native", "ocr"]
        assert result.extraction.mode == "mixed"
        assert len(engine.calls) == 1

    def test_forced_ocr_keeps_the_embedded_text(self, tmp_path):
        processor, engine = make_processor(tmp_path)
        pdf = text_pdf(tmp_path / "a.pdf", ["Embedded text that is long enough"])

        result = processor.process(pdf, "a.pdf", options(mode=Mode.ocr))

        assert len(engine.calls) == 1
        assert result.pages[0].method == "ocr"
        assert result.pages[0].text == "FAKE OCR TEXT"
        assert result.pages[0].native_text == "Embedded text that is long enough"

    def test_native_mode_never_runs_ocr_and_warns_for_scanned_pages(self, tmp_path, samples):
        processor, engine = make_processor(tmp_path)

        result = processor.process(samples / "scanned.pdf", "scanned.pdf", options(mode=Mode.native))

        assert engine.calls == []
        assert [page.text for page in result.pages] == ["", ""]
        assert len(result.warnings) == 2
        assert "no embedded text" in result.warnings[0]

    def test_page_with_too_little_text_falls_back_to_ocr_and_keeps_it(self, tmp_path):
        processor, engine = make_processor(tmp_path, native_min_chars=20)
        pdf = text_pdf(tmp_path / "a.pdf", ["tiny"])

        result = processor.process(pdf, "a.pdf", options())

        assert result.pages[0].method == "ocr"
        assert result.pages[0].native_text == "tiny"

    def test_page_limit_rejects_large_pdf(self, tmp_path):
        processor, _ = make_processor(tmp_path, max_pages=3)
        pdf = text_pdf(tmp_path / "big.pdf", ["x"] * 5)

        with pytest.raises(AppError) as error:
            processor.process(pdf, "big.pdf", options())

        assert error.value.code == "PAGE_LIMIT_EXCEEDED"

    def test_corrupted_pdf_is_rejected(self, tmp_path):
        processor, _ = make_processor(tmp_path)
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"%PDF-1.7\nthis is not a real pdf body")

        with pytest.raises(AppError) as error:
            processor.process(broken, "broken.pdf", options())

        assert error.value.code == "CORRUPTED_DOCUMENT"

    def test_timeout_stops_processing(self, tmp_path, samples):
        engine = FakeEngine(delay=0.3)
        processor, _ = make_processor(tmp_path, engine=engine, processing_timeout_seconds=0)

        with pytest.raises(AppError) as error:
            processor.process(samples / "scanned.pdf", "scanned.pdf", options())

        assert error.value.code == "TIMEOUT"

    def test_progress_is_reported_per_page(self, tmp_path, samples):
        processor, _ = make_processor(tmp_path)
        reported = []

        processor.process(samples / "scanned.pdf", "scanned.pdf", options(), lambda done, total: reported.append((done, total)))

        assert reported[0] == (0, 2)
        assert reported[-1] == (2, 2)


class TestImage:
    def test_image_is_ocred_and_pixels_are_not_downscaled_below_limit(self, tmp_path, samples):
        processor, engine = make_processor(tmp_path)

        result = processor.process(samples / "english.png", "english.png", options(languages=("en", "fr")))

        assert engine.calls[0][1] == ["en", "fr"]
        assert result.document.content_type == "image/png"
        assert result.pages[0].page_number == 1

    def test_large_image_is_downscaled(self, tmp_path, samples):
        processor, engine = make_processor(tmp_path, max_image_side=500)

        processor.process(samples / "english.png", "english.png", options())

        assert max(engine.calls[0][0]) == 500

    def test_native_mode_on_image_is_rejected(self, tmp_path, samples):
        processor, _ = make_processor(tmp_path)

        with pytest.raises(AppError) as error:
            processor.process(samples / "english.png", "english.png", options(mode=Mode.native))

        assert error.value.code == "UNSUPPORTED_MODE"

    def test_image_with_too_many_pixels_is_rejected(self, tmp_path, samples):
        processor, _ = make_processor(tmp_path, max_image_pixels=1000)

        with pytest.raises(AppError) as error:
            processor.process(samples / "english.png", "english.png", options())

        assert error.value.code == "IMAGE_TOO_LARGE"

    def test_truncated_image_is_rejected(self, tmp_path, samples):
        processor, _ = make_processor(tmp_path)
        truncated = tmp_path / "cut.png"
        truncated.write_bytes((samples / "english.png").read_bytes()[:3000])

        with pytest.raises(AppError) as error:
            processor.process(truncated, "cut.png", options())

        assert error.value.code == "CORRUPTED_DOCUMENT"

    def test_multi_frame_tiff_gives_one_page_per_frame(self, tmp_path, samples):
        from PIL import Image

        frames = [Image.open(samples / "english.png").convert("RGB"), Image.open(samples / "french.png").convert("RGB")]
        tiff = tmp_path / "multi.tiff"
        frames[0].save(tiff, save_all=True, append_images=frames[1:])
        processor, engine = make_processor(tmp_path)

        result = processor.process(tiff, "multi.tiff", options())

        assert [page.page_number for page in result.pages] == [1, 2]
        assert len(engine.calls) == 2


class TestDocxAndText:
    def test_docx_paragraphs_and_tables_keep_document_order(self, tmp_path, samples):
        processor, engine = make_processor(tmp_path)

        result = processor.process(samples / "document.docx", "document.docx", options())

        assert engine.calls == []
        lines = result.pages[0].text.split("\n")
        assert lines[0] == "First paragraph of the loan agreement."
        assert lines[1:4] == ["Item\tAmount", "Principal\t10000.00", "Fee\t250.00"]
        assert lines[4] == "Paragraph after the table: Relevé et échéance."
        assert result.pages[0].tables[0].rows == [["Item", "Amount"], ["Principal", "10000.00"], ["Fee", "250.00"]]
        assert result.pages[0].tables[0].position == 1

    def test_merged_table_cells_are_not_duplicated(self, tmp_path):
        document = docx.Document()
        table = document.add_table(rows=1, cols=3)
        table.rows[0].cells[0].merge(table.rows[0].cells[1])
        table.rows[0].cells[0].text = "merged"
        table.rows[0].cells[2].text = "last"
        path = tmp_path / "merged.docx"
        document.save(path)
        processor, _ = make_processor(tmp_path)

        result = processor.process(path, "merged.docx", options())

        assert result.pages[0].tables[0].rows == [["merged", "last"]]

    def test_forced_ocr_on_docx_extracts_natively_and_warns(self, tmp_path, samples):
        processor, engine = make_processor(tmp_path)

        result = processor.process(samples / "document.docx", "document.docx", options(mode=Mode.ocr))

        assert engine.calls == []
        assert result.extraction.mode == "native"
        assert "does not apply" in result.warnings[0]

    def test_text_file_is_returned_unchanged(self, tmp_path):
        path = tmp_path / "note.txt"
        path.write_text("Bonjour — مرحبا\nline two", encoding="utf-8")
        processor, _ = make_processor(tmp_path)

        result = processor.process(path, "note.txt", options())

        assert result.pages[0].text == "Bonjour — مرحبا\nline two"

    def test_corrupted_docx_is_rejected(self, tmp_path):
        import zipfile

        path = tmp_path / "broken.docx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", "<not-valid-docx/>")
        processor, _ = make_processor(tmp_path)

        with pytest.raises(AppError) as error:
            processor.process(path, "broken.docx", options())

        assert error.value.code == "CORRUPTED_DOCUMENT"


class TestEngineSelection:
    def test_unavailable_engine_gives_actionable_error(self, tmp_path, samples):
        processor, _ = make_processor(tmp_path)

        with pytest.raises(AppError) as error:
            processor.process(samples / "english.png", "e.png", options(engine=EngineChoice.tesseract))

        assert error.value.code == "ENGINE_UNAVAILABLE"
        assert "Tesseract binary" in error.value.message

    def test_auto_falls_back_to_tesseract_and_reports_it(self, tmp_path, samples):
        broken_paddle = FakeEngine(languages=frozenset(), problems=["PaddleOCR is not installed."])
        tesseract = FakeEngine(text="TESS")
        tesseract.name = "tesseract"
        processor, _ = make_processor(tmp_path, engine=broken_paddle, tesseract=tesseract)

        result = processor.process(samples / "english.png", "e.png", options())

        assert result.extraction.engine == "tesseract"
        assert result.extraction.requested_engine == EngineChoice.auto
        assert "Engine 'auto' used 'tesseract' because paddleocr: PaddleOCR is not installed." in result.warnings[0]

    def test_auto_with_no_usable_engine_fails(self, tmp_path, samples):
        broken_paddle = FakeEngine(languages=frozenset(), problems=["PaddleOCR is not installed."])
        processor, _ = make_processor(tmp_path, engine=broken_paddle)

        with pytest.raises(AppError) as error:
            processor.process(samples / "english.png", "e.png", options())

        assert error.value.code == "ENGINE_UNAVAILABLE"

    def test_engine_without_language_support_is_rejected(self, tmp_path, samples):
        processor, _ = make_processor(tmp_path, engine=FakeEngine(languages=frozenset({"en"})))

        with pytest.raises(AppError) as error:
            processor.process(samples / "english.png", "e.png", options(engine=EngineChoice.paddleocr, languages=("en", "ar")))

        assert error.value.code == "UNSUPPORTED_LANGUAGE"
        assert "ar" in error.value.message

    def test_native_pdf_does_not_need_any_engine(self, tmp_path):
        processor, _ = make_processor(tmp_path, engine=FakeEngine(problems=["missing"]))
        pdf = text_pdf(tmp_path / "a.pdf", ["Plenty of embedded text here"])

        result = processor.process(pdf, "a.pdf", options())

        assert result.pages[0].method == "native"

    def test_every_page_failing_is_an_error(self, tmp_path, samples):
        class FailingEngine(FakeEngine):
            def recognize(self, image, languages):
                raise AppError("OCR_FAILED", "boom", 500)

        processor, _ = make_processor(tmp_path, engine=FailingEngine())

        with pytest.raises(AppError) as error:
            processor.process(samples / "scanned.pdf", "s.pdf", options())

        assert error.value.code == "OCR_FAILED"

    def test_one_failing_page_becomes_a_warning(self, tmp_path, samples):
        class FlakyEngine(FakeEngine):
            def recognize(self, image, languages):
                if not self.calls:
                    self.calls.append(("seen", languages))
                    raise AppError("OCR_FAILED", "boom", 500)
                return super().recognize(image, languages)

        processor, _ = make_processor(tmp_path, engine=FlakyEngine())

        result = processor.process(samples / "scanned.pdf", "s.pdf", options())

        assert result.pages[0].text == ""
        assert result.pages[1].text == "FAKE OCR TEXT"
        assert result.warnings == ["Page 1: boom"]


def test_text_rendering_has_predictable_page_separators(tmp_path, samples):
    processor, _ = make_processor(tmp_path)

    multi = render_text(processor.process(samples / "scanned.pdf", "s.pdf", options()))
    single = render_text(processor.process(samples / "english.png", "e.png", options()))

    assert multi == "--- Page 1 ---\nFAKE OCR TEXT\n\n--- Page 2 ---\nFAKE OCR TEXT"
    assert single == "FAKE OCR TEXT"
