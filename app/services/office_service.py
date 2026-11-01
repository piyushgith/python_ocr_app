from pathlib import Path

import docx
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph

from app.core.errors import corrupted_document
from app.schemas.ocr import Table


def extract_docx(path: Path) -> tuple[str, list[Table]]:
    """Returns the body text in document order and the tables. Tables are also rendered into the text as TSV rows."""
    try:
        document = docx.Document(str(path))
    except Exception as error:
        raise corrupted_document("The DOCX file is corrupted or cannot be read.") from error

    parts: list[str] = []
    tables: list[Table] = []
    for position, element in enumerate(document.element.body.iterchildren()):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            parts.append(Paragraph(element, document).text)
        elif tag == "tbl":
            rows = _table_rows(DocxTable(element, document))
            tables.append(Table(position=position, rows=rows))
            parts.extend("\t".join(row) for row in rows)
    return "\n".join(parts).strip(), tables


def _table_rows(table: DocxTable) -> list[list[str]]:
    rows = []
    for row in table.rows:
        cells = []
        seen = set()
        for cell in row.cells:
            if cell._tc in seen:
                continue
            seen.add(cell._tc)
            cells.append(cell.text.replace("\n", " ").strip())
        rows.append(cells)
    return rows


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()
