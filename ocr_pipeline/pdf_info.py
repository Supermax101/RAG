from pathlib import Path
from typing import Optional
from pypdf import PdfReader


def get_pdf_page_count(pdf_path: Path) -> Optional[int]:
    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        return None
