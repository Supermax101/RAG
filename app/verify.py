from pathlib import Path
from typing import Tuple

from .config import PARSED_DIR


def verify_doc_outputs(doc_id: str) -> Tuple[bool, str]:
    base = PARSED_DIR / doc_id
    if not base.exists():
        return False, f"missing dir: {base}"
    for rel in ["full.md", "full.Rmd", "doc.json", "pages"]:
        p = base / rel
        if not p.exists():
            return False, f"missing: {p}"
    return True, "ok"
