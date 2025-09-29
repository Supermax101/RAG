from pathlib import Path
import hashlib
from typing import Tuple
from slugify import slugify


def compute_file_sha256(file_path: Path) -> Tuple[str, int]:
    hasher = hashlib.sha256()
    size_bytes = 0
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            hasher.update(chunk)
            size_bytes += len(chunk)
    return hasher.hexdigest(), size_bytes


def derive_doc_id(original_filename: str, sha256_hex: str) -> str:
    filename_no_ext = Path(original_filename).stem
    slug = slugify(filename_no_ext, separator="_")
    return f"{slug}__{sha256_hex[:6]}"
