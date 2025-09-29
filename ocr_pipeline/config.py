import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables from .env at project root if present
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
PARSED_DIR = DATA_DIR / "parsed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHROMADB_DIR = EMBEDDINGS_DIR / "chromadb"
VECTORS_DIR = EMBEDDINGS_DIR / "vectors"
METADATA_DIR = EMBEDDINGS_DIR / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"

# OCR Configuration
DEFAULT_OCR_MODEL = os.getenv("OCR_MODEL", "mistral-ocr-latest")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OCR_BASE_URL = os.getenv("OCR_BASE_URL", "https://api.mistral.ai")
OCR_ENDPOINT_PATH = os.getenv("OCR_ENDPOINT_PATH", "/v1/ocr")

# Embedding Configuration
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "mistral-embed")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://api.mistral.ai")
EMBED_ENDPOINT_PATH = os.getenv("EMBED_ENDPOINT_PATH", "/v1/embeddings")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))
EMBED_CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE", "512"))

# ChromaDB Configuration
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "medical_docs")

# Limits per docs FAQ
MAX_FILE_MB = float(os.getenv("OCR_MAX_FILE_MB", "50"))
MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "1000"))


def ensure_directories() -> None:
    RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# OCRConfig removed - unused complexity


def get_logger(name: str = "app", logfile: Optional[Path] = LOGS_DIR / "app.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if logfile is not None:
        file_handler = logging.FileHandler(str(logfile))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
