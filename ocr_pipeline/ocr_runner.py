from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .config import MISTRAL_API_KEY, DEFAULT_OCR_MODEL, OCR_BASE_URL, OCR_ENDPOINT_PATH


"""
Mistral Document AI / OCR client adapter

Official docs:
- Basic OCR: https://docs.mistral.ai/capabilities/document_ai/basic_ocr/
- Document AI overview: https://docs.mistral.ai/capabilities/document_ai/

This module normalizes the OCR response into the project's internal schema.
The base URL and endpoint path are configurable via env (OCR_BASE_URL, OCR_ENDPOINT_PATH).
Optionally, if the `mistralai` SDK is installed and OCR_USE_SDK=1, the SDK will be used.
"""


@dataclass
class OCRImage:
    page_index: int  # 1-based
    caption: Optional[str]
    bbox: Dict[str, float]
    image_base64: str  # PNG base64
    id: Optional[str] = None  # Image ID/reference from markdown


@dataclass
class OCRPage:
    index: int  # 1-based
    markdown: str
    images: List[OCRImage]


@dataclass
class OCRResult:
    request_id: str
    pages: List[OCRPage]
    page_count: int


def _extract_image_b64(img: Dict) -> Optional[str]:
    for key in ("image_base64", "base64", "png_base64", "data", "content"):
        if key in img and isinstance(img[key], str) and len(img[key]) > 0:
            return img[key]
    return None


def run_ocr(pdf_path: Path, *, model: Optional[str] = None, timeout_sec: int = 600, max_retries: int = 4) -> OCRResult:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment")

    model_to_use = model or DEFAULT_OCR_MODEL

    with pdf_path.open("rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode("ascii")

    data = None
    request_id = ""

    # Optional SDK path
    use_sdk = (os.getenv("OCR_USE_SDK", "0") == "1")
    if use_sdk:
        try:
            from mistralai import Mistral  # type: ignore
            client = Mistral(api_key=MISTRAL_API_KEY)
            sdk_resp = client.ocr.process(
                model=model_to_use,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_b64}",
                },
                include_image_base64=True,
            )
            # Convert SDK response to dict if needed
            if hasattr(sdk_resp, "model_dump"):
                data = sdk_resp.model_dump()
            elif hasattr(sdk_resp, "to_dict"):
                data = sdk_resp.to_dict()
            else:
                data = dict(sdk_resp) if isinstance(sdk_resp, dict) else json.loads(json.dumps(sdk_resp))
        except Exception:
            data = None
            # Fall back to HTTP path

    if data is None:
        # HTTP path
        # Construct base URL and endpoint from configuration
        base = OCR_BASE_URL.rstrip("/")
        path = OCR_ENDPOINT_PATH if OCR_ENDPOINT_PATH.startswith("/") else "/" + OCR_ENDPOINT_PATH
        url = f"{base}{path}"
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_to_use,
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_b64}",
            },
            "include_image_base64": True,
        }

        # Robust retry with exponential backoff for 429/5xx and timeouts
        attempt = 0
        last_err = None
        while attempt <= max_retries:
            try:
                started = time.time()
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(15, 120),
                )
                _ = time.time() - started
                if resp.status_code in (429, 500, 502, 503, 504):
                    attempt += 1
                    sleep_sec = min(2 ** attempt, 30)
                    time.sleep(sleep_sec)
                    continue
                if resp.status_code >= 400:
                    try:
                        err = resp.json()
                    except Exception:
                        err = {"text": resp.text}
                    raise RuntimeError(f"Mistral OCR error {resp.status_code}: {err}")
                break
            except requests.Timeout as te:
                last_err = te
                attempt += 1
                time.sleep(min(2 ** attempt, 30))
            except requests.RequestException as re:
                last_err = re
                attempt += 1
                time.sleep(min(2 ** attempt, 30))
        else:
            # If we exit loop without break
            raise RuntimeError(f"Mistral OCR request failed after retries: {last_err}")

        data = resp.json()
        # Request ID may be in headers or body per docs
        request_id = resp.headers.get("x-request-id") or request_id

    # For SDK path, request_id may be present in body
    request_id = request_id or data.get("request_id") or data.get("id") or data.get("job_id") or ""
    if not request_id:
        request_id = f"local-{int(time.time())}"

    pages_raw = data.get("pages") or data.get("result") or data.get("document", {}).get("pages") or []
    reported_page_count = int(
        data.get("page_count")
        or (data.get("usage_info") or {}).get("pages_processed")
        or len(pages_raw)
        or 0
    )

    pages: List[OCRPage] = []
    for idx, page in enumerate(pages_raw, start=1):
        page_index = page.get("index") or idx
        markdown = (
            page.get("markdown")
            or page.get("text_markdown")
            or page.get("text")
            or ""
        )
        images_raw = page.get("images") or []
        norm_images: List[OCRImage] = []
        for img_i, img in enumerate(images_raw, start=1):
            b64 = _extract_image_b64(img)
            if not b64:
                continue
            bbox = img.get("bbox") or img.get("bounding_box") or {}
            norm_bbox = {
                "x": float(bbox.get("x", 0.0)),
                "y": float(bbox.get("y", 0.0)),
                "w": float(bbox.get("w", bbox.get("width", 0.0))),
                "h": float(bbox.get("h", bbox.get("height", 0.0))),
            }
            caption = img.get("caption") or img.get("alt") or None
            img_id = img.get("id") or img.get("image_id") or None
            norm_images.append(OCRImage(page_index=page_index, caption=caption, bbox=norm_bbox, image_base64=b64, id=img_id))
        pages.append(OCRPage(index=page_index, markdown=markdown, images=norm_images))

    return OCRResult(request_id=request_id, pages=pages, page_count=reported_page_count)
