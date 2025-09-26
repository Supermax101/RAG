#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from typing import Any

import requests

# Reuse project config (loads .env automatically)
from app.config import MISTRAL_API_KEY, OCR_BASE_URL, OCR_ENDPOINT_PATH, DEFAULT_OCR_MODEL


def main() -> int:
    if not MISTRAL_API_KEY:
        print("MISTRAL_API_KEY is not set. Populate .env first.", file=sys.stderr)
        return 2

    base = OCR_BASE_URL.rstrip("/")
    path = OCR_ENDPOINT_PATH if OCR_ENDPOINT_PATH.startswith("/") else "/" + OCR_ENDPOINT_PATH
    url = f"{base}{path}"

    # 1x1 transparent PNG
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+pRnkAAAAASUVORK5CYII="
    )
    data_url = f"data:image/png;base64,{tiny_png_b64}"

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": DEFAULT_OCR_MODEL,
        "document": {
            "type": "image_url",
            "image_url": data_url,
        },
        "include_image_base64": True,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=(15, 30))
    except requests.RequestException as e:
        print(f"Request error: {e}", file=sys.stderr)
        return 1

    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
    except Exception:
        print(resp.text)
        return 1 if resp.status_code >= 400 else 0

    if resp.status_code >= 400:
        print(json.dumps(data, indent=2))
        return 1

    pages = data.get("pages") or []
    usage = data.get("usage_info") or {}
    request_id = resp.headers.get("x-request-id") or data.get("request_id") or data.get("id")
    print(
        json.dumps(
            {
                "request_id": request_id,
                "pages": len(pages),
                "pages_processed": usage.get("pages_processed"),
                "model": data.get("model"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
