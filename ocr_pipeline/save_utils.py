from __future__ import annotations
import json
import base64
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .config import PARSED_DIR
from .ocr_runner import OCRResult, OCRImage

def _image_basename_sha(base64_data: str) -> str:
    """Generate SHA-based image filename for global uniqueness."""
    img_hash = _compute_image_hash(base64_data)
    return f"{img_hash}.png"

def _compute_image_hash(base64_data: str) -> str:
    """Compute SHA-256 hash of base64 image data."""
    # Remove data URL prefix if present
    if base64_data.startswith('data:'):
        base64_data = base64_data.split(',', 1)[1]
    return hashlib.sha256(base64_data.encode()).hexdigest()[:16]

def _create_index_data(doc_id: str, original_filename: str, rmd_content: str, images_metadata: List[Dict], ocr_result: OCRResult) -> Dict:
    """Create simple index.json data for future embeddings - basic version for Phase 1."""
    
    # Simple block detection for now
    blocks = []
    lines = rmd_content.split('\n')
    current_section = ""
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Detect headings
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            heading_text = line.strip('#').strip()
            current_section = heading_text
            blocks.append({
                "type": "heading",
                "level": level,
                "section": heading_text,
                "line": line_num
            })
        
        # Detect images
        elif '![' in line and '](' in line:
            blocks.append({
                "type": "image",
                "section": current_section,
                "line": line_num,
                "content": line
            })
        
        # Detect tables
        elif '|' in line:
            blocks.append({
                "type": "table",
                "section": current_section,
                "line": line_num,
                "preview": line[:100]
            })
        
        # Everything else is text
        else:
            blocks.append({
                "type": "text",
                "section": current_section,
                "line": line_num,
                "preview": line[:100]
            })
    
    return {
        "doc_id": doc_id,
        "original_filename": original_filename,
        "md_path": f"data/parsed/{original_filename.replace('.pdf', '')}/{original_filename.replace('.pdf', '.md')}",
        "rmd_path": f"data/parsed/{original_filename.replace('.pdf', '')}/{original_filename.replace('.pdf', '.rmd')}",
        "blocks": blocks,
        "images": images_metadata,
        "extraction": {
            "ocr_model": "mistral-ocr-latest",
            "request_id": ocr_result.request_id,
            "ts": datetime.now(timezone.utc).isoformat()
        }
    }

def write_outputs(
    *,
    doc_id: str,
    original_filename: str,
    sha256_hex: str,
    source_rel_path: str,
    ocr_model: str,
    page_count: int,
    ocr_result: OCRResult,
) -> Dict:
    """
    Write outputs using hybrid approach:
    1. Individual folder per PDF for clean organization
    2. Single clean .rmd file with proper image references (no base64 mess)
    3. Separate image files for easy access
    4. .index.json with image-text relationships for future embeddings
    """
    # Create individual folder for this PDF
    pdf_folder_name = original_filename.replace('.pdf', '')
    pdf_dir = PARSED_DIR / pdf_folder_name
    images_dir = pdf_dir / "images"
    
    # Ensure directories exist
    pdf_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all pages and create clean markdown
    all_pages_markdown = []
    images_metadata = []
    image_id_to_filename = {}  # Map image IDs to actual filenames
    
    for page in sorted(ocr_result.pages, key=lambda x: x.index):
        page_markdown = page.markdown or ""
        
        # Save images for this page and build filename mapping
        for img_idx, img in enumerate(page.images, start=1):
            # Generate SHA-based filename for global uniqueness
            img_filename = _image_basename_sha(img.image_base64)
            img_path = images_dir / img_filename
            
            # Save the image as separate file
            try:
                # Extract base64 data (remove data URL prefix if present)
                img_data = img.image_base64
                if img_data.startswith('data:'):
                    img_data = img_data.split(',', 1)[1]
                
                # Decode and save
                image_bytes = base64.b64decode(img_data)
                img_path.write_bytes(image_bytes)
                
                # Store comprehensive metadata for embeddings
                img_hash = _compute_image_hash(img.image_base64)
                pdf_folder_name = original_filename.replace('.pdf', '')
                images_metadata.append({
                    "sha_name": img_filename,
                    "original_ref": img.id or f"img-{img_idx-1}",
                    "page": int(page.index),
                    "relative_path": f"images/{img_filename}",
                    "absolute_path": str((PARSED_DIR / pdf_folder_name / "images" / img_filename).resolve()),
                    "bbox": img.bbox,
                    "caption": img.caption or "",
                    "hash": img_hash
                })
                
                # Map original image reference to SHA-based filename (clean for LLMs)
                if img.id:
                    image_id_to_filename[img.id] = f"images/{img_filename}"
                
            except Exception as e:
                print(f"Warning: Could not save image {img_filename}: {e}")
        
        # Add page separator for multi-page documents
        if all_pages_markdown:
            all_pages_markdown.append(f"\n\n--- page {page.index} ---\n\n")
        
        all_pages_markdown.append(page_markdown)
    
    # Combine all pages into single markdown
    combined_markdown = "".join(all_pages_markdown)
    
    # Fix image references to point to actual image files (keep clean format)
    for original_ref, file_path in image_id_to_filename.items():
        # Replace patterns like ![img-0.jpeg](img-0.jpeg) with ![img-0.jpeg](images/actual_file.png)
        patterns = [
            f"![{original_ref}]({original_ref})",
            f"![]({original_ref})",
        ]
        
        for pattern in patterns:
            if pattern in combined_markdown:
                # Keep the clean reference format, just fix the path
                clean_replacement = f"![{original_ref}]({file_path})"
                combined_markdown = combined_markdown.replace(pattern, clean_replacement)
    
    # Create both .md (pure markdown) and .rmd (R Markdown) files
    
    # 1. Pure markdown file (.md) - exactly what Mistral sends
    md_filename = original_filename.replace(".pdf", ".md")
    md_path = pdf_dir / md_filename
    md_path.write_text(combined_markdown, encoding="utf-8")
    
    # 2. R Markdown file (.rmd) - with YAML header for R/RStudio
    rmd_filename = original_filename.replace(".pdf", ".rmd")
    rmd_path = pdf_dir / rmd_filename
    
    rmd_header = (
        f"---\n"
        f"title: \"{original_filename.replace('.pdf', '')}\"\n"
        f"output: github_document\n"
        f"---\n\n"
    )
    
    rmd_path.write_text(rmd_header + combined_markdown, encoding="utf-8")
    
    # Create .index.json for future embeddings
    index_data = _create_index_data(
        doc_id=doc_id,
        original_filename=original_filename,
        rmd_content=combined_markdown,
        images_metadata=images_metadata,
        ocr_result=ocr_result
    )
    
    index_path = pdf_dir / f"{doc_id}.index.json"
    index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
    
    # Return summary
    return {
        "doc_id": doc_id,
        "md_path": str(md_path),
        "rmd_path": str(rmd_path),
        "index_path": str(index_path),
        "images_count": len(images_metadata),
        "pages_count": len(ocr_result.pages),
        "has_images": len(images_metadata) > 0
    }

def expected_artifacts_exist(doc_id: str, original_filename: str) -> bool:
    """Check if all expected output files exist for a document."""
    pdf_folder_name = original_filename.replace('.pdf', '')
    pdf_dir = PARSED_DIR / pdf_folder_name
    md_filename = original_filename.replace(".pdf", ".md")
    rmd_filename = original_filename.replace(".pdf", ".rmd")
    
    return (
        (pdf_dir / md_filename).exists() and
        (pdf_dir / rmd_filename).exists() and
        (pdf_dir / f"{doc_id}.index.json").exists() and
        (pdf_dir / "images").exists()
    )
