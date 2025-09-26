from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional, List

import typer

from .config import (
    RAW_PDFS_DIR,
    PARSED_DIR,
    DEFAULT_OCR_MODEL,
    DEFAULT_EMBED_MODEL,
    ensure_directories,
    get_logger,
    MAX_FILE_MB,
    MAX_PAGES,
)
from .hashing import compute_file_sha256, derive_doc_id
from .pdf_info import get_pdf_page_count
from .ocr_runner import run_ocr
from .save_utils import write_outputs, expected_artifacts_exist
from .embedding_runner import create_embeddings_for_document, embeddings_exist_for_doc, load_document_embeddings
from .search import search_documents, print_search_results, load_all_embeddings_to_chromadb
# No manifest needed - use simple file-based skip logic


app = typer.Typer(no_args_is_help=True, add_completion=False)
logger = get_logger("test-ingest")


def _pick_next_batch_pdfs(batch_size: int = 10) -> List[Path]:
    """Pick the next batch of unprocessed PDFs."""
    # Get all PDFs sorted alphabetically for consistent ordering
    all_pdfs = sorted([p for p in RAW_PDFS_DIR.glob("*.pdf") if p.is_file()])
    
    if not all_pdfs:
        return []
    
    # Check which PDFs are already processed by looking for folder names
    unprocessed_pdfs = []
    for pdf_path in all_pdfs:
        # Create expected folder name (same as PDF name without .pdf)
        expected_folder_name = pdf_path.name.replace('.pdf', '')
        expected_folder_path = PARSED_DIR / expected_folder_name
        
        # If folder doesn't exist, this PDF needs processing
        if not expected_folder_path.exists():
            unprocessed_pdfs.append(pdf_path)
        else:
            # Double-check that the folder contains the expected files
            md_file = expected_folder_path / f"{expected_folder_name}.md"
            rmd_file = expected_folder_path / f"{expected_folder_name}.rmd"
            images_dir = expected_folder_path / "images"
            
            # Only skip if all essential files exist
            if not (md_file.exists() and rmd_file.exists() and images_dir.exists()):
                unprocessed_pdfs.append(pdf_path)
    
    # Return the first batch_size unprocessed PDFs
    return unprocessed_pdfs[:batch_size]


@app.command("test-ingest")
@app.command("test_ingest")
def test_ingest(
    force: bool = typer.Option(False, "--force", help="Force reprocessing even if hash matches and outputs exist"),
    batch_size: int = typer.Option(10, "--batch-size", help="Number of PDFs to process in this run"),
) -> None:
    ensure_directories()

    # Get next batch of PDFs to process
    if force:
        # If force flag is used, get all PDFs regardless of processing status
        pdf_batch = sorted([p for p in RAW_PDFS_DIR.glob("*.pdf") if p.is_file()])[:batch_size]
    else:
        # Get next unprocessed batch
        pdf_batch = _pick_next_batch_pdfs(batch_size)

    if not pdf_batch:
        typer.echo(f"No unprocessed PDFs found in {RAW_PDFS_DIR}.")
        typer.echo("All PDFs have been processed or no PDFs exist.")
        raise typer.Exit(code=0)

    typer.echo(f"Processing batch of {len(pdf_batch)} PDFs:")
    for i, pdf_path in enumerate(pdf_batch, 1):
        typer.echo(f"  {i}. {pdf_path.name}")

    # Process each PDF in the batch
    batch_results = []
    batch_start_time = time.time()

    for pdf_idx, pdf_path in enumerate(pdf_batch, 1):
        typer.echo(f"\n[{pdf_idx}/{len(pdf_batch)}] Processing: {pdf_path.name}")
        logger.info(f"Processing PDF {pdf_idx}/{len(pdf_batch)}: {pdf_path}")

        sha256_hex, size_bytes = compute_file_sha256(pdf_path)
        doc_id = derive_doc_id(pdf_path.name, sha256_hex)

        # Simple skip logic (unless force is used) - check folder exists
        expected_folder_name = pdf_path.name.replace('.pdf', '')
        expected_folder_path = PARSED_DIR / expected_folder_name
        
        if expected_folder_path.exists() and not force:
            # Double-check essential files exist
            md_file = expected_folder_path / f"{expected_folder_name}.md"
            rmd_file = expected_folder_path / f"{expected_folder_name}.rmd"
            images_dir = expected_folder_path / "images"
            
            if md_file.exists() and rmd_file.exists() and images_dir.exists():
                typer.echo("  -> Already processed, skipping")
                batch_results.append({
                    "pdf": pdf_path.name,
                    "status": "skipped",
                    "reason": "already processed"
                })
                continue

        page_count = get_pdf_page_count(pdf_path) or 0

        started = time.time()
        try:
            # Validate known limits proactively
            size_mb = size_bytes / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                raise RuntimeError(f"File size {size_mb:.2f} MB exceeds limit {MAX_FILE_MB} MB")
            if page_count and page_count > MAX_PAGES:
                raise RuntimeError(f"Page count {page_count} exceeds limit {MAX_PAGES}")

            ocr_result = run_ocr(pdf_path, model=DEFAULT_OCR_MODEL)

            # Dry-run validations before saving
            # Cross-check page counts
            ocr_pages = len(ocr_result.pages)
            ocr_reported = ocr_result.page_count or ocr_pages
            if page_count and ocr_reported and page_count != ocr_reported:
                logger.warning(f"Page count mismatch: PDF={page_count} OCR={ocr_reported}")

            # Validate page count integrity
            if ocr_pages == 0:
                raise RuntimeError("OCR returned zero pages")
            # len(pages) should match reported page_count if present
            if ocr_result.page_count and ocr_pages != ocr_result.page_count:
                raise RuntimeError(
                    f"len(pages) != page_count (pages={ocr_pages}, page_count={ocr_result.page_count})"
                )
            # Ensure at least one page has non-empty text
            if not any((p.markdown or "").strip() for p in ocr_result.pages):
                raise RuntimeError("No non-empty page markdown returned by OCR")

            # Save outputs (clean hybrid structure)
            output_summary = write_outputs(
                doc_id=doc_id,
                original_filename=pdf_path.name,
                sha256_hex=sha256_hex,
                source_rel_path=str((RAW_PDFS_DIR / pdf_path.name).as_posix()),
                ocr_model=DEFAULT_OCR_MODEL,
                page_count=ocr_reported or ocr_pages,
                ocr_result=ocr_result,
            )

            # Verify outputs exist
            if not expected_artifacts_exist(doc_id, pdf_path.name):
                raise RuntimeError("Expected output artifacts not found")

            duration = time.time() - started
            
            typer.echo(f"  -> Completed in {duration:.1f}s ({output_summary['images_count']} images)")
            
            batch_results.append({
                "pdf": pdf_path.name,
                "status": "completed",
                "doc_id": doc_id,
                "images": output_summary["images_count"],
                "seconds": round(duration, 2)
            })

        except Exception as e:
            duration = time.time() - started
            typer.echo(f"  -> Failed after {duration:.1f}s: {e}")
            logger.error(f"OCR failed for {pdf_path.name}: {e}")
            
            batch_results.append({
                "pdf": pdf_path.name,
                "status": "failed",
                "error": str(e),
                "seconds": round(duration, 2)
            })

    # Print batch summary
    batch_duration = time.time() - batch_start_time
    successful = [r for r in batch_results if r["status"] == "completed"]
    skipped = [r for r in batch_results if r["status"] == "skipped"]
    failed = [r for r in batch_results if r["status"] == "failed"]

    typer.echo(f"\n{'='*50}")
    typer.echo(f"BATCH COMPLETED in {batch_duration:.1f}s")
    typer.echo(f"{'='*50}")
    typer.echo(f"Processed: {len(successful)}")
    typer.echo(f"Skipped: {len(skipped)}")
    typer.echo(f"Failed: {len(failed)}")

    if successful:
        total_images = sum(r.get("images", 0) for r in successful)
        typer.echo(f"Total images extracted: {total_images}")
    
    # Print next steps
    remaining_pdfs = _pick_next_batch_pdfs(batch_size)
    if remaining_pdfs:
        typer.echo(f"\nNext run will process {len(remaining_pdfs)} more PDFs.")
    else:
        typer.echo(f"\n🎉 All PDFs processed!")

    # Exit with error code if any failures
    if failed:
        raise typer.Exit(code=1)


def _find_parsed_documents() -> List[Path]:
    """Find all parsed documents with index.json files."""
    index_files = []
    
    for folder in PARSED_DIR.iterdir():
        if folder.is_dir():
            # Look for index.json files in each parsed document folder
            index_files.extend(folder.glob("*.index.json"))
    
    return sorted(index_files)


@app.command("test-embedding")
@app.command("test_embedding")
def test_embedding(
    model: str = typer.Option(DEFAULT_EMBED_MODEL, "--model", help="Embedding model to use"),
) -> None:
    """Test embedding creation for the first parsed document (like test-ingest for OCR)."""
    ensure_directories()
    
    # Find all parsed documents
    all_index_files = _find_parsed_documents()
    
    if not all_index_files:
        typer.echo(f"No parsed documents found in {PARSED_DIR}.")
        typer.echo("Run 'test-ingest' first to process PDFs.")
        raise typer.Exit(code=0)
    
    # Pick the first document alphabetically
    index_file = all_index_files[0]
    doc_id = index_file.stem.replace('.index', '')
    
    typer.echo(f"Testing embedding creation for: {index_file.parent.name}")
    typer.echo(f"Using model: {model}")
    
    # Check if embeddings already exist
    if embeddings_exist_for_doc(doc_id):
        typer.echo("  -> Embeddings already exist for this document")
        typer.echo("  -> Use 'create-embeddings --force' to recreate")
        
        # Show existing embedding info
        doc_embeddings = load_document_embeddings(doc_id)
        if doc_embeddings:
            typer.echo(f"  -> Existing: {len(doc_embeddings.chunks)} chunks, {doc_embeddings.total_tokens} tokens")
        raise typer.Exit(code=0)
    
    started = time.time()
    try:
        typer.echo("  -> Creating embeddings...")
        doc_embeddings = create_embeddings_for_document(index_file, model=model)
        
        if doc_embeddings:
            duration = time.time() - started
            typer.echo(f"  -> ✅ Success in {duration:.1f}s")
            typer.echo(f"     📊 {len(doc_embeddings.chunks)} chunks embedded")
            typer.echo(f"     🎯 {doc_embeddings.total_tokens} tokens used")
            typer.echo(f"     💾 Saved to: data/embeddings/vectors/{doc_id}.embeddings.json")
            
            # Show chunk breakdown
            chunk_types = {}
            for chunk in doc_embeddings.chunks:
                chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            
            typer.echo(f"     🔍 Chunk types: {dict(chunk_types)}")
            
        else:
            duration = time.time() - started
            typer.echo(f"  -> ❌ Failed after {duration:.1f}s: No embeddings created")
            raise typer.Exit(code=1)
    
    except Exception as e:
        duration = time.time() - started
        typer.echo(f"  -> ❌ Failed after {duration:.1f}s: {e}")
        logger.error(f"Test embedding failed for {index_file}: {e}")
        raise typer.Exit(code=1)


@app.command("create-embeddings")
@app.command("create_embeddings")
def create_embeddings(
    force: bool = typer.Option(False, "--force", help="Force re-creation even if embeddings exist"),
    batch_size: int = typer.Option(10, "--batch-size", help="Number of documents to process in this run"),
    model: str = typer.Option(DEFAULT_EMBED_MODEL, "--model", help="Embedding model to use"),
) -> None:
    """Create embeddings for parsed documents using Mistral embeddings API."""
    ensure_directories()
    
    # Find all parsed documents
    all_index_files = _find_parsed_documents()
    
    if not all_index_files:
        typer.echo(f"No parsed documents found in {PARSED_DIR}.")
        typer.echo("Run 'test-ingest' first to process PDFs.")
        raise typer.Exit(code=0)
    
    # Filter for documents that need embeddings
    if force:
        # Process all documents if force flag is used
        docs_to_process = all_index_files[:batch_size]
    else:
        # Only process documents without existing embeddings
        docs_to_process = []
        for index_file in all_index_files:
            # Extract doc_id from filename (e.g., "document__abc123.index.json" -> "document__abc123")
            doc_id = index_file.stem.replace('.index', '')
            
            if not embeddings_exist_for_doc(doc_id):
                docs_to_process.append(index_file)
                
                if len(docs_to_process) >= batch_size:
                    break
    
    if not docs_to_process:
        typer.echo("All parsed documents already have embeddings.")
        typer.echo("Use --force to regenerate embeddings.")
        raise typer.Exit(code=0)
    
    typer.echo(f"Creating embeddings for {len(docs_to_process)} documents using model: {model}")
    
    # Process each document
    batch_results = []
    batch_start_time = time.time()
    
    for doc_idx, index_file in enumerate(docs_to_process, 1):
        typer.echo(f"\n[{doc_idx}/{len(docs_to_process)}] Processing: {index_file.parent.name}")
        logger.info(f"Creating embeddings for document {doc_idx}/{len(docs_to_process)}: {index_file}")
        
        started = time.time()
        try:
            doc_embeddings = create_embeddings_for_document(index_file, model=model)
            
            if doc_embeddings:
                duration = time.time() - started
                typer.echo(f"  -> Completed in {duration:.1f}s ({len(doc_embeddings.chunks)} chunks, {doc_embeddings.total_tokens} tokens)")
                
                batch_results.append({
                    "document": index_file.parent.name,
                    "status": "completed",
                    "chunks": len(doc_embeddings.chunks),
                    "tokens": doc_embeddings.total_tokens,
                    "seconds": round(duration, 2)
                })
            else:
                duration = time.time() - started
                typer.echo(f"  -> Failed after {duration:.1f}s: No embeddings created")
                
                batch_results.append({
                    "document": index_file.parent.name,
                    "status": "failed",
                    "error": "No embeddings created",
                    "seconds": round(duration, 2)
                })
        
        except Exception as e:
            duration = time.time() - started
            typer.echo(f"  -> Failed after {duration:.1f}s: {e}")
            logger.error(f"Embedding creation failed for {index_file}: {e}")
            
            batch_results.append({
                "document": index_file.parent.name,
                "status": "failed",
                "error": str(e),
                "seconds": round(duration, 2)
            })
    
    # Print batch summary
    batch_duration = time.time() - batch_start_time
    successful = [r for r in batch_results if r["status"] == "completed"]
    failed = [r for r in batch_results if r["status"] == "failed"]
    
    typer.echo(f"\n{'='*50}")
    typer.echo(f"EMBEDDING BATCH COMPLETED in {batch_duration:.1f}s")
    typer.echo(f"{'='*50}")
    typer.echo(f"Processed: {len(successful)}")
    typer.echo(f"Failed: {len(failed)}")
    
    if successful:
        total_chunks = sum(r.get("chunks", 0) for r in successful)
        total_tokens = sum(r.get("tokens", 0) for r in successful)
        typer.echo(f"Total chunks embedded: {total_chunks}")
        typer.echo(f"Total tokens used: {total_tokens}")
    
    # Print next steps
    remaining_docs = len([f for f in all_index_files if not embeddings_exist_for_doc(f.stem.replace('.index', ''))])
    if remaining_docs > 0:
        typer.echo(f"\nNext run will process {min(remaining_docs, batch_size)} more documents.")
    else:
        typer.echo(f"\n🎉 All documents have embeddings!")
    
    # Exit with error code if any failures
    if failed:
        raise typer.Exit(code=1)


@app.command("load-chromadb")
@app.command("load_chromadb")
def load_chromadb(
    force: bool = typer.Option(False, "--force", help="Force reload even if embeddings exist in ChromaDB"),
) -> None:
    """Load all embeddings into ChromaDB for fast vector search."""
    ensure_directories()
    
    typer.echo("Loading embeddings into ChromaDB...")
    
    try:
        loaded_count, skipped_count = load_all_embeddings_to_chromadb(force=force)
        
        typer.echo(f"\n{'='*50}")
        typer.echo(f"CHROMADB LOADING COMPLETED")
        typer.echo(f"{'='*50}")
        typer.echo(f"Loaded: {loaded_count} documents")
        typer.echo(f"Skipped: {skipped_count} documents")
        
        if loaded_count > 0:
            typer.echo(f"\n✅ ChromaDB is ready for search!")
        elif skipped_count > 0:
            typer.echo(f"\n✅ ChromaDB already contains all embeddings!")
        else:
            typer.echo(f"\n⚠️  No embeddings found. Run 'create-embeddings' first.")
        
    except Exception as e:
        typer.echo(f"❌ Failed to load ChromaDB: {e}")
        raise typer.Exit(code=1)


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of results to return"),
    show_content: bool = typer.Option(True, "--content/--no-content", help="Show full content previews"),
    show_images: bool = typer.Option(True, "--images/--no-images", help="Show nearby images"),
    model: str = typer.Option(DEFAULT_EMBED_MODEL, "--model", help="Embedding model to use"),
) -> None:
    """Search through processed documents using vector similarity."""
    ensure_directories()
    
    typer.echo(f"🔍 Searching for: '{query}'")
    
    try:
        results = search_documents(query, limit=limit, model=model)
        
        if results.results:
            print_search_results(results, show_content=show_content, show_images=show_images)
        else:
            typer.echo("\n❌ No results found.")
            typer.echo("💡 Tips:")
            typer.echo("   • Try different keywords")
            typer.echo("   • Check if embeddings exist: run 'create-embeddings'")
            typer.echo("   • Load embeddings into ChromaDB: run 'load-chromadb'")
        
    except Exception as e:
        typer.echo(f"❌ Search failed: {e}")
        logger.error(f"Search failed for query '{query}': {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
