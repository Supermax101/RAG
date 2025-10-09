"""
DPT2 Document Loader - loads pre-chunked documents from DPT2 JSON output.

This loader:
- Reads DPT2 JSON files with pre-chunked medical documents
- Preserves chunk boundaries (NO re-chunking)
- Maintains chunk UUIDs, types, page numbers, and bounding boxes
- Passes chunks directly to embeddings without modification
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..models.documents import DocumentChunk
from ...config.settings import settings


class DPT2DocumentLoader:
    """Service to load pre-chunked DPT2 documents without re-chunking."""
    
    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.dpt2_dir = settings.project_root / "data" / "dpt2_output"
        
        # Validate directory exists
        if not self.dpt2_dir.exists():
            raise ValueError(f"DPT2 output directory not found: {self.dpt2_dir}")
    
    async def load_all_documents(self) -> Dict[str, Any]:
        """Load all DPT2 JSON documents into the vector store."""
        print("Loading DPT2 pre-chunked documents into vector store...")
        
        # Find all JSON files
        json_files = list(self.dpt2_dir.glob("*_response.json"))
        
        if not json_files:
            print(f"No DPT2 JSON files found in {self.dpt2_dir}")
            return {"loaded": 0, "failed": 0, "total_chunks": 0}
        
        print(f"Found {len(json_files)} DPT2 documents")
        
        loaded_count = 0
        failed_count = 0
        total_chunks = 0
        
        for i, json_file in enumerate(json_files, 1):
            try:
                print(f"Processing document {i}/{len(json_files)}: {json_file.stem}")
                
                # Load document chunks (as-is, no re-chunking)
                chunks = await self._load_document_chunks(json_file)
                
                if chunks:
                    doc_name = json_file.stem.replace("_response", "")
                    print(f"Adding {len(chunks)} pre-chunked pieces to ChromaDB...")
                    await self.rag_service.add_document_chunks(chunks, doc_name)
                    loaded_count += 1
                    total_chunks += len(chunks)
                    print(f"Loaded {len(chunks)} chunks from {doc_name}")
                    print(f"Progress: {i}/{len(json_files)} docs, {total_chunks} total chunks")
                else:
                    print(f"No chunks found in {json_file.name}")
                    
            except Exception as e:
                failed_count += 1
                print(f"Failed to load {json_file.name}: {e}")
                import traceback
                print(f"Error details: {traceback.format_exc()}")
        
        result = {
            "loaded": loaded_count,
            "failed": failed_count,
            "total_chunks": total_chunks
        }
        
        print(f"Loading complete: {result}")
        return result
    
    async def _load_document_chunks(self, json_file: Path) -> List[DocumentChunk]:
        """Load chunks from DPT2 JSON file WITHOUT re-chunking."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to read {json_file}: {e}")
            return []
        
        # Extract metadata
        metadata = data.get('metadata', {})
        source_filename = metadata.get('filename', json_file.stem)
        page_count = metadata.get('page_count', 0)
        
        # Extract pre-chunked data
        chunks = data.get('chunks', [])
        
        if not chunks:
            print(f"No chunks found in {json_file.name}")
            return []
        
        print(f"  Found {len(chunks)} pre-chunked pieces in {source_filename}")
        
        # Convert DPT2 chunks to DocumentChunk objects
        document_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Extract chunk data
            chunk_id = chunk.get('id', f"chunk_{i}")
            chunk_type = chunk.get('type', 'unknown')
            chunk_markdown = chunk.get('markdown', '')
            grounding = chunk.get('grounding', {})
            
            # Clean markdown content (remove anchor tags for cleaner embedding)
            cleaned_content = self._clean_chunk_content(chunk_markdown)
            
            # Skip empty or very short chunks
            if len(cleaned_content.strip()) < 10:
                continue
            
            # Skip logo chunks (not useful for RAG)
            if chunk_type == 'logo':
                continue
            
            # Extract page and bounding box
            page = grounding.get('page', 0)
            bounding_box = grounding.get('box', {})
            
            # Create rich metadata
            chunk_metadata = {
                "source_file": source_filename,
                "document_type": "medical_guideline",
                "chunk_type": chunk_type,
                "page": page,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "bounding_box": json.dumps(bounding_box) if bounding_box else "",
                "chunk_strategy": "dpt2_prechunked",
                "page_count": page_count,
            }
            
            # Create DocumentChunk
            doc_chunk = DocumentChunk(
                chunk_id=chunk_id,  # Use DPT2's UUID
                doc_id=source_filename,
                content=cleaned_content,
                chunk_type=f"dpt2_{chunk_type}",
                section=self._extract_section(cleaned_content),
                metadata=chunk_metadata
            )
            
            document_chunks.append(doc_chunk)
        
        print(f"  Created {len(document_chunks)} DocumentChunks (filtered from {len(chunks)})")
        
        # Print chunk type distribution
        type_counts = {}
        for chunk in document_chunks:
            ctype = chunk.metadata.get('chunk_type', 'unknown')
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
        
        print(f"  Chunk types: {type_counts}")
        
        return document_chunks
    
    def _clean_chunk_content(self, markdown: str) -> str:
        """Clean chunk markdown for better embedding."""
        # Remove anchor tags
        content = re.sub(r"<a id='[^']+'>\s*</a>\s*", '', markdown)
        
        # Clean up excessive whitespace
        content = re.sub(r'\n\n\n+', '\n\n', content)
        
        # Remove figure notation tags (keep description)
        content = re.sub(r'<::', '', content)
        content = re.sub(r'::>', '', content)
        content = re.sub(r': figure::', '', content)
        
        return content.strip()
    
    def _extract_section(self, content: str) -> str:
        """Extract section/heading from chunk content."""
        lines = content.split('\n')[:5]
        
        for line in lines:
            line = line.strip()
            # Check for markdown heading
            if line.startswith('#'):
                return line.replace('#', '').strip()
            # Check for bold text (might be heading)
            if line.startswith('**') and line.endswith('**'):
                return line.replace('**', '').strip()
        
        # Return first substantial line
        for line in lines:
            line = line.strip()
            if len(line) > 20 and not line.startswith('<'):
                return line[:100]
        
        return "Medical Content"
    
    async def load_single_document(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a single DPT2 document by filename."""
        # Find the JSON file
        json_file = self.dpt2_dir / f"{filename}_response.json"
        
        if not json_file.exists():
            # Try without _response suffix
            json_file = self.dpt2_dir / f"{filename}.json"
            
        if not json_file.exists():
            print(f"Document not found: {filename}")
            return None
        
        try:
            chunks = await self._load_document_chunks(json_file)
            
            if chunks:
                doc_name = filename.replace("_response", "")
                await self.rag_service.add_document_chunks(chunks, doc_name)
                return {
                    "document": doc_name,
                    "chunks_loaded": len(chunks),
                    "status": "success"
                }
            else:
                return {
                    "document": filename,
                    "chunks_loaded": 0,
                    "status": "no_content"
                }
                
        except Exception as e:
            return {
                "document": filename,
                "chunks_loaded": 0,
                "status": "failed",
                "error": str(e)
            }
    
    def get_available_documents(self) -> List[str]:
        """Get list of available DPT2 documents."""
        json_files = list(self.dpt2_dir.glob("*_response.json"))
        return [f.stem.replace("_response", "") for f in json_files]
    
    def get_document_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a specific document."""
        json_file = self.dpt2_dir / f"{filename}_response.json"
        
        if not json_file.exists():
            json_file = self.dpt2_dir / f"{filename}.json"
            
        if not json_file.exists():
            return None
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            chunks = data.get('chunks', [])
            
            # Count chunk types
            chunk_types = {}
            for chunk in chunks:
                ctype = chunk.get('type', 'unknown')
                chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
            
            return {
                "filename": metadata.get('filename', filename),
                "page_count": metadata.get('page_count', 0),
                "total_chunks": len(chunks),
                "chunk_types": chunk_types,
                "version": metadata.get('version', 'unknown'),
                "processing_time_ms": metadata.get('duration_ms', 0),
            }
            
        except Exception as e:
            print(f"Failed to get document info: {e}")
            return None

