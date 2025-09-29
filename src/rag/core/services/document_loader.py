"""
Enhanced document loader service with medical context-preserving chunking strategies.
"""
import json
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
import tiktoken  # For token counting

from ..models.documents import DocumentChunk
from ...config.settings import settings


class DocumentLoader:
    """Enhanced service to load parsed documents with medical context-preserving chunking."""
    
    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.parsed_dir = settings.parsed_dir
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = int(self.chunk_size * 0.2)  # 20% overlap
        
        # Initialize advanced text splitters
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize token counter for medical content
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None  # Fallback to len() if tiktoken fails
    
    async def load_all_documents(self) -> Dict[str, Any]:
        """Load all parsed documents into the vector store."""
        print("🔄 Loading documents into vector store...")
        
        # Find all parsed document directories
        doc_dirs = [d for d in self.parsed_dir.iterdir() if d.is_dir()]
        
        if not doc_dirs:
            print(f"❌ No parsed documents found in {self.parsed_dir}")
            return {"loaded": 0, "failed": 0, "total_chunks": 0}
        
        print(f"📁 Found {len(doc_dirs)} document directories")
        
        loaded_count = 0
        failed_count = 0
        total_chunks = 0
        
        for i, doc_dir in enumerate(doc_dirs, 1):
            try:
                print(f"📄 Processing document {i}/{len(doc_dirs)}: {doc_dir.name}")
                
                # Load document chunks
                chunks = await self._load_document_chunks(doc_dir)
                
                if chunks:
                    doc_name = doc_dir.name
                    print(f"🔄 Adding {len(chunks)} chunks to ChromaDB...")
                    await self.rag_service.add_document_chunks(chunks, doc_name)
                    loaded_count += 1
                    total_chunks += len(chunks)
                    print(f"✅ Loaded {len(chunks)} chunks from {doc_name}")
                    print(f"📊 Progress: {i}/{len(doc_dirs)} docs, {total_chunks} total chunks")
                else:
                    print(f"⚠️  No chunks found in {doc_dir.name}")
                    
            except Exception as e:
                failed_count += 1
                print(f"❌ Failed to load {doc_dir.name}: {e}")
                import traceback
                print(f"Error details: {traceback.format_exc()}")
        
        result = {
            "loaded": loaded_count,
            "failed": failed_count,
            "total_chunks": total_chunks
        }
        
        print(f"📊 Loading complete: {result}")
        return result
    
    async def _load_document_chunks(self, doc_dir: Path) -> List[DocumentChunk]:
        """Load chunks from a single document directory with enhanced processing."""
        # Look for markdown file and index file
        md_files = list(doc_dir.glob("*.md"))
        
        if not md_files:
            print(f"⚠️  No .md file found in {doc_dir.name}")
            return []
        
        md_file = md_files[0]  # Use first markdown file
        doc_id = doc_dir.name
        
        # Read the full document content
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Failed to read {md_file}: {e}")
            return []
        
        # Load enhanced metadata
        metadata = await self._load_enhanced_metadata(doc_dir)
        
        # Apply advanced medical chunking strategies
        chunks = await self._create_medical_chunks(
            content=content,
            doc_id=doc_id,
            doc_name=doc_dir.name,
            metadata=metadata
        )
        
        return chunks
    
    async def _load_enhanced_metadata(self, doc_dir: Path) -> Dict[str, Any]:
        """Load document metadata with enhanced medical classification."""
        metadata = {"document_type": "medical_text"}
        
        # Load existing index metadata
        index_files = list(doc_dir.glob("*.index.json"))
        if index_files:
            try:
                with open(index_files[0], 'r', encoding='utf-8') as f:
                    existing_meta = json.load(f)
                    metadata.update(existing_meta)
            except Exception as e:
                print(f"⚠️  Failed to load metadata: {e}")
        
        # Enhanced metadata extraction
        doc_name = doc_dir.name.lower()
        
        # Classify document type based on name patterns
        if any(term in doc_name for term in ['aspen', 'guideline', 'protocol']):
            metadata['document_type'] = 'clinical_guideline'
            metadata['authority'] = 'ASPEN'
        elif any(term in doc_name for term in ['nicu', 'neonatal', 'pediatric']):
            metadata['document_type'] = 'pediatric_protocol'
            metadata['specialty'] = 'pediatrics'
        elif any(term in doc_name for term in ['tpn', 'parenteral']):
            metadata['document_type'] = 'nutrition_protocol'
            metadata['focus_area'] = 'parenteral_nutrition'
        elif any(term in doc_name for term in ['fluid', 'electrolyte']):
            metadata['document_type'] = 'physiology_reference'
            metadata['focus_area'] = 'fluid_electrolyte'
        
        # Extract year from filename if present
        year_match = re.search(r'(19|20)\d{2}', doc_name)
        if year_match:
            metadata['year'] = int(year_match.group())
        
        return metadata
    
    async def _create_medical_chunks(
        self,
        content: str,
        doc_id: str,
        doc_name: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create chunks using enhanced medical context-preserving strategies."""
        
        all_chunks = []
        
        # Strategy 1: Medical Context-Preserving Chunking
        try:
            medical_chunks = await self._medical_context_chunking(
                content, doc_id, doc_name, metadata
            )
            all_chunks.extend(medical_chunks)
            print(f"✅ Medical context chunking: {len(medical_chunks)} chunks")
        except Exception as e:
            print(f"⚠️  Medical context chunking failed: {e}")
        
        # Strategy 2: Clinical Relationship Chunking
        try:
            relationship_chunks = await self._clinical_relationship_chunking(
                content, doc_id, doc_name, metadata
            )
            all_chunks.extend(relationship_chunks)
            print(f"✅ Clinical relationship chunking: {len(relationship_chunks)} chunks")
        except Exception as e:
            print(f"⚠️  Clinical relationship chunking failed: {e}")
        
        # Strategy 3: Enhanced Markdown-aware chunking
        try:
            md_chunks = await self._enhanced_markdown_chunking(
                content, doc_id, doc_name, metadata
            )
            all_chunks.extend(md_chunks)
            print(f"✅ Enhanced markdown chunking: {len(md_chunks)} chunks")
        except Exception as e:
            print(f"⚠️  Enhanced markdown chunking failed: {e}")
        
        # Strategy 4: Table and numerical data preservation
        try:
            table_chunks = await self._preserve_tables_and_data(
                content, doc_id, doc_name, metadata
            )
            all_chunks.extend(table_chunks)
            print(f"✅ Table preservation chunking: {len(table_chunks)} chunks")
        except Exception as e:
            print(f"⚠️  Table chunking failed: {e}")
        
        # Remove duplicates and optimize while preserving medical context
        optimized_chunks = self._optimize_medical_chunks(all_chunks)
        
        return optimized_chunks
    
    async def load_single_document(self, doc_name: str) -> Optional[Dict[str, Any]]:
        """Load a single document by name."""
        doc_dir = self.parsed_dir / doc_name
        
        if not doc_dir.exists():
            print(f"❌ Document directory not found: {doc_name}")
            return None
        
        try:
            chunks = await self._load_document_chunks(doc_dir)
            
            if chunks:
                await self.rag_service.add_document_chunks(chunks, doc_name)
                return {
                    "document": doc_name,
                    "chunks_loaded": len(chunks),
                    "status": "success"
                }
            else:
                return {
                    "document": doc_name,
                    "chunks_loaded": 0,
                    "status": "no_content"
                }
                
        except Exception as e:
            return {
                "document": doc_name,
                "chunks_loaded": 0,
                "status": "failed",
                "error": str(e)
            }
    
    # =============== MEDICAL CONTEXT-PRESERVING CHUNKING METHODS ===============

    async def _medical_context_chunking(
        self,
        content: str,
        doc_id: str,
        doc_name: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Enhanced chunking that preserves medical context and clinical relationships."""
        
        chunks = []
        
        # Medical context patterns that should stay together
        medical_patterns = [
            # Dosage patterns - keep dose with indications
            (r'(?i)(dose|dosage|dosing|mg/kg|g/kg|ml/kg|units/kg).*?(?=\n\n|\n#{1,3}|\n\*\*|\Z)', 'dosage_guidance'),
            # Contraindication patterns - keep with context
            (r'(?i)(contraindic|caution|warning|precaution|adverse|side effect).*?(?=\n\n|\n#{1,3}|\n\*\*|\Z)', 'safety_information'),
            # Monitoring patterns - keep with what to monitor
            (r'(?i)(monitor|surveillance|follow.?up|lab|laboratory).*?(?=\n\n|\n#{1,3}|\n\*\*|\Z)', 'monitoring_guidance'),
            # Age-specific guidance - keep age with recommendations  
            (r'(?i)(preterm|neonatal|pediatric|adult|infant|child).*?(?=\n\n|\n#{1,3}|\n\*\*|\Z)', 'age_specific'),
            # Calculation patterns - keep formulas with context
            (r'(?i)(calculat|formula|equation|determine).*?(?=\n\n|\n#{1,3}|\n\*\*|\Z)', 'calculation_guidance')
        ]
        
        processed_spans = set()  # Track what we've already processed
        
        for pattern, chunk_type in medical_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
            
            for match in matches:
                start, end = match.span()
                
                # Skip if already processed by another pattern
                if any(start <= existing_end and end >= existing_start 
                       for existing_start, existing_end in processed_spans):
                    continue
                
                # Expand context to include surrounding sentences for better understanding
                expanded_content = self._expand_medical_context(content, start, end)
                
                if len(expanded_content.strip()) < 100:  # Skip very short chunks
                    continue
                
                # Extract section information
                section = self._extract_section_from_position(content, start)
                
                # LangChain best practice: Simple metadata only
                simple_metadata = {
                    "document_type": metadata.get("document_type", "medical_text"),
                    "document_name": doc_name,
                    "section": section,
                    "chunk_strategy": "medical_context_preserving",
                    "medical_focus": chunk_type,
                    "content_type": chunk_type,
                    "authority": metadata.get("authority", ""),
                    "year": metadata.get("year", 0) if metadata.get("year") else 0
                }
                
                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_medical_{chunk_type}_{len(chunks)}",
                    doc_id=doc_id,
                    content=expanded_content,
                    chunk_type=f"medical_{chunk_type}",
                    section=section,
                    metadata=simple_metadata
                )
                
                chunks.append(chunk)
                processed_spans.add((start, end))
        
        return chunks

    async def _clinical_relationship_chunking(
        self,
        content: str,
        doc_id: str,
        doc_name: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunk content preserving clinical relationships and dependencies."""
        
        chunks = []
        
        # Split content into paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Look for clinical relationships and group related paragraphs
        i = 0
        while i < len(paragraphs):
            current_chunk = [paragraphs[i]]
            chunk_context = paragraphs[i].lower()
            
            # Look ahead for related clinical content
            j = i + 1
            while j < len(paragraphs) and j < i + 4:  # Look up to 3 paragraphs ahead
                next_para = paragraphs[j].lower()
                
                # Check for clinical relationships
                if self._has_clinical_relationship(chunk_context, next_para):
                    current_chunk.append(paragraphs[j])
                    chunk_context += " " + next_para
                    j += 1
                else:
                    break
            
            # Create chunk if substantial
            combined_content = '\n\n'.join(current_chunk)
            if len(combined_content.strip()) > 200:  # Ensure substantial content
                
                # Determine clinical focus
                clinical_focus = self._determine_clinical_focus(combined_content)
                section = self._extract_section_from_content(combined_content)
                
                # LangChain best practice: Simple metadata only
                simple_metadata = {
                    "document_type": metadata.get("document_type", "medical_text"),
                    "document_name": doc_name,
                    "section": section,
                    "chunk_strategy": "clinical_relationship",
                    "clinical_focus": clinical_focus,
                    "content_type": clinical_focus,
                    "authority": metadata.get("authority", ""),
                    "year": metadata.get("year", 0) if metadata.get("year") else 0
                }
                
                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_clinical_{clinical_focus}_{len(chunks)}",
                    doc_id=doc_id,
                    content=combined_content,
                    chunk_type="clinical_relationship",
                    section=section,
                    metadata=simple_metadata
                )
                
                chunks.append(chunk)
            
            i = j if j > i + 1 else i + 1
        
        return chunks

    async def _enhanced_markdown_chunking(
        self,
        content: str,
        doc_id: str,
        doc_name: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Enhanced markdown chunking with better medical context preservation."""
        
        chunks = []
        
        # Use LangChain's markdown splitter with medical-optimized settings
        # Larger chunks to preserve more context
        md_splitter = MarkdownTextSplitter(
            chunk_size=1024,  # Increased from 512
            chunk_overlap=256,  # Increased overlap for better context
            length_function=len
        )
        
        docs = md_splitter.create_documents([content])
        
        for i, doc in enumerate(docs):
            if len(doc.page_content.strip()) < 100:
                continue
            
            # Ensure we don't cut off mid-sentence or mid-clinical-thought
            optimized_content = self._optimize_chunk_boundaries(doc.page_content)
            
            # Extract header/section information
            section = self._extract_markdown_section(doc.page_content)
            
            # LangChain best practice: Simple metadata only
            simple_metadata = {
                "document_type": metadata.get("document_type", "medical_text"),
                "document_name": doc_name,
                "section": section,
                "chunk_strategy": "enhanced_markdown",
                "content_optimized": "true",  # String instead of boolean
                "authority": metadata.get("authority", ""),
                "year": metadata.get("year", 0) if metadata.get("year") else 0
            }
            
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_enhanced_md_{i}",
                doc_id=doc_id,
                content=optimized_content,
                chunk_type="enhanced_markdown",
                section=section,
                metadata=simple_metadata
            )
            
            chunks.append(chunk)
        
        return chunks

    async def _preserve_tables_and_data(
        self,
        content: str,
        doc_id: str,
        doc_name: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Preserve tables, lists, and numerical data as complete units."""
        
        chunks = []
        
        # Find and preserve tables
        table_patterns = [
            # Markdown tables
            r'\|.*?\|.*?\n(?:\|[-:\s]*\|[-:\s]*\n)?(?:\|.*?\|.*?\n)*',
            # Lists with numerical data
            r'(?:^|\n)(?:\d+\.|[-*])\s+.*?(?:\d+(?:\.\d+)?(?:\s*(?:mg|g|ml|kg|units?|mmol|mEq)/?(?:kg|day|hour|min)?)?.*?)(?=\n\n|\n\d+\.|\n[-*]|\Z)',
            # Dosage tables/ranges
            r'(?i)(?:age|weight|dose|dosage|range).*?:.*?(?:\n.*?(?:\d+.*?(?:mg|g|ml|kg|units?).*?))*'
        ]
        
        processed_positions = set()
        
        for pattern in table_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
            
            for match in matches:
                start, end = match.span()
                
                # Skip overlapping matches
                if any(start <= pos_end and end >= pos_start 
                       for pos_start, pos_end in processed_positions):
                    continue
                
                table_content = match.group().strip()
                
                if len(table_content) < 50:  # Skip very small matches
                    continue
                
                # Add context before and after table
                context_before = self._get_context_before(content, start, 200)
                context_after = self._get_context_after(content, end, 200)
                
                full_content = f"{context_before}\n\n{table_content}\n\n{context_after}".strip()
                
                # Determine table type
                table_type = "numerical_data"
                if "|" in table_content:
                    table_type = "markdown_table"
                elif any(term in table_content.lower() for term in ['dose', 'dosage', 'mg/kg']):
                    table_type = "dosage_table"
                
                section = self._extract_section_from_position(content, start)
                
                # LangChain best practice: Simple metadata only
                simple_metadata = {
                    "document_type": metadata.get("document_type", "medical_text"),
                    "document_name": doc_name,
                    "section": section,
                    "chunk_strategy": "table_preservation",
                    "table_type": table_type,
                    "numerical_data": "true",  # String instead of boolean
                    "content_type": "reference_values" if "range" in table_content.lower() else "dosage_recommendation",
                    "authority": metadata.get("authority", ""),
                    "year": metadata.get("year", 0) if metadata.get("year") else 0
                }
                
                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_table_{table_type}_{len(chunks)}",
                    doc_id=doc_id,
                    content=full_content,
                    chunk_type=f"table_{table_type}",
                    section=section,
                    metadata=simple_metadata
                )
                
                chunks.append(chunk)
                processed_positions.add((start, end))
        
        return chunks

    def _optimize_medical_chunks(self, all_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Optimize chunks while preserving medical context and removing harmful duplicates."""
        
        if not all_chunks:
            return []
        
        # Group chunks by content similarity for deduplication
        content_signatures = {}
        for chunk in all_chunks:
            # Create signature from first and last 100 characters
            signature = (chunk.content[:100] + chunk.content[-100:]).lower()
            signature_key = hash(signature)
            
            if signature_key not in content_signatures:
                content_signatures[signature_key] = []
            content_signatures[signature_key].append(chunk)
        
        # Select best chunk from each group
        optimized_chunks = []
        for signature_key, chunk_group in content_signatures.items():
            if len(chunk_group) == 1:
                optimized_chunks.append(chunk_group[0])
            else:
                # Prefer medical context chunks, then clinical relationship chunks
                best_chunk = max(chunk_group, key=lambda c: (
                    1 if c.chunk_type.startswith('medical_') else 0,
                    1 if c.chunk_type == 'clinical_relationship' else 0,
                    len(c.content),
                    1 if c.metadata.get('context_preserved', False) else 0
                ))
                optimized_chunks.append(best_chunk)
        
        return optimized_chunks

    # =============== HELPER METHODS FOR CONTEXT PRESERVATION ===============

    def _expand_medical_context(self, content: str, start: int, end: int) -> str:
        """Expand context around medical content to preserve clinical meaning."""
        
        # Find sentence boundaries
        expanded_start = max(0, start - 300)
        expanded_end = min(len(content), end + 300)
        
        # Find natural sentence boundaries
        while expanded_start > 0 and content[expanded_start] not in '.!?\n':
            expanded_start -= 1
        
        while expanded_end < len(content) and content[expanded_end] not in '.!?\n':
            expanded_end += 1
            
        return content[expanded_start:expanded_end].strip()

    def _has_clinical_relationship(self, para1: str, para2: str) -> bool:
        """Check if two paragraphs have clinical relationships that should be preserved."""
        
        # Medical relationship indicators
        relationship_patterns = [
            # Sequential clinical steps
            ('first', 'then'), ('initial', 'subsequent'), ('before', 'after'),
            # Cause and effect
            ('if', 'then'), ('when', 'should'), ('may cause', 'monitor'),
            # Age/condition relationships  
            ('preterm', 'dose'), ('neonatal', 'consider'), ('pediatric', 'adjust'),
            # Clinical decision patterns
            ('contraindicated', 'alternative'), ('recommended', 'avoid'),
            ('monitor', 'adjust'), ('assess', 'modify')
        ]
        
        for pattern1, pattern2 in relationship_patterns:
            if pattern1 in para1 and pattern2 in para2:
                return True
        
        # Check for shared medical terms indicating related content
        medical_terms = ['tpn', 'parenteral', 'nutrition', 'dose', 'monitor', 'adverse', 'contraindic']
        shared_terms = sum(1 for term in medical_terms if term in para1 and term in para2)
        
        return shared_terms >= 2

    def _determine_clinical_focus(self, content: str) -> str:
        """Determine the primary clinical focus of content."""
        
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['dose', 'dosage', 'mg/kg', 'g/kg', 'calculate']):
            return 'dosage_calculation'
        elif any(term in content_lower for term in ['contraindic', 'caution', 'warning', 'adverse']):
            return 'safety_information'  
        elif any(term in content_lower for term in ['monitor', 'surveillance', 'lab', 'follow']):
            return 'monitoring_protocol'
        elif any(term in content_lower for term in ['preterm', 'neonatal', 'pediatric', 'infant']):
            return 'age_specific_guidance'
        else:
            return 'general_clinical_text'

    def _extract_section_from_position(self, content: str, position: int) -> str:
        """Extract section heading from content position."""
        
        # Look backwards for the nearest heading
        lines_before = content[:position].split('\n')
        for line in reversed(lines_before[-20:]):  # Check last 20 lines
            line = line.strip()
            if line.startswith('#'):
                return line.replace('#', '').strip()
            elif line.startswith('**') and line.endswith('**'):
                return line.replace('**', '').strip()
        
        return "General"

    def _extract_section_from_content(self, content: str) -> str:
        """Extract section from chunk content."""
        
        lines = content.split('\n')[:5]  # Check first 5 lines
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                return line.replace('#', '').strip()
        
        return "Clinical Content"

    def _optimize_chunk_boundaries(self, content: str) -> str:
        """Optimize chunk boundaries to avoid cutting medical concepts."""
        
        # If content ends mid-sentence, try to complete it
        if not content.rstrip().endswith(('.', '!', '?', ':', '\n')):
            # Add ellipsis to indicate continuation
            content = content.rstrip() + "..."
        
        return content

    def _extract_markdown_section(self, content: str) -> str:
        """Extract section from markdown content."""
        
        lines = content.split('\n')[:3]
        for line in lines:
            if line.startswith('#'):
                return line.replace('#', '').strip()
        
        return "Document Section"

    def _get_context_before(self, content: str, position: int, max_chars: int) -> str:
        """Get context before a position."""
        start = max(0, position - max_chars)
        context = content[start:position].strip()
        
        # Find sentence boundary
        if '.' in context:
            context = context[context.rfind('.') + 1:].strip()
        
        return context

    def _get_context_after(self, content: str, position: int, max_chars: int) -> str:
        """Get context after a position."""
        end = min(len(content), position + max_chars)
        context = content[position:end].strip()
        
        # Find sentence boundary
        if '.' in context:
            context = context[:context.find('.') + 1].strip()
        
        return context
