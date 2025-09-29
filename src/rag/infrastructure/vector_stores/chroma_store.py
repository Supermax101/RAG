"""
ChromaDB vector store implementation.
"""
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from ...core.interfaces.embeddings import VectorStore
from ...core.models.documents import DocumentChunk
from ...config.settings import settings


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store implementation."""
    
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.chroma_collection_name
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Disable telemetry
            import chromadb.telemetry
            chromadb.telemetry.telemetry = None
            
            self.client = chromadb.PersistentClient(
                path=str(settings.chromadb_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except ValueError:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Document chunks for RAG"}
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")
    
    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        doc_name: str
    ) -> None:
        """Add document chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []
        
        for chunk, embedding in zip(chunks, embeddings):
            # Use chunk_id or generate one if missing
            chunk_id = chunk.chunk_id or str(uuid.uuid4())
            
            ids.append(chunk_id)
            documents.append(chunk.content)
            embeddings_list.append(embedding)
            
            metadata = {
                "doc_id": chunk.doc_id,
                "document_name": doc_name,
                "chunk_type": chunk.chunk_type,
                "section": chunk.section or "",
                **chunk.metadata
            }
            
            if chunk.page_num is not None:
                metadata["page_num"] = chunk.page_num
                
            metadatas.append(metadata)
        
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
        except Exception as e:
            raise RuntimeError(f"Failed to add chunks to ChromaDB: {e}")
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        try:
            where_clause = filters or {}
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause if where_clause else None
            )
            
            # Convert ChromaDB results to our format
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    result = {
                        "chunk_id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                        "doc_id": results["metadatas"][0][i].get("doc_id", ""),
                        "document_name": results["metadatas"][0][i].get("document_name", "Unknown"),
                        "chunk_type": results["metadatas"][0][i].get("chunk_type", "text"),
                        "section": results["metadatas"][0][i].get("section", ""),
                        "page_num": results["metadatas"][0][i].get("page_num"),
                        "metadata": results["metadatas"][0][i]
                    }
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            raise RuntimeError(f"Failed to search ChromaDB: {e}")
    
    async def delete_document(self, doc_id: str) -> None:
        """Delete all chunks for a document."""
        try:
            self.collection.delete(
                where={"doc_id": doc_id}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to delete document from ChromaDB: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            
            if count == 0:
                return {
                    "total_chunks": 0,
                    "total_documents": 0,
                    "collection_name": self.collection_name,
                    "sample_size": 0
                }
            
            # For accurate document count, get all metadata (or a larger sample)
            # Since we need accurate counts for 52 documents, get more samples
            sample_limit = min(5000, count)  # Sample more to get better accuracy
            sample_results = self.collection.get(limit=sample_limit)
            
            # Count unique documents
            unique_docs = set()
            if sample_results["metadatas"]:
                for metadata in sample_results["metadatas"]:
                    if metadata.get("doc_id"):
                        unique_docs.add(metadata["doc_id"])
                    elif metadata.get("document_name"):  # Fallback to document_name
                        unique_docs.add(metadata["document_name"])
            
            # If we sampled all data and still have low unique count, something's wrong
            estimated_total_docs = len(unique_docs)
            if sample_limit < count and len(unique_docs) > 0:
                # Rough estimation: if we sampled X% and found Y docs, estimate total
                sampling_ratio = sample_limit / count
                estimated_total_docs = max(len(unique_docs), int(len(unique_docs) / sampling_ratio))
            
            return {
                "total_chunks": count,
                "total_documents": estimated_total_docs,
                "unique_docs_in_sample": len(unique_docs),
                "collection_name": self.collection_name,
                "sample_size": len(sample_results["ids"]) if sample_results["ids"] else 0
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get ChromaDB stats: {e}")
    
    def reset_collection(self) -> None:
        """Reset the collection (for development only)."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks for RAG"}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to reset collection: {e}")
