import json
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using sentence transformers."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self._initialized = True
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            embedding = self.model.encode(text, convert_to_numpy=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=False)
            return [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise


class VectorDatabaseService:
    """Service for managing vector database with Chroma."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorDatabaseService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        try:
            logger.info(f"Initializing Chroma vector DB at {settings.VECTOR_DB_PATH}")
            self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
            self.embedding_service = EmbeddingService()
            self.collection = None
            self._initialized = True
            logger.info("Vector database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            raise
    
    def get_or_create_collection(self, name: str = "documents") -> Any:
        """Get or create a collection."""
        try:
            self.collection = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{name}' ready")
            return self.collection
        except Exception as e:
            logger.error(f"Error managing collection: {str(e)}")
            raise
    
    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the vector database."""
        try:
            if self.collection is None:
                self.get_or_create_collection()
            
            # Generate embeddings
            embeddings = self.embedding_service.embed_batch(documents)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas or [{"index": i} for i in range(len(ids))]
            )
            logger.info(f"Added {len(ids)} documents to vector database")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Search for similar documents."""
        try:
            if self.collection is None:
                self.get_or_create_collection()
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            logger.info(f"Retrieved {len(results['ids'][0]) if results['ids'] else 0} documents")
            return results
        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}")
            raise
    
    def delete_document(self, document_id: str) -> None:
        """Delete a document from the vector database."""
        try:
            if self.collection is None:
                self.get_or_create_collection()
            
            self.collection.delete(ids=[document_id])
            logger.info(f"Deleted document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise
    
    def clear_collection(self) -> None:
        """Clear all documents from collection."""
        try:
            if self.collection is None:
                self.get_or_create_collection()
            
            # Get all IDs and delete
            items = self.collection.get()
            if items["ids"]:
                self.collection.delete(ids=items["ids"])
            logger.info("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise
