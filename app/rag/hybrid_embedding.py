"""
Hybrid Embedding Service
Supports both local embeddings (Sentence Transformers) and USF API embeddings.
"""

from typing import List, Optional
import asyncio

from app.core.config import get_settings
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


class HybridEmbeddingService:
    """Service for generating embeddings using local or API models."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HybridEmbeddingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.use_usf_api = settings.USE_USF_API
        self.model = None
        
        if self.use_usf_api:
            logger.info("Using USF API for embeddings")
            from app.rag.usf_api_service import USFAPIService
            self.usf_service = USFAPIService()
        else:
            logger.info(f"Using local embedding model: {settings.EMBEDDING_MODEL}")
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logger.info("Local embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load local embedding model: {str(e)}")
                raise
        
        self._initialized = True
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text (synchronous).
        For async version, use embed_async().
        """
        if self.use_usf_api:
            # For sync context, we need to run async in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(self.usf_service.embed(text))
        else:
            try:
                embedding = self.model.encode(text, convert_to_numpy=False)
                return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                raise
    
    async def embed_async(self, text: str) -> List[float]:
        """Generate embedding for a single text (async)."""
        if self.use_usf_api:
            return await self.usf_service.embed(text)
        else:
            try:
                embedding = self.model.encode(text, convert_to_numpy=False)
                return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (synchronous).
        For async version, use embed_batch_async().
        """
        if self.use_usf_api:
            # For sync context, we need to run async in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(self.usf_service.embed_batch(texts))
        else:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=False)
                return [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {str(e)}")
                raise
    
    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (async)."""
        if self.use_usf_api:
            return await self.usf_service.embed_batch(texts)
        else:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=False)
                return [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {str(e)}")
                raise
