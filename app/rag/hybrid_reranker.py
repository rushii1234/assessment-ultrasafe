"""
Hybrid Reranker Service
Supports both local BM25 and USF API reranking.
"""

from typing import List, Dict, Any, Optional
import asyncio

from app.core.config import get_settings
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


class HybridRerankerService:
    """Service for reranking documents using local or API models."""
    
    def __init__(self):
        self.use_usf_api = settings.USE_USF_API
        self.bm25 = None
        self.documents = []
        
        if self.use_usf_api:
            logger.info("Using USF API for reranking")
            from app.rag.usf_api_service import USFAPIService
            self.usf_service = USFAPIService()
        else:
            logger.info("Using local BM25 reranker")
            try:
                from rank_bm25 import BM25Okapi
                self.BM25Okapi = BM25Okapi
                logger.info("BM25 model imported successfully")
            except ImportError:
                logger.error("rank_bm25 package not installed")
                raise
    
    def fit(self, documents: List[str]) -> None:
        """Fit BM25 model with documents (local reranker only)."""
        if self.use_usf_api:
            logger.debug("Skipping fit for USF API reranker")
            return
        
        try:
            # Tokenize documents
            tokenized_docs = [doc.lower().split() for doc in documents]
            self.bm25 = self.BM25Okapi(tokenized_docs)
            self.documents = documents
            logger.info(f"BM25 model fitted with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error fitting BM25 model: {str(e)}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using local BM25 (avoid async issues in sync context).
        """
        # Always use local BM25 in sync context to avoid event loop conflicts
        return self._rerank_bm25(query, documents, top_k)
    
    async def rerank_async(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance (async)."""
        if self.use_usf_api:
            return await self.usf_service.rerank(query, documents)
        else:
            return self._rerank_bm25(query, documents, top_k)
    
    def _rerank_bm25(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Internal BM25 reranking logic."""
        try:
            if not self.bm25:
                # Fit model if not already fitted
                self.fit(documents)
            
            # Get BM25 scores
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            
            # Create result list
            results = []
            for idx, score in enumerate(scores):
                results.append({
                    "index": idx,
                    "score": float(score),
                    "document": documents[idx] if idx < len(documents) else ""
                })
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            reranked = results[:top_k]
            
            logger.info(f"Reranked {len(documents)} documents using BM25")
            return reranked
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            raise
    
    @staticmethod
    def hybrid_score(
        semantic_score: float,
        rerank_score: float,
        semantic_weight: float = 0.7
    ) -> float:
        """
        Calculate hybrid score combining semantic and reranker scores.
        
        Args:
            semantic_score: Semantic similarity score (0-1)
            rerank_score: Reranker score (varies by implementation)
            semantic_weight: Weight for semantic score
        
        Returns:
            Normalized hybrid score (0-1)
        """
        # Normalize rerank score (typically 0-infinity for BM25)
        normalized_rerank = min(rerank_score / 10.0, 1.0)  # Cap at 1.0
        
        # Calculate weighted average
        hybrid = (semantic_weight * semantic_score) + ((1 - semantic_weight) * normalized_rerank)
        return min(hybrid, 1.0)
