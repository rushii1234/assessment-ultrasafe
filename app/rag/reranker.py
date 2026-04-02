from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class RerankerService:
    """Service for reranking retrieval results."""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
    
    def fit(self, documents: List[str]) -> None:
        """Fit BM25 model with documents."""
        try:
            # Tokenize documents
            tokenized_docs = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
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
        """Rerank documents based on BM25 scores."""
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
            
            logger.info(f"Reranked {len(documents)} documents, returning top {len(reranked)}")
            return reranked
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            raise
    
    @staticmethod
    def hybrid_score(
        semantic_score: float,
        bm25_score: float,
        semantic_weight: float = 0.7
    ) -> float:
        """Calculate hybrid score combining semantic and BM25 scores."""
        # Normalize BM25 score (typically 0-infinity)
        normalized_bm25 = min(bm25_score / 10.0, 1.0)  # Cap at 1.0
        
        # Calculate weighted average
        hybrid = (semantic_weight * semantic_score) + ((1 - semantic_weight) * normalized_bm25)
        return min(hybrid, 1.0)
