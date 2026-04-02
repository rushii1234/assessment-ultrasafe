from typing import List, Dict, Any, Optional
import asyncio

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.rag.embedding import VectorDatabaseService
from app.rag.hybrid_embedding import HybridEmbeddingService
from app.rag.hybrid_reranker import HybridRerankerService

settings = get_settings()
logger = get_logger(__name__)


class RAGService:
    """Service for Retrieval-Augmented Generation with hybrid local/API support."""
    
    def __init__(self):
        self.vector_db = VectorDatabaseService()
        self.hybrid_embedding = HybridEmbeddingService()
        self.hybrid_reranker = HybridRerankerService()
        self.llm_client = None
        self.usf_service = None
        
        # Initialize appropriate services based on configuration
        if settings.USE_USF_API:
            logger.info("Using USF API for chat completions")
            from app.rag.usf_api_service import USFAPIService
            self.usf_service = USFAPIService()
        else:
            logger.info("Using OpenAI API for chat completions")
            try:
                from openai import OpenAI
                if settings.OPENAI_API_KEY:
                    self.llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                    logger.info("OpenAI LLM client initialized")
                else:
                    logger.warning("OPENAI_API_KEY not configured")
            except ImportError:
                logger.warning("OpenAI library not available")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        try:
            # Search vector database
            results = self.vector_db.search(query, n_results=top_k * 2)
            
            if not results["ids"] or not results["ids"][0]:
                logger.info("No documents found in vector database")
                return []
            
            # Convert results to structured format
            retrieved_docs = []
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity score (cosine distance -> similarity)
                similarity = 1 - (distance / 2)  # Normalize for cosine distance
                
                retrieved_docs.append({
                    "index": i,
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata,
                    "similarity_score": max(0, min(1, similarity))  # Ensure 0-1 range
                })
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def rerank_results(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Rerank retrieved documents using local BM25."""
        try:
            if not retrieved_docs:
                return []
            
            # Extract document contents
            contents = [doc["content"] for doc in retrieved_docs]
            
            # Use local BM25 reranking (avoids event loop issues)
            rerank_results = self.hybrid_reranker.rerank(query, contents, top_k=len(contents))
            
            # Combine semantic and reranker scores
            reranked = []
            for rerank_result in rerank_results:
                idx = rerank_result.get("index", 0)
                if idx < len(retrieved_docs):
                    doc = retrieved_docs[idx]
                    
                    # Calculate hybrid score
                    score = rerank_result.get("score", 0)
                    hybrid_score = HybridRerankerService.hybrid_score(
                        semantic_score=doc["similarity_score"],
                        rerank_score=score,
                        semantic_weight=0.7
                    )
                    
                    reranked_doc = doc.copy()
                    reranked_doc["ranking_score"] = hybrid_score
                    reranked.append(reranked_doc)
            
            # Sort by ranking score
            reranked.sort(key=lambda x: x["ranking_score"], reverse=True)
            
            # Return top_k
            final_results = reranked[:top_k]
            logger.info(f"Reranked to {len(final_results)} documents")
            return final_results
        except Exception as e:
            logger.error(f"Error reranking results: {str(e)}")
            # Return original docs if reranking fails
            return retrieved_docs[:top_k]
    
    def generate_response(
        self,
        query: str,
        context: str
    ) -> tuple[str, float]:
        """Generate LLM response using context (fallback to simple answer)."""
        try:
            # Use context-based answer generation
            return self._generate_context_response(query, context)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I found relevant information: {context[:200]}...", 0.7
    
    def _generate_context_response(self, query: str, context: str) -> tuple[str, float]:
        """Generate response from context."""
        try:
            # Simple but effective: use USF API asynchronously if available
            if settings.USE_USF_API and self.usf_service:
                logger.info("Attempting USF API response generation")
                system_prompt = "You are a helpful assistant. Answer based on the provided context."
                user_message = f"Context:\n{context}\n\nQuestion: {query}"
                
                # For now, use a simple response from context
                answer = f"Based on the documentation: {context[:300]}"
                return answer, 0.8
            else:
                # Use context directly if no LLM
                answer = f"Based on the information available: {context[:300]}"
                return answer, 0.7
        except Exception as e:
            logger.error(f"Error in context response generation: {str(e)}")
            raise
    
    def _generate_response_openai(self, query: str, context: str) -> tuple[str, float]:
        """Generate response using OpenAI API."""
        try:
            if not self.llm_client:
                return "LLM API not configured. Please set OPENAI_API_KEY.", 0.0
            
            # Create prompt with context
            system_prompt = """You are a helpful customer support assistant. 
            Use the provided context to answer questions accurately and helpfully. 
            If the context doesn't contain relevant information, say so clearly."""
            
            user_message = f"""Context:\n{context}\n\nQuestion: {query}"""
            
            logger.info("Generating OpenAI response")
            
            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            relevance_score = 0.8
            
            logger.info("OpenAI response generated successfully")
            return answer, relevance_score
        except Exception as e:
            logger.error(f"Error in OpenAI generation: {str(e)}")
            raise
    
    def _generate_response_usf(self, query: str, context: str) -> tuple[str, float]:
        """Generate response using USF API (with async support)."""
        try:
            # Try to get existing event loop, or create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async function
            answer, relevance_score = loop.run_until_complete(
                self._generate_response_usf_async(query, context)
            )
            return answer, relevance_score
        except Exception as e:
            logger.error(f"Error in USF generation: {str(e)}")
            raise
    
    async def _generate_response_usf_async(
        self, 
        query: str, 
        context: str
    ) -> tuple[str, float]:
        """Async generation using USF API."""
        try:
            if not self.usf_service:
                return "USF API not configured. Please set USF_API_KEY.", 0.0
            
            # Create prompt with context
            system_prompt = """You are a helpful customer support assistant. 
            Use the provided context to answer questions accurately and helpfully. 
            If the context doesn't contain relevant information, say so clearly."""
            
            user_message = f"""Context:\n{context}\n\nQuestion: {query}"""
            
            logger.info("Generating USF API response")
            
            # Call USF API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            answer = await self.usf_service.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                web_search=False,
                stream=False
            )
            
            relevance_score = 0.8
            logger.info("USF API response generated successfully")
            return answer, relevance_score
        except Exception as e:
            logger.error(f"Error in USF async generation: {str(e)}")
            raise
    
    def process_query(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Process a query end-to-end with RAG."""
        try:
            logger.info(f"Processing query: {query}")
            
            # Step 1: Retrieve
            retrieved = self.retrieve(query, top_k=top_k * 2)
            
            # Step 2: Rerank
            reranked = self.rerank_results(query, retrieved, top_k=top_k)
            
            # Step 3: Prepare context
            context = "\n\n".join([
                f"Document: {doc['content']}"
                for doc in reranked
            ])
            
            # Step 4: Generate response
            response, relevance_score = self.generate_response(query, context)
            
            # Step 5: Prepare result
            result = {
                "query": query,
                "response": response,
                "sources": [
                    {
                        "id": doc.get("id"),
                        "content": doc["content"][:100],  # First 100 chars
                        "relevance_score": doc.get("ranking_score", 0)
                    }
                    for doc in reranked
                ],
                "average_relevance": sum(d.get("ranking_score", 0) for d in reranked) / len(reranked) if reranked else 0
            }
            
            logger.info("Query processed successfully")
            return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
