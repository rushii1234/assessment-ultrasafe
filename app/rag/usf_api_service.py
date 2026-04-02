"""
USF Third-Party API Service
Handles chat completions, embeddings, and reranking using the USF API.
"""

from typing import List, Dict, Any, Optional
import httpx

from app.core.config import get_settings
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


class USFAPIService:
    """Service for communicating with third-party USF API."""
    
    def __init__(self):
        self.base_url = settings.USF_BASE_URL
        self.api_key = settings.USF_API_KEY
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        self.chat_model = settings.USF_CHAT_MODEL
        self.embed_model = settings.USF_EMBED_MODEL
        self.rerank_model = settings.USF_RERANK_MODEL
        
        if not self.api_key:
            logger.warning("USF_API_KEY not configured. USF API calls will fail.")
    
    async def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Make a request to the USF API."""
        try:
            url = f"{self.base_url}{endpoint}"
            logger.debug(f"Making request to {url}")
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self.headers
                )
                response.raise_for_status()
                result = response.json()
                logger.debug(f"Received response from {endpoint}")
                return result
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling {endpoint}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error calling {endpoint}: {str(e)}")
            raise
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        web_search: bool = False,
        stream: bool = False
    ) -> str:
        """
        Generate chat completion using USF API.
        
        Args:
            messages: List of message objects with role and content
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            web_search: Enable web search in response
            stream: Stream the response (not fully implemented)
        
        Returns:
            Assistant message content
        """
        try:
            payload = {
                "model": self.chat_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "web_search": web_search,
                "stream": stream
            }
            
            logger.info("Calling USF chat completions API")
            response = await self._make_request(
                "/hiring/chat/completions",
                payload,
                timeout=settings.USF_CHAT_TIMEOUT
            )
            
            # Extract content from response
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0].get("message", {}).get("content", "")
                logger.info("Chat completion successful")
                return content
            else:
                logger.error("No choices in API response")
                raise ValueError("Invalid response structure from USF API")
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise
    
    async def embed(
        self,
        text: str
    ) -> List[float]:
        """
        Generate embeddings for text using USF API.
        
        Args:
            text: Text to embed
        
        Returns:
            List of floats representing the embedding
        """
        try:
            payload = {
                "model": self.embed_model,
                "input": text
            }
            
            logger.debug(f"Embedding text: {text[:50]}...")
            response = await self._make_request(
                "/embed/embeddings",
                payload,
                timeout=settings.USF_EMBED_TIMEOUT
            )
            
            # Extract embedding from response
            if "data" in response and len(response["data"]) > 0:
                embedding = response["data"][0].get("embedding", [])
                logger.debug("Embedding generated successfully")
                return embedding
            else:
                logger.error("No embeddings in API response")
                raise ValueError("Invalid response structure from USF API")
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    async def embed_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embeddings
        """
        try:
            payload = {
                "model": self.embed_model,
                "input": texts
            }
            
            logger.debug(f"Batch embedding {len(texts)} texts")
            response = await self._make_request(
                "/embed/embeddings",
                payload,
                timeout=settings.USF_EMBED_TIMEOUT
            )
            
            # Extract embeddings from response
            if "data" in response:
                # Sort by index to maintain order
                embeddings = sorted(response["data"], key=lambda x: x.get("index", 0))
                result = [e.get("embedding", []) for e in embeddings]
                logger.debug(f"Batch embedding completed for {len(result)} texts")
                return result
            else:
                logger.error("No embeddings in API response")
                raise ValueError("Invalid response structure from USF API")
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            raise
    
    async def rerank(
        self,
        query: str,
        texts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance using USF API.
        
        Args:
            query: Query string
            texts: List of text chunks to rerank
        
        Returns:
            List of reranked results with scores
        """
        try:
            payload = {
                "model": self.rerank_model,
                "query": query,
                "texts": texts
            }
            
            logger.debug(f"Reranking {len(texts)} texts for query: {query}")
            response = await self._make_request(
                "/embed/reranker",
                payload,
                timeout=settings.USF_RERANK_TIMEOUT
            )
            
            # Extract reranking results
            if "results" in response:
                results = response["results"]
                logger.debug(f"Reranking completed for {len(results)} texts")
                return results
            else:
                logger.error("No results in reranker response")
                raise ValueError("Invalid response structure from USF API")
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check if USF API is accessible."""
        try:
            # Try a simple embed request to verify connectivity
            await self.embed("health")
            logger.info("USF API health check passed")
            return True
        except Exception as e:
            logger.error(f"USF API health check failed: {str(e)}")
            return False
