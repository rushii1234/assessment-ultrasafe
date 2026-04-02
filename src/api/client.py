import httpx
import json
from typing import Any, Dict, List, Optional
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class APIClient:
    """Client for interacting with custom API models"""
    
    def __init__(self):
        self.base_url = settings.api_base_url
        self.api_key = settings.api_key
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a chat completion request
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (defaults to usf1-mini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            web_search: Enable web search capability
            tools: List of tools the model can use
            stream: Enable streaming response
            
        Returns:
            Response from the API
        """
        url = f"{self.base_url}/hiring/chat/completions"
        
        payload = {
            "model": model or settings.chat_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.temperature,
            "max_tokens": max_tokens or settings.max_tokens,
            "web_search": web_search,
            "stream": stream
        }
        
        if tools:
            payload["tools"] = tools
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Chat completion request failed: {str(e)}")
            raise
    
    async def get_embeddings(
        self,
        text_input: str | List[str],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get embeddings for text
        
        Args:
            text_input: Text or list of texts to embed
            model: Model name (defaults to usf1-embed)
            
        Returns:
            Embedding response
        """
        url = f"{self.base_url}/embed/embeddings"
        
        payload = {
            "model": model or settings.embedding_model,
            "input": text_input
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Embedding request failed: {str(e)}")
            raise
    
    async def rerank_documents(
        self,
        query: str,
        texts: List[str],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: Query string
            texts: List of texts to rerank
            model: Model name (defaults to usf1-rerank)
            
        Returns:
            Reranking response with relevance scores
        """
        url = f"{self.base_url}/embed/reranker"
        
        payload = {
            "model": model or settings.rerank_model,
            "query": query,
            "texts": texts
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Reranking request failed: {str(e)}")
            raise


# Singleton instance
api_client = APIClient()
