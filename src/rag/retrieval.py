import logging
from typing import List, Dict, Any, Tuple
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class DocumentChunker:
    """Handles document chunking strategies"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentences to maintain context
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_size(self, text: str) -> List[str]:
        """
        Chunk text by character count with overlap
        
        Args:
            text: Input text
            
        Returns:
            List of overlapping chunks
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunks.append(text[i:i + self.chunk_size])
        
        return chunks
    
    def chunk_document(self, text: str, strategy: str = "sentence") -> List[str]:
        """
        Chunk document using specified strategy
        
        Args:
            text: Input text
            strategy: 'sentence' or 'size'
            
        Returns:
            List of chunks
        """
        if strategy == "sentence":
            return self.chunk_by_sentences(text)
        elif strategy == "size":
            return self.chunk_by_size(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")


class HybridRetriever:
    """Hybrid retrieval system using keywords and embeddings"""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.vectorizer = None
        self.chunks = []
        self.chunk_embeddings = []
    
    def add_documents(self, chunks: List[str], embeddings: List[List[float]]):
        """
        Add documents with their embeddings to the retriever
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
        """
        self.chunks = chunks
        self.chunk_embeddings = embeddings
        
        # Initialize TF-IDF vectorizer for keyword search
        if chunks:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                lowercase=True
            )
            self.vectorizer.fit(chunks)
    
    def keyword_search(self, query: str, k: int = None) -> Tuple[List[str], List[float]]:
        """
        Retrieve documents using TF-IDF keyword search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            Tuple of (documents, scores)
        """
        k = k or self.top_k
        
        if not self.vectorizer or not self.chunks:
            return [], []
        
        query_vec = self.vectorizer.transform([query])
        chunk_vecs = self.vectorizer.transform(self.chunks)
        
        similarities = cosine_similarity(query_vec, chunk_vecs)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = [self.chunks[i] for i in top_indices]
        scores = [float(similarities[i]) for i in top_indices]
        
        return results, scores
    
    def embedding_search(self, query_embedding: List[float], k: int = None) -> Tuple[List[str], List[float]]:
        """
        Retrieve documents using embedding similarity
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (documents, scores)
        """
        k = k or self.top_k
        
        if not self.chunk_embeddings:
            return [], []
        
        query_vec = np.array([query_embedding])
        chunk_vecs = np.array(self.chunk_embeddings)
        
        similarities = cosine_similarity(query_vec, chunk_vecs)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = [self.chunks[i] for i in top_indices]
        scores = [float(similarities[i]) for i in top_indices]
        
        return results, scores
    
    def hybrid_search(self, query: str, query_embedding: List[float], k: int = None, alpha: float = 0.5) -> Tuple[List[str], List[float]]:
        """
        Hybrid search combining keyword and embedding-based retrieval
        
        Args:
            query: Search query
            query_embedding: Query embedding vector
            k: Number of results to return
            alpha: Weight for keyword search (1-alpha for embedding search)
            
        Returns:
            Tuple of (documents, scores)
        """
        k = k or self.top_k
        
        # Get results from both methods
        keyword_docs, keyword_scores = self.keyword_search(query, k=k*2)
        embedding_docs, embedding_scores = self.embedding_search(query_embedding, k=k*2)
        
        # Combine and deduplicate
        combined = {}
        
        for doc, score in zip(keyword_docs, keyword_scores):
            combined[doc] = combined.get(doc, 0) + alpha * score
        
        for doc, score in zip(embedding_docs, embedding_scores):
            combined[doc] = combined.get(doc, 0) + (1-alpha) * score
        
        # Sort by combined score
        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        results = [doc for doc, _ in sorted_docs[:k]]
        scores = [score for _, score in sorted_docs[:k]]
        
        return results, scores


class ContextualCompressor:
    """Compresses retrieved documents for contextual relevance"""
    
    def __init__(self, max_context_length: int = 2000):
        self.max_context_length = max_context_length
    
    def compress_documents(self, query: str, documents: List[str]) -> str:
        """
        Compress and concatenate documents for context
        
        Args:
            query: Original query
            documents: List of retrieved documents
            
        Returns:
            Compressed context string
        """
        compressed = ""
        
        for doc in documents:
            if len(compressed) + len(doc) < self.max_context_length:
                compressed += doc + "\n\n"
            else:
                # Truncate to fit
                remaining = self.max_context_length - len(compressed)
                if remaining > 100:  # Only add if meaningful length remains
                    compressed += doc[:remaining-50] + "..."
                break
        
        return compressed.strip()
    
    def extract_relevant_sentences(self, query: str, document: str, num_sentences: int = 3) -> str:
        """
        Extract most relevant sentences from document
        
        Args:
            query: Query string
            document: Document text
            num_sentences: Number of sentences to extract
            
        Returns:
            Extracted sentences
        """
        sentences = sent_tokenize(document)
        
        # Score sentences by relevance to query
        query_words = set(query.lower().split())
        
        scored_sentences = []
        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)
            scored_sentences.append((sent, overlap))
        
        # Sort by score and return top sentences
        top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:num_sentences]
        return " ".join([sent for sent, _ in top_sentences])


class CrossDocumentSynthesizer:
    """Synthesizes information across multiple documents"""
    
    def __init__(self):
        pass
    
    def extract_key_points(self, documents: List[str]) -> Dict[str, Any]:
        """
        Extract key points from multiple documents
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary with extracted information
        """
        key_points = {
            "documents": len(documents),
            "total_length": sum(len(doc) for doc in documents),
            "unique_concepts": set(),
            "themes": []
        }
        
        # Simple keyword extraction (can be enhanced with NLP)
        for doc in documents:
            words = re.findall(r'\b[a-z]{4,}\b', doc.lower())
            key_points["unique_concepts"].update(words[:10])
        
        return key_points
    
    def synthesize_findings(self, documents: List[str], query: str) -> str:
        """
        Synthesize findings across documents
        
        Args:
            documents: List of documents
            query: Original query
            
        Returns:
            Synthesized summary
        """
        if not documents:
            return ""
        
        synthesis = f"Based on {len(documents)} sources about '{query}':\n\n"
        
        key_points = self.extract_key_points(documents)
        
        synthesis += f"Total reviewed content: {key_points['total_length']} characters\n"
        synthesis += f"Unique concepts identified: {', '.join(list(key_points['unique_concepts'])[:5])}\n\n"
        
        synthesis += "Key findings:\n"
        for i, doc in enumerate(documents[:3], 1):
            first_sentence = sent_tokenize(doc)[0] if sent_tokenize(doc) else doc[:100]
            synthesis += f"{i}. {first_sentence}\n"
        
        return synthesis
