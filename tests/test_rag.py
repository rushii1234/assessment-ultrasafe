from app.rag.embedding import EmbeddingService, VectorDatabaseService
from app.rag.reranker import RerankerService
import pytest


class TestEmbedding:
    """Test embedding service."""
    
    def test_embedding_service_initialization(self):
        """Test embedding service initialization."""
        service = EmbeddingService()
        assert service.model is not None
    
    def test_embed_single_text(self):
        """Test embedding a single text."""
        service = EmbeddingService()
        embedding = service.embed("Test text for embedding")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)
    
    def test_embed_batch(self):
        """Test batch embedding."""
        service = EmbeddingService()
        texts = ["First text", "Second text", "Third text"]
        embeddings = service.embed_batch(texts)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) > 0


class TestVectorDatabase:
    """Test vector database service."""
    
    def test_vector_db_initialization(self):
        """Test vector database initialization."""
        db = VectorDatabaseService()
        assert db.client is not None
        assert db.embedding_service is not None
    
    def test_get_or_create_collection(self):
        """Test collection creation."""
        db = VectorDatabaseService()
        collection = db.get_or_create_collection("test_collection")
        assert collection is not None
        assert db.collection is not None
    
    def test_add_documents(self):
        """Test adding documents to vector database."""
        db = VectorDatabaseService()
        db.get_or_create_collection("test_collection")
        
        ids = ["doc1", "doc2"]
        documents = ["First document", "Second document"]
        metadatas = [{"title": "Doc 1"}, {"title": "Doc 2"}]
        
        db.add_documents(ids, documents, metadatas)
        assert db.collection is not None
    
    def test_search(self):
        """Test searching in vector database."""
        db = VectorDatabaseService()
        db.get_or_create_collection("test_search")
        
        # Add documents
        ids = ["doc1", "doc2"]
        documents = [
            "Python is a programming language",
            "Java is also a programming language"
        ]
        db.add_documents(ids, documents)
        
        # Search
        results = db.search("programming language", n_results=2)
        assert "ids" in results
        assert "documents" in results


class TestReranker:
    """Test reranker service."""
    
    def test_reranker_initialization(self):
        """Test reranker initialization."""
        reranker = RerankerService()
        assert reranker.bm25 is None
        assert reranker.documents == []
    
    def test_fit_bm25(self):
        """Test fitting BM25 model."""
        reranker = RerankerService()
        documents = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "NLP processes natural language"
        ]
        
        reranker.fit(documents)
        assert reranker.bm25 is not None
        assert len(reranker.documents) == 3
    
    def test_rerank(self):
        """Test reranking documents."""
        reranker = RerankerService()
        documents = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "NLP processes natural language"
        ]
        
        results = reranker.rerank("machine learning", documents, top_k=2)
        assert isinstance(results, list)
        assert len(results) <= 2
        assert all("score" in r and "document" in r for r in results)
    
    def test_hybrid_score(self):
        """Test hybrid scoring."""
        score = RerankerService.hybrid_score(
            semantic_score=0.8,
            bm25_score=5.0,
            semantic_weight=0.7
        )
        assert 0 <= score <= 1
        assert score > 0  # Should have some value
