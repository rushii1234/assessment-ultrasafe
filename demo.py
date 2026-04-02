#!/usr/bin/env python3
"""
Demo and test script for the Multi-Agent Research Assistant.

This script demonstrates:
1. Document chunking
2. Hybrid retrieval
3. Contextual compression
4. Complete workflow execution
5. RAG operations
"""

import asyncio
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from src.config import settings
from src.workflow.orchestrator import orchestrator, WorkflowInput
from src.rag.retrieval import (
    DocumentChunker,
    HybridRetriever,
    ContextualCompressor,
    CrossDocumentSynthesizer
)
from src.agents.base import ResearchQuery
from src.api.client import api_client


# Sample documents for testing
SAMPLE_PAPERS = [
    {
        "title": "Deep Learning in Medical Imaging",
        "abstract": "This paper reviews recent advances in deep learning applications for medical image analysis. We examine convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformer architectures. Key findings include achieving >95% accuracy in detecting certain cancers and significant reduction in training time through transfer learning."
    },
    {
        "title": "Machine Learning for Personalized Medicine",
        "abstract": "Personalized medicine aims to tailor treatment plans based on individual patient characteristics. This study evaluates ML approaches for predicting treatment response in cancer patients. Ensemble methods combining genomic and clinical features achieve 78% accuracy in treatment response prediction."
    },
    {
        "title": "Neural Networks for Diagnostic Support",
        "abstract": "Clinical decision support systems using neural networks show promise in improving diagnostic accuracy. This systematic review analyzes 150 studies and finds that neural networks match or exceed radiologist performance in image interpretation tasks across different medical specialties."
    }
]


async def demo_document_chunking():
    """Demonstrate document chunking strategies"""
    logger.info("=" * 60)
    logger.info("DEMO 1: Document Chunking")
    logger.info("=" * 60)
    
    chunker = DocumentChunker(chunk_size=300, overlap=50)
    
    # Test with sample paper
    paper = SAMPLE_PAPERS[0]
    logger.info(f"Chunking paper: {paper['title']}")
    logger.info(f"Original length: {len(paper['abstract'])} characters")
    
    # Sentence-based chunking
    chunks_sentence = chunker.chunk_by_sentences(paper['abstract'])
    logger.info(f"\nSentence-based chunking:")
    logger.info(f"  Number of chunks: {len(chunks_sentence)}")
    for i, chunk in enumerate(chunks_sentence, 1):
        logger.info(f"  Chunk {i}: {chunk[:60]}...")
    
    # Size-based chunking
    chunks_size = chunker.chunk_by_size(paper['abstract'])
    logger.info(f"\nSize-based chunking:")
    logger.info(f"  Number of chunks: {len(chunks_size)}")
    for i, chunk in enumerate(chunks_size[:3], 1):
        logger.info(f"  Chunk {i}: {chunk[:60]}...")


async def demo_hybrid_retrieval():
    """Demonstrate hybrid retrieval system"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: Hybrid Retrieval System")
    logger.info("=" * 60)
    
    # Prepare documents
    documents = [p['abstract'] for p in SAMPLE_PAPERS]
    
    logger.info(f"Loading {len(documents)} documents into retriever...")
    
    # Create mock embeddings (in real system, would use API)
    import numpy as np
    embeddings = [np.random.randn(768).tolist() for _ in documents]
    
    retriever = HybridRetriever(top_k=2)
    retriever.add_documents(documents, embeddings)
    logger.info("Retriever initialized")
    
    # Perform different searches
    test_queries = [
        "neural networks in diagnosis",
        "personalized medicine treatment",
        "deep learning accuracy"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        
        # Keyword search
        kw_results, kw_scores = retriever.keyword_search(query, k=2)
        logger.info(f"  Keyword search results: {len(kw_results)} documents")
        for doc, score in zip(kw_results, kw_scores):
            logger.info(f"    Score: {score:.3f} - {doc[:50]}...")
        
        # Embedding search (mock)
        emb_results, emb_scores = retriever.embedding_search(embeddings[0], k=2)
        logger.info(f"  Embedding search results: {len(emb_results)} documents")


async def demo_contextual_compression():
    """Demonstrate contextual compression"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 3: Contextual Compression")
    logger.info("=" * 60)
    
    compressor = ContextualCompressor(max_context_length=500)
    
    documents = [p['abstract'] for p in SAMPLE_PAPERS]
    query = "neural networks in healthcare"
    
    logger.info(f"Original documents total length: {sum(len(d) for d in documents)} characters")
    
    # Compress
    compressed = compressor.compress_documents(query, documents)
    logger.info(f"Compressed context length: {len(compressed)} characters")
    logger.info(f"Compression ratio: {len(compressed) / sum(len(d) for d in documents):.2%}")
    
    logger.info(f"\nCompressed context:\n{compressed[:300]}...")
    
    # Extract relevant sentences
    logger.info(f"\nExtracting relevant sentences for query: '{query}'")
    relevant = compressor.extract_relevant_sentences(
        query=query,
        document=SAMPLE_PAPERS[0]['abstract'],
        num_sentences=2
    )
    logger.info(f"Relevant sentences:\n{relevant}")


async def demo_cross_document_synthesis():
    """Demonstrate cross-document synthesis"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 4: Cross-Document Synthesis")
    logger.info("=" * 60)
    
    synthesizer = CrossDocumentSynthesizer()
    
    documents = [p['abstract'] for p in SAMPLE_PAPERS]
    query = "machine learning in healthcare"
    
    # Extract key points
    logger.info(f"Analyzing {len(documents)} documents...")
    key_points = synthesizer.extract_key_points(documents)
    
    logger.info(f"\nKey Points:")
    logger.info(f"  Total documents: {key_points['documents']}")
    logger.info(f"  Total content length: {key_points['total_length']} characters")
    logger.info(f"  Unique concepts found: {len(key_points['unique_concepts'])}")
    if key_points['unique_concepts']:
        logger.info(f"    Sample concepts: {', '.join(list(key_points['unique_concepts'])[:5])}")
    
    # Synthesize findings
    synthesis = synthesizer.synthesize_findings(documents, query)
    logger.info(f"\nSynthesized Findings:\n{synthesis}")


async def demo_complete_workflow():
    """Demonstrate complete research workflow"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 5: Complete Research Workflow")
    logger.info("=" * 60)
    
    # Create research input
    workflow_input = WorkflowInput(
        query="applications of neural networks in medical diagnosis",
        topic="AI in Healthcare",
        keywords=["neural networks", "diagnosis", "classification", "accuracy"],
        num_papers=3,
        enable_rag=True,
        enable_reranking=True
    )
    
    logger.info(f"\nStarting research workflow:")
    logger.info(f"  Query: {workflow_input.query}")
    logger.info(f"  Topic: {workflow_input.topic}")
    logger.info(f"  Keywords: {', '.join(workflow_input.keywords)}")
    
    try:
        result = await orchestrator.execute_workflow(workflow_input)
        
        logger.info(f"\nWorkflow completed successfully!")
        logger.info(f"  Papers retrieved: {len(result.papers)}")
        logger.info(f"  Quality score: {result.quality_score:.2f}/10")
        logger.info(f"  Number of insights: {len(result.insights)}")
        
        if result.insights:
            logger.info(f"\nKey Insights:")
            for insight in result.insights[:3]:
                logger.info(f"  - {insight}")
        
        if result.summary:
            logger.info(f"\nReport Preview (first 300 chars):")
            logger.info(f"  {result.summary[:300]}...")
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")


async def demo_system_stats():
    """Display system statistics"""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 6: System Statistics")
    logger.info("=" * 60)
    
    stats = orchestrator.get_workflow_statistics()
    
    logger.info(f"\nWorkflow Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info(f"\nAgent Status:")
    for agent in [
        orchestrator.research_agent,
        orchestrator.summarization_agent,
        orchestrator.critic_agent,
        orchestrator.writer_agent
    ]:
        state = agent.get_state()
        logger.info(f"  {state['name']} ({state['role']})")
        logger.info(f"    State: {state['state']}")
        logger.info(f"    Capabilities: {', '.join(state['capabilities'])}")
        logger.info(f"    Messages: {state['message_count']}")


async def main():
    """Run all demos"""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " MULTI-AGENT RESEARCH ASSISTANT - DEMO ".center(58) + "║")
    logger.info("╚" + "=" * 58 + "╝")
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  API Base URL: {settings.api_base_url}")
    logger.info(f"  Chat Model: {settings.chat_model}")
    logger.info(f"  Embedding Model: {settings.embedding_model}")
    logger.info(f"  Rerank Model: {settings.rerank_model}")
    
    try:
        # Run demos
        await demo_document_chunking()
        await demo_hybrid_retrieval()
        await demo_contextual_compression()
        await demo_cross_document_synthesis()
        await demo_complete_workflow()
        await demo_system_stats()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL DEMOS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        logger.info("\nNext Steps:")
        logger.info("1. Start the FastAPI server:")
        logger.info("   python -m uvicorn src.main:app --reload")
        logger.info("\n2. Access the API at http://localhost:8000")
        logger.info("\n3. See docs/USAGE_EXAMPLES.md for API examples")
        
    except Exception as e:
        logger.error(f"\nDemo failed with error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
