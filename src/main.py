from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import uuid
from datetime import datetime

from src.config import settings
from src.workflow.orchestrator import orchestrator, WorkflowInput
from src.agents.base import ResearchQuery, ResearchResult
from src.rag.retrieval import DocumentChunker, HybridRetriever, ContextualCompressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="Advanced research system with multiple agents and RAG",
    version="1.0.0"
)

# Request/Response Models
class SearchRequest(BaseModel):
    """Request model for research search"""
    query: str = Field(..., description="Research query")
    topic: str = Field(..., description="Research topic")
    keywords: List[str] = Field(default_factory=list, description="Keywords for search")
    num_papers: int = Field(default=5, description="Number of papers to retrieve")
    enable_rag: bool = Field(default=True, description="Enable RAG for retrieval")
    enable_reranking: bool = Field(default=True, description="Enable reranking")


class PaperSummary(BaseModel):
    """Summary of a paper"""
    title: str
    summary: str
    key_points: List[str]
    quality_score: float


class EvaluationResult(BaseModel):
    """Quality evaluation result"""
    title: str
    evaluation: str
    quality_score: float
    recommendation: str


class ResearchResponse(BaseModel):
    """Response model for research results"""
    request_id: str
    query: str
    topic: str
    total_papers: int
    quality_score: float
    summary: str
    insights: List[str]
    completed_at: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    agents_ready: List[str]


class DocumentChunkRequest(BaseModel):
    """Request for document chunking"""
    text: str
    chunk_strategy: str = Field(default="sentence", description="Strategy: 'sentence' or 'size'")
    chunk_size: int = Field(default=1000)
    overlap: int = Field(default=200)


class ChunkResponse(BaseModel):
    """Response from document chunking"""
    original_length: int
    num_chunks: int
    chunks: List[str]


class EmbeddingRequest(BaseModel):
    """Request for embeddings"""
    text: str | List[str]


class RerankerRequest(BaseModel):
    """Request for reranking"""
    query: str
    documents: List[str]


# Track active workflows
active_workflows: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Multi-Agent Research Assistant")
    logger.info(f"API Base URL: {settings.api_base_url}")
    logger.info(f"Models: {settings.chat_model}, {settings.embedding_model}, {settings.rerank_model}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    agents_ready = [
        orchestrator.research_agent.name,
        orchestrator.summarization_agent.name,
        orchestrator.critic_agent.name,
        orchestrator.writer_agent.name
    ]
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agents_ready=agents_ready
    )


@app.post("/research", response_model=ResearchResponse)
async def start_research(request: SearchRequest, background_tasks: BackgroundTasks):
    """
    Start a research workflow
    
    Process:
    1. Retrieve papers based on query
    2. Summarize papers
    3. Evaluate quality
    4. Generate comprehensive report
    """
    try:
        request_id = str(uuid.uuid4())
        
        # Create workflow input
        workflow_input = WorkflowInput(
            query=request.query,
            topic=request.topic,
            keywords=request.keywords,
            num_papers=request.num_papers,
            enable_rag=request.enable_rag,
            enable_reranking=request.enable_reranking
        )
        
        # Execute workflow
        result = await orchestrator.execute_workflow(workflow_input)
        
        # Store in active workflows
        active_workflows[request_id] = {
            "query": request.query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        return ResearchResponse(
            request_id=request_id,
            query=request.query,
            topic=request.topic,
            total_papers=len(result.papers),
            quality_score=result.quality_score,
            summary=result.summary,
            insights=result.insights,
            completed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Research request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


@app.get("/research/{request_id}")
async def get_research_result(request_id: str):
    """Get results of a completed research request"""
    if request_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Request not found")
    
    workflow_data = active_workflows[request_id]
    result = workflow_data["result"]
    
    return {
        "request_id": request_id,
        "query": workflow_data["query"],
        "papers": len(result.papers),
        "quality_score": result.quality_score,
        "summary": result.summary[:500] + "...",  # Preview
        "completed_at": datetime.now().isoformat()
    }


@app.post("/chunk", response_model=ChunkResponse)
async def chunk_document(request: DocumentChunkRequest):
    """
    Chunk a document using specified strategy
    
    Strategies:
    - 'sentence': Chunks by sentences to maintain context
    - 'size': Chunks by character size with overlap
    """
    try:
        chunker = DocumentChunker(
            chunk_size=request.chunk_size,
            overlap=request.overlap
        )
        
        chunks = chunker.chunk_document(request.text, strategy=request.chunk_strategy)
        
        return ChunkResponse(
            original_length=len(request.text),
            num_chunks=len(chunks),
            chunks=chunks
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chunking failed: {str(e)}")


@app.post("/embed")
async def get_embeddings(request: EmbeddingRequest):
    """Get embeddings for text using the embedding model"""
    try:
        from src.api.client import api_client
        
        response = await api_client.get_embeddings(
            text_input=request.text
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.post("/rerank")
async def rerank_documents(request: RerankerRequest):
    """Rerank documents based on query relevance"""
    try:
        from src.api.client import api_client
        
        response = await api_client.rerank_documents(
            query=request.query,
            texts=request.documents
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """Get statistics about the system"""
    return {
        "workflow_stats": orchestrator.get_workflow_statistics(),
        "active_requests": len(active_workflows),
        "agents": {
            "research": orchestrator.research_agent.get_state(),
            "summarization": orchestrator.summarization_agent.get_state(),
            "critic": orchestrator.critic_agent.get_state(),
            "writer": orchestrator.writer_agent.get_state()
        }
    }


@app.get("/agents")
async def get_agents_info():
    """Get information about all agents"""
    return {
        "agents": [
            {
                "name": orchestrator.research_agent.name,
                "role": orchestrator.research_agent.role,
                "capabilities": orchestrator.research_agent.capabilities
            },
            {
                "name": orchestrator.summarization_agent.name,
                "role": orchestrator.summarization_agent.role,
                "capabilities": orchestrator.summarization_agent.capabilities
            },
            {
                "name": orchestrator.critic_agent.name,
                "role": orchestrator.critic_agent.role,
                "capabilities": orchestrator.critic_agent.capabilities
            },
            {
                "name": orchestrator.writer_agent.name,
                "role": orchestrator.writer_agent.role,
                "capabilities": orchestrator.writer_agent.capabilities
            }
        ],
        "workflow_orchestrator": {
            "type": "LangGraph-based",
            "stages": ["Initialize", "Search", "Summarize", "Evaluate", "Write", "Synthesize"]
        }
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Multi-Agent Research Assistant",
        "version": "1.0.0",
        "description": "Advanced research system with multiple agents and RAG",
        "endpoints": {
            "health": "/health",
            "start_research": "/research",
            "get_result": "/research/{request_id}",
            "chunk_document": "/chunk",
            "embeddings": "/embed",
            "rerank": "/rerank",
            "statistics": "/stats",
            "agents_info": "/agents"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, debug=settings.debug)
