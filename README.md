# Multi-Agent Research Assistant

## Overview

A sophisticated research system that combines multiple specialized agents with advanced RAG (Retrieval-Augmented Generation) to analyze academic papers and generate comprehensive research reports.

**Key Features:**
- 🤖 **Multi-Agent Architecture**: Research, Summarization, Critic, and Writer agents
- 🔄 **Advanced RAG System**: Hybrid retrieval, document chunking, contextual compression
- 🧠 **AI-Powered Analysis**: Uses GPT-like models for intelligent processing
- 📊 **Quality Evaluation**: Automated assessment of information quality
- 📝 **Professional Reports**: Generates well-structured research reports
- ⚡ **Fast API**: Async REST API for easy integration
- 📈 **Performance Monitoring**: Built-in metrics and statistics

## Architecture

```
┌─────────────────┐
│  FastAPI App    │
└────────┬────────┘
         │
    ┌────▼────┐
    │ 4 Agents │ (LangGraph Orchestrated)
    │Research  │
    │Summarize │
    │Critic    │
    │Writer    │
    └────┬────┘
         │
    ┌────▼────┐
    │ RAG      │
    │System   │
    └────┬────┘
         │
    ┌────▼────────┐
    │ Custom APIs │
    │ (api.us.inc)│
    └─────────────┘
```

## Quick Start

### 1. Prerequisites
- Python 3.8+
- pip or conda
- API key from api.us.inc

### 2. Installation

```bash
# Clone or download the project
cd Assessment-2

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API_KEY
```

### 3. Run the Server

```bash
python -m uvicorn src.main:app --reload --port 8000
```

Access the API at: `http://localhost:8000`

### 4. First API Call

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning in healthcare",
    "topic": "AI in Medicine",
    "keywords": ["diagnosis", "neural networks"],
    "num_papers": 5
  }'
```

## Project Structure

```
Assessment-2/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   ├── agents/
│   │   ├── base.py            # Agent classes
│   │   │   ├── ResearchAgent
│   │   │   ├── SummarizationAgent
│   │   │   ├── CriticAgent
│   │   │   └── WriterAgent
│   ├── api/
│   │   └── client.py          # Custom API client
│   ├── rag/
│   │   └── retrieval.py       # RAG system
│   │       ├── DocumentChunker
│   │       ├── HybridRetriever
│   │       ├── ContextualCompressor
│   │       └── CrossDocumentSynthesizer
│   └── workflow/
│       └── orchestrator.py    # LangGraph workflow
├── docs/
│   ├── ARCHITECTURE.md        # System design
│   ├── RAG_IMPLEMENTATION.md  # RAG details
│   └── USAGE_EXAMPLES.md      # API examples
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## Agents

### Research Agent
Searches for and retrieves relevant academic papers
- **Capabilities**: paper_retrieval, web_search, topic_analysis
- **Output**: List of papers with metadata

### Summarization Agent
Extracts key information from papers
- **Capabilities**: text_summarization, key_point_extraction, concept_identification
- **Output**: Structured summaries with key points

### Critic Agent
Evaluates quality and credibility of information
- **Capabilities**: quality_assessment, fact_checking, source_validation
- **Output**: Quality scores and recommendations

### Writer Agent
Compiles findings into comprehensive reports
- **Capabilities**: report_generation, content_creation, formatting
- **Output**: Well-structured markdown reports

## RAG System

The system implements advanced Retrieval-Augmented Generation:

1. **Document Chunking**
   - Sentence-based: Maintains semantic boundaries
   - Size-based: Configurable fixed-size chunks

2. **Hybrid Retrieval**
   - Keyword search (TF-IDF)
   - Semantic search (embeddings)
   - Combined hybrid scoring

3. **Contextual Compression**
   - Extracts relevant sentences
   - Manages context length
   - Preserves information density

4. **Cross-Document Synthesis**
   - Identifies common themes
   - Extracts key concepts
   - Synthesizes findings

## API Endpoints

### Research Operations
- `POST /research` - Start research workflow
- `GET /research/{request_id}` - Get results

### RAG Operations
- `POST /chunk` - Split document into chunks
- `POST /embed` - Get text embeddings
- `POST /rerank` - Rerank documents by relevance

### System
- `GET /health` - Health check
- `GET /stats` - Workflow statistics
- `GET /agents` - Agent information
- `GET /` - API information

## API Testing (Payloads & Params)

Base URL (local): `http://127.0.0.1:8000`

### `GET /`

- **Params**: None
- **Body**: None

```bash
curl http://127.0.0.1:8000/
```

### `GET /health`

- **Params**: None
- **Body**: None

```bash
curl http://127.0.0.1:8000/health
```

### `POST /research`

Example JSON:

```json
{
  "query": "machine learning in healthcare",
  "topic": "AI in Medicine",
  "keywords": ["diagnosis", "neural networks"],
  "num_papers": 5,
  "enable_rag": true,
  "enable_reranking": true
}
```

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"machine learning in healthcare\",\"topic\":\"AI in Medicine\",\"keywords\":[\"diagnosis\",\"neural networks\"],\"num_papers\":5,\"enable_rag\":true,\"enable_reranking\":true}"
```

### `GET /research/{request_id}`

Example:

```bash
curl http://127.0.0.1:8000/research/8b6d4a7f-0c3f-4b2d-9f1d-9f4f9d2a1c8a
```

### `POST /chunk`

Example JSON:

```json
{
  "text": "This is a long document...",
  "chunk_strategy": "sentence",
  "chunk_size": 1000,
  "overlap": 200
}
```

```bash
curl -X POST http://127.0.0.1:8000/chunk \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"This is a long document...\",\"chunk_strategy\":\"sentence\",\"chunk_size\":1000,\"overlap\":200}"
```

### `POST /embed`

Example JSON:

```json
{
  "text": ["hello world", "second text"]
}
```

```bash
curl -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d "{\"text\":[\"hello world\",\"second text\"]}"
```

### `POST /rerank`

Example JSON:

```json
{
  "query": "neural networks in diagnosis",
  "documents": [
    "doc 1 text...",
    "doc 2 text..."
  ]
}
```

```bash
curl -X POST http://127.0.0.1:8000/rerank \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"neural networks in diagnosis\",\"documents\":[\"doc 1 text...\",\"doc 2 text...\"]}"
```

### `GET /stats`

```bash
curl http://127.0.0.1:8000/stats
```

### `GET /agents`

```bash
curl http://127.0.0.1:8000/agents
```

## Configuration

### Environment Variables

```bash
# API Configuration
API_BASE_URL=https://api.us.inc/usf/v1
API_KEY=your_api_key_here

# Model Configuration
CHAT_MODEL=usf1-mini
EMBEDDING_MODEL=usf1-embed
RERANK_MODEL=usf1-rerank

# Parameters
TEMPERATURE=0.7
MAX_TOKENS=1000
# Backwards-compatible aliases (also supported)
CHAT_TEMPERATURE=0.7
CHAT_MAX_TOKENS=1000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
RERANK_TOP_K=3

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

## Example Usage

### Python
```python
import asyncio
from src.workflow.orchestrator import orchestrator, WorkflowInput

async def main():
    input_data = WorkflowInput(
        query="machine learning in healthcare",
        topic="AI in Medicine",
        keywords=["diagnosis", "neural networks"],
        num_papers=5
    )
    
    result = await orchestrator.execute_workflow(input_data)
    print(f"Quality Score: {result.quality_score}")
    print(f"Papers: {len(result.papers)}")
    print(result.summary)

asyncio.run(main())
```

### cURL
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quantum computing applications",
    "topic": "Quantum Technologies",
    "num_papers": 5
  }'
```

## Workflow Stages

1. **Initialize** - Parse input and create research query
2. **Search** - Retrieve relevant papers using web search
3. **Summarize** - Extract key information from papers
4. **Evaluate** - Assess quality and reliability
5. **Write** - Generate comprehensive report
6. **Synthesize** - Extract cross-document insights

## Performance Metrics

- **Retrieval**: Papers retrieved, relevance scores
- **Summarization**: Compression ratios, key points extracted
- **Quality**: Average quality score (0-10 scale)
- **System**: Request latency, throughput

## Error Handling

The system includes:
- Automatic retry logic
- Fallback mechanisms
- Graceful degradation
- Comprehensive error logging

## Advanced Features

### Hybrid Search
Combines keyword (TF-IDF) and semantic (embedding) search for better results.

### Smart Compression
Intelligently compresses documents while preserving key information.

### Quality Scoring
Automated evaluation of information quality from 0-10.

### Agent Communication
Structured message passing between agents for coordination.

## Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and components
- **[RAG Implementation](docs/RAG_IMPLEMENTATION.md)** - Detailed RAG explanation
- **[Usage Examples](docs/USAGE_EXAMPLES.md)** - API examples and code samples

## Troubleshooting

### "API Key not configured"
- Check `.env` file exists and has `API_KEY` set
- Verify API key is correct from api.us.inc

### "Connection timeout"
- Verify `API_BASE_URL` is correct
- Check network connectivity
- Ensure api.us.inc is accessible

### "Low quality scores"
- Increase `num_papers` for more source material
- Adjust `enable_reranking` settings
- Review search `keywords`

## Performance Optimization

1. **Caching**
   - Cache embeddings to avoid recomputation
   - Store frequently used documents

2. **Batching**
   - Process multiple documents in batch API calls
   - Reduces API overhead

3. **Configuration Tuning**
   - Adjust chunk sizes for your domain
   - Tune hybrid search alpha parameter
   - Configure appropriate TOP_K values

## Future Enhancements

- [ ] Multi-modal retrieval (images, tables)
- [ ] Knowledge graph integration
- [ ] Fine-tuned models for specific domains
- [ ] Distributed processing
- [ ] Analytics dashboard
- [ ] Document database integration
- [ ] Real-time collaboration

## API Models Used

- **usf1-mini**: Chat completion with web search
- **usf1-embed**: Text embeddings for semantic search
- **usf1-rerank**: Document reranking by relevance

All models accessed via: `https://api.us.inc/usf/v1`

## License

This is an assessment project.

## Support

For issues:
1. Check the documentation
2. Review error logs
3. Verify API configuration
4. Check internet connectivity

## Metrics

By default, the system tracks:
- Workflows executed
- Completion rate
- Papers retrieved
- Quality scores
- Agent performance
- Message passing

Access via `GET /stats` endpoint.

---

**Built with:**
- FastAPI
- LangChain & LangGraph
- Python async/await
- Custom RAG pipeline
- usf1 models from api.us.inc
