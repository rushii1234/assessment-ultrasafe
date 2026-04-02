# FastAPI Chatbot with RAG - Backend

A production-ready FastAPI application implementing a customer support chatbot with Retrieval-Augmented Generation (RAG) capabilities.

## Features

### API Development
- **FastAPI Framework**: Modern, high-performance Python web framework
- **User Authentication**: JWT-based authentication with secure password hashing
- **Session Management**: Persistent session tracking in PostgreSQL
- **Conversation History**: Full conversation persistence with message tracking

### RAG Implementation (Hybrid Architecture)
- **Flexible Integration**: Support for both local models and third-party APIs
- **Vector Database Integration**: Chroma for efficient similarity search
- **Embedding Pipeline**: Local (Sentence Transformers) or USF API embeddings
- **Semantic Retrieval**: Find relevant context documents for queries
- **Hybrid Ranking**: Combine semantic similarity with BM25 or USF reranking
- **LLM Integration**: OpenAI API or USF API for context-aware response generation

### API Gateway Support
- **Third-Party API Integration**: Full support for USF API (chat, embeddings, reranking)
- **Single Authentication**: Unified API key for all operations
- **Easy Switching**: Toggle between local and API-based models via configuration

### Testing & Monitoring
- **Comprehensive Unit Tests**: Test coverage for all endpoints
- **RAG Performance Metrics**: Relevance scoring and source attribution
- **Structured Logging**: Detailed logs for debugging and monitoring
- **Error Handling**: Robust error handling with meaningful error messages

## Project Structure

```
├── app/
│   ├── api/                 # API endpoints
│   │   ├── auth.py         # Authentication endpoints
│   │   ├── chat.py         # Chat and conversation endpoints
│   │   ├── documents.py    # Document management endpoints
│   │   ├── dependencies.py # Dependency injection
│   │   └── health.py       # Health check endpoint
│   ├── core/               # Core application logic
│   │   ├── config.py       # Configuration management
│   │   ├── logging_config.py # Logging setup
│   │   └── security.py     # JWT and password utilities
│   ├── db/                 # Database layer
│   │   ├── database.py     # Database connection and session
│   │   └── models.py       # SQLAlchemy ORM models
│   ├── rag/                # RAG implementation
│   │   ├── embedding.py    # Embedding and vector DB services
│   │   ├── reranker.py     # Document reranking service
│   │   └── rag_service.py  # Main RAG orchestration
│   ├── schemas/            # Pydantic schemas
│   │   └── schemas.py      # Request/response schemas
│   ├── services/           # Business logic services
│   │   ├── user_service.py
│   │   ├── document_service.py
│   │   └── conversation_service.py
│   └── main.py            # FastAPI application entry point
├── tests/                  # Test suite
│   ├── conftest.py        # Pytest configuration and fixtures
│   ├── test_auth.py       # Authentication tests
│   ├── test_documents.py  # Document tests
│   ├── test_conversations.py # Conversation tests
│   └── test_rag.py        # RAG component tests
├── docs/                   # Documentation
├── logs/                   # Application logs (created at runtime)
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment configuration
└── README.md             # This file
```

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- pip

### Setup Steps

1. **Clone and navigate to project**
```bash
cd c:\Users\Admin\Desktop\Assessment
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
copy .env.example .env
```

Edit `.env` with your configuration:
```
DATABASE_URL=postgresql://user:password@localhost:5432/task1_db
OPENAI_API_KEY=your-api-key-here
SECRET_KEY=change-this-in-production
```

5. **Initialize database**
```bash
# Database tables will be created automatically on first run
```

## Running the Application

### Development Server
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000` with documentation at `/docs` (Swagger UI) or `/redoc` (ReDoc).

## API Documentation

### Authentication Endpoints

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "securepassword123"
}
```

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=john_doe&password=securepassword123
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Document Management Endpoints

#### Create Document
```http
POST /api/v1/documents/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "title": "Company Policy",
  "content": "All company policies are documented here...",
  "source": "internal_docs",
  "metadata": "{\"category\": \"policy\"}"
}
```

#### List Documents
```http
GET /api/v1/documents/?skip=0&limit=100
Authorization: Bearer {access_token}
```

#### Get Document
```http
GET /api/v1/documents/{document_id}
Authorization: Bearer {access_token}
```

#### Update Document
```http
PUT /api/v1/documents/{document_id}
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "title": "Updated Title",
  "content": "Updated content..."
}
```

#### Delete Document
```http
DELETE /api/v1/documents/{document_id}
Authorization: Bearer {access_token}
```

### Conversation & Chat Endpoints

#### Create Conversation
```http
POST /api/v1/conversations/
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "title": "Customer Support Chat"
}
```

#### List Conversations
```http
GET /api/v1/conversations/?skip=0&limit=50
Authorization: Bearer {access_token}
```

#### Get Conversation with History
```http
GET /api/v1/conversations/{conversation_id}
Authorization: Bearer {access_token}
```

#### Send Chat Message (RAG-powered)
```http
POST /api/v1/conversations/{conversation_id}/chat
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "message": "How do I reset my password?"
}
```

Response:
```json
{
  "conversation_id": 1,
  "user_message": {
    "id": 1,
    "role": "user",
    "content": "How do I reset my password?",
    "created_at": "2024-03-31T10:00:00"
  },
  "assistant_message": {
    "id": 2,
    "role": "assistant",
    "content": "To reset your password...",
    "created_at": "2024-03-31T10:00:01"
  },
  "sources": ["Document excerpt 1", "Document excerpt 2"],
  "relevance_score": 0.85
}
```

#### Delete Conversation
```http
DELETE /api/v1/conversations/{conversation_id}
Authorization: Bearer {access_token}
```

### Health Check
```http
GET /health
```

## 🧪 Quick API Testing Guide

### Testing Order
1. Register User
2. Login → Save Token
3. Create Documents (3x)
4. Create Conversation
5. Send Chat Messages
6. View Results

### 1. Register User
**Endpoint:** `POST /api/v1/auth/register`

**Payload:**
```json
{
  "username": "testuser",
  "email": "test@example.com",
  "password": "TestPass123"
}
```

---

### 2. Login (Get Token)
**Endpoint:** `POST /api/v1/auth/login?username=testuser&password=TestPass123`

**Response:**
```json
{
  "access_token": "YOUR_TOKEN_HERE",
  "token_type": "bearer",
  "expires_in": 1800
}
```

💾 **Save the token** - use it for all protected endpoints

---

### 3. Create Document - Security Policy
**Endpoint:** `POST /api/v1/documents/`

**Headers:**
```
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json
```

**Payload:**
```json
{
  "title": "Company Security Policy",
  "content": "All employees must use strong passwords. Change passwords every 90 days. Enable two-factor authentication on all accounts.",
  "source": "security_docs",
  "metadata_json": "{\"version\": \"1.0\"}"
}
```

---

### 4. Create Document - Password Reset Guide
**Endpoint:** `POST /api/v1/documents/`

**Payload:**
```json
{
  "title": "How to Reset Password",
  "content": "Click forgot password link on login page. Check your email for reset instructions. Click the link and create new password.",
  "source": "help",
  "metadata_json": "{\"category\": \"support\"}"
}
```

---

### 5. Create Document - API Guide
**Endpoint:** `POST /api/v1/documents/`

**Payload:**
```json
{
  "title": "API Authentication Guide",
  "content": "All requests need Bearer token. Put token in Authorization header. Format: Bearer YOUR_TOKEN. Tokens valid for 30 minutes.",
  "source": "api_docs",
  "metadata_json": "{\"category\": \"technical\"}"
}
```

---

### 6. List Documents
**Endpoint:** `GET /api/v1/documents/`

**Headers:**
```
Authorization: Bearer YOUR_TOKEN
```

---

### 7. Create Conversation
**Endpoint:** `POST /api/v1/conversations/`

**Headers:**
```
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json
```

**Payload:**
```json
{
  "title": "Support Chat - Account Help"
}
```

**Response:** Check the response for `conversation_id` (e.g., 1)

---

### 8. Send Chat Message - Password Reset
**Endpoint:** `POST /api/v1/conversations/1/chat`

**Headers:**
```
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json
```

**Payload:**
```json
{
  "message": "How do I reset my password?"
}
```

**Response:** RAG will search documents and return answer + sources

---

### 9. Send Chat Message - Security Policy
**Endpoint:** `POST /api/v1/conversations/1/chat`

**Payload:**
```json
{
  "message": "What is the security policy?"
}
```

---

### 10. Send Chat Message - 2FA
**Endpoint:** `POST /api/v1/conversations/1/chat`

**Payload:**
```json
{
  "message": "How do I enable two-factor authentication?"
}
```

---

### 11. Send Chat Message - API Auth
**Endpoint:** `POST /api/v1/conversations/1/chat`

**Payload:**
```json
{
  "message": "Tell me about API authentication"
}
```

---

### 12. Get Conversation History
**Endpoint:** `GET /api/v1/conversations/1`

**Headers:**
```
Authorization: Bearer YOUR_TOKEN
```

---

## Using Swagger UI (Easiest)

1. Open: http://127.0.0.1:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Paste the payload
5. Click "Execute"
6. See response instantly

---

## Using Curl Commands

### Register
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com","password":"TestPass123"}'
```

### Login
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/auth/login?username=testuser&password=TestPass123"
```

### Create Document
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents/" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"Security Policy","content":"All employees must use strong passwords","source":"docs","metadata_json":"{\"version\":\"1.0\"}"}'
```

### Create Conversation
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/conversations/" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"Support Chat"}'
```

### Send Chat
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/conversations/1/chat" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"How do I reset my password?"}'
```

### Health Check
```http
GET /health
```

---

## API Response Examples

### Chat Response
```json
{
  "conversation_id": 1,
  "user_message": {
    "id": 1,
    "role": "user",
    "content": "How do I reset my password?"
  },
  "assistant_message": {
    "id": 2,
    "role": "assistant",
    "content": "Based on the documentation: Click forgot password link on login page...",
    "created_at": "2026-04-02T10:00:00"
  },
  "sources": ["How to Reset Password"],
  "relevance_score": 0.92
}
```

### Document Response
```json
{
  "id": 1,
  "title": "Company Security Policy",
  "content": "All employees must use...",
  "source": "security_docs",
  "metadata_json": "{\"version\": \"1.0\"}",
  "is_active": true,
  "created_at": "2026-04-02T09:00:00"
}
```

### Conversation Response
```json
{
  "id": 1,
  "user_id": 1,
  "title": "Support Chat - Account Help",
  "messages": [
    {
      "id": 1,
      "conversation_id": 1,
      "role": "user",
      "content": "How do I reset my password?",
      "created_at": "2026-04-02T10:00:00"
    },
    {
      "id": 2,
      "conversation_id": 1,
      "role": "assistant",
      "content": "Based on the documentation...",
      "created_at": "2026-04-02T10:00:01"
    }
  ],
  "created_at": "2026-04-02T09:55:00",
  "updated_at": "2026-04-02T10:00:01"
}
```

---

## Key Implementation Details

### How RAG Works

1. **User sends message** → Saved to database
2. **Message is embedded** → Converted to vector using Sentence Transformers
3. **Vector search** → Chroma finds similar documents
4. **Documents retrieved** → Context is prepared from top matches
5. **Documents reranked** → BM25 scoring for better relevance
6. **Response generated** → Answer created from context
7. **Message saved** → Assistant response saved to database
8. **Response returned** → User sees answer with sources

### What Happens with Documents

1. **Document created** → Title + content stored
2. **Auto-embedded** → Content converted to vectors
3. **Stored in Chroma** → Vector DB indexed for fast search
4. **Available for RAG** → Used when answering questions
5. **Searchable** → Can be found by keyword or semantic similarity

### Authentication Flow

1. **Register** → Create user account (username, email, password)
2. **Password hashed** → Bcrypt with salt
3. **Login** → Verify credentials
4. **Token generated** → JWT token created
5. **Session saved** → Server tracks active sessions
6. **Token used** → Include in Authorization header
7. **Token verified** → Checked on each request
8. **Logout** → Session invalidated

---

## Error Codes

| Code | Meaning | Fix |
|------|---------|-----|
| 200 | Success | OK |
| 201 | Created | OK |
| 400 | Bad Request | Check payload format |
| 401 | Unauthorized | Add authentication token |
| 403 | Forbidden | No access to resource |
| 404 | Not Found | Resource doesn't exist |
| 500 | Server Error | Check logs |

---



### Users Table
- id: Primary key
- username: Unique username
- email: Unique email
- hashed_password: Bcrypt hashed password
- is_active: User status
- created_at, updated_at: Timestamps

### Sessions Table
- id: Primary key
- user_id: Foreign key to users
- token: JWT token
- is_active: Session status
- created_at, expires_at: Timestamps

### Conversations Table
- id: Primary key
- user_id: Foreign key to users
- title: Conversation title
- created_at, updated_at: Timestamps

### Messages Table
- id: Primary key
- conversation_id: Foreign key to conversations
- role: "user" or "assistant"
- content: Message text
- created_at: Timestamp

### Documents Table
- id: Primary key
- title: Document title
- content: Document content
- source: Document source
- metadata: JSON metadata
- is_active: Document status
- created_at, updated_at: Timestamps

### DocumentEmbeddings Table
- id: Primary key
- document_id: Foreign key to documents
- chunk_index: Index of chunk in document
- chunk_content: Chunk text
- vector_id: Reference to vector database

## RAG Architecture

### Retrieval Process

1. **Query Embedding**: User query is converted to embeddings using Sentence Transformers
2. **Semantic Search**: Embeddings are compared against document embeddings in Chroma
3. **Initial Ranking**: Top K documents are retrieved based on similarity scores
4. **Reranking**: BM25 algorithm reranks documents for better relevance
5. **Context Preparation**: Top documents are combined into context string

### Response Generation

1. **Context Assembly**: Retrieved documents are formatted as context
2. **Prompt Construction**: System prompt + context + user query
3. **LLM Generation**: OpenAI API generates response using context
4. **Response Enrichment**: Response includes source attribution and relevance scores

### Performance Metrics

- **Relevance Score**: Average similarity score of retrieved documents (0-1)
- **Source Attribution**: Original document references included in response
- **Retrieval Quality**: BM25 and semantic similarity combination

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_auth.py -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=app --cov-report=html
```

### Test Categories

- **Authentication Tests** (`test_auth.py`): User registration, login, authentication
- **Document Tests** (`test_documents.py`): Document CRUD operations
- **Conversation Tests** (`test_conversations.py`): Conversation and message management
- **RAG Tests** (`test_rag.py`): Embedding, vector database, and reranking

## Logging

Logs are stored in the `logs/` directory:
- `app.log`: Main application log with rotation (10MB per file, 5 backups kept)
- Console output: Real-time logging to stdout

### Log Levels
- DEBUG: Detailed diagnostic information
- INFO: General informational messages
- WARNING: Warning messages for potential issues
- ERROR: Error messages for failures

## Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app.main:app
```

### Using Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Environment Variables

See `.env.example` for all available configuration options:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/task1_db

# JWT
SECRET_KEY=your-super-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# LLM
OPENAI_API_KEY=your-openai-key
LLM_MODEL=gpt-3.5-turbo

# Vector Database
VECTOR_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Application
DEBUG=True
LOG_LEVEL=INFO
```

## Performance Considerations

1. **Database Indexing**: User lookups optimized with database indexes
2. **Connection Pooling**: SQLAlchemy manages connection pool
3. **Embedding Caching**: Vector database caches embeddings
4. **Batch Operations**: Support for batch document processing
5. **Async Operations**: FastAPI enables async request handling

## Security Features

1. **Password Security**: Bcrypt hashing with salt
2. **JWT Authentication**: Secure token-based authentication
3. **Session Management**: Server-side session tracking
4. **Input Validation**: Pydantic schema validation
5. **CORS Protection**: Configurable CORS settings
6. **Error Messages**: Non-revealing error messages in production

## Troubleshooting

### Database Connection Error
```
SQLALCHEMY_DATABASE_URL must point to a valid PostgreSQL database
```
Check `DATABASE_URL` in `.env` file

### Vector Database Error
```
chromadb PersistentClient initialization failed
```
Ensure `VECTOR_DB_PATH` directory is writable

### LLM API Error
```
OpenAI API key is not set
```
Set `OPENAI_API_KEY` in `.env` file

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

## Future Enhancements

1. Implement document chunking for large documents
2. Add support for multiple embedding models
3. Implement result caching for common queries
4. Add user-specific document access control
5. Implement feedback loop for RAG quality improvement
6. Add support for additional LLM providers
7. Implement conversation export/import
8. Add advanced search/filtering capabilities

## License

Proprietary - All rights reserved

## Support

For issues or questions, please contact the development team.
