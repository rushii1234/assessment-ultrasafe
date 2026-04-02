from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, EmailStr, Field


# User Schemas
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=255)
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=72)


class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Authentication Schemas
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    sub: Optional[str] = None
    exp: Optional[float] = None


# Session Schemas
class SessionCreate(BaseModel):
    user_id: int
    token: str
    expires_at: datetime


class SessionResponse(BaseModel):
    id: int
    user_id: int
    is_active: bool
    created_at: datetime
    expires_at: datetime
    
    class Config:
        from_attributes = True


# Message Schemas
class MessageCreate(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., min_length=1)


class MessageResponse(MessageCreate):
    id: int
    conversation_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# Conversation Schemas
class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: int
    user_id: int
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ConversationWithMessages(ConversationResponse):
    messages: List[MessageResponse] = []


# Document Schemas
class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    source: Optional[str] = None
    metadata_json: Optional[str] = None


class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    metadata_json: Optional[str] = None
    is_active: Optional[bool] = None


class DocumentResponse(DocumentCreate):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Chat Schemas
class ChatRequest(BaseModel):
    conversation_id: Optional[int] = None
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    conversation_id: int
    user_message: MessageResponse
    assistant_message: MessageResponse
    sources: List[str] = Field(default_factory=list, description="Source documents used for RAG")
    relevance_score: float = Field(default=0.0, description="Average relevance score of retrieved documents")
    
    class Config:
        from_attributes = True


# RAG Schemas
class RetrievalResult(BaseModel):
    document_id: int
    title: str
    content: str
    relevance_score: float
    source: Optional[str] = None


class RankingResult(BaseModel):
    document_id: int
    content: str
    rank_score: float


# Health Check Schema
class HealthResponse(BaseModel):
    status: str = "healthy"
    message: str = "Service is running"
