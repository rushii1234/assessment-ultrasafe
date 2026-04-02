from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.logging_config import get_logger
from app.db.database import get_db
from app.db.models import User
from app.schemas import (
    ConversationCreate,
    ConversationResponse,
    ConversationWithMessages,
    ChatRequest,
    ChatResponse,
    MessageCreate,
    MessageResponse
)
from app.services.conversation_service import ConversationService
from app.rag.rag_service import RAGService

logger = get_logger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])
rag_service = RAGService()


@router.post("/", response_model=ConversationResponse)
async def create_conversation(
    conversation_create: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new conversation."""
    try:
        conversation = ConversationService.create_conversation(
            db, current_user.id, conversation_create
        )
        return conversation
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")


@router.get("/", response_model=List[ConversationResponse])
async def list_conversations(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all conversations for current user."""
    try:
        conversations = ConversationService.list_user_conversations(
            db, current_user.id, skip, limit
        )
        return conversations
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list conversations")


@router.get("/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a conversation with its message history."""
    try:
        conversation = ConversationService.get_conversation(db, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify ownership
        if conversation.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation")


@router.post("/{conversation_id}/chat", response_model=ChatResponse)
async def chat(
    conversation_id: int,
    chat_request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a chat message and get RAG-powered response."""
    try:
        # Verify conversation exists and user owns it
        conversation = ConversationService.get_conversation(db, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        if conversation.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Add user message to conversation
        user_message_create = MessageCreate(role="user", content=chat_request.message)
        user_message = ConversationService.add_message(db, conversation_id, user_message_create)
        
        # Process query with RAG
        rag_result = rag_service.process_query(chat_request.message, top_k=3)
        
        # Add assistant message to conversation
        assistant_message_create = MessageCreate(
            role="assistant",
            content=rag_result["response"]
        )
        assistant_message = ConversationService.add_message(
            db, conversation_id, assistant_message_create
        )
        
        # Prepare response
        return ChatResponse(
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=assistant_message,
            sources=[s["content"] for s in rag_result["sources"]],
            relevance_score=rag_result["average_relevance"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation."""
    try:
        conversation = ConversationService.get_conversation(db, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        if conversation.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        ConversationService.delete_conversation(db, conversation_id)
        return {"message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")
