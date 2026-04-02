from typing import List, Optional

from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.db.models import Conversation, Message
from app.schemas import ConversationCreate, MessageCreate

logger = get_logger(__name__)


class ConversationService:
    """Service for conversation and message management."""
    
    @staticmethod
    def create_conversation(
        db: Session,
        user_id: int,
        conversation_create: ConversationCreate
    ) -> Conversation:
        """Create a new conversation."""
        try:
            conversation = Conversation(
                user_id=user_id,
                title=conversation_create.title or "New Conversation"
            )
            
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            
            logger.info(f"Conversation created: {conversation.id} for user: {user_id}")
            return conversation
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating conversation: {str(e)}")
            raise
    
    @staticmethod
    def get_conversation(db: Session, conversation_id: int) -> Optional[Conversation]:
        """Get conversation by ID."""
        try:
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            return conversation
        except Exception as e:
            logger.error(f"Error fetching conversation: {str(e)}")
            raise
    
    @staticmethod
    def list_user_conversations(
        db: Session,
        user_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> List[Conversation]:
        """List all conversations for a user."""
        try:
            conversations = db.query(Conversation).filter(
                Conversation.user_id == user_id
            ).order_by(
                Conversation.updated_at.desc()
            ).offset(skip).limit(limit).all()
            return conversations
        except Exception as e:
            logger.error(f"Error listing conversations: {str(e)}")
            raise
    
    @staticmethod
    def add_message(
        db: Session,
        conversation_id: int,
        message_create: MessageCreate
    ) -> Message:
        """Add a message to a conversation."""
        try:
            message = Message(
                conversation_id=conversation_id,
                role=message_create.role,
                content=message_create.content
            )
            
            db.add(message)
            db.commit()
            db.refresh(message)
            
            # Update conversation timestamp
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            if conversation:
                from datetime import datetime
                conversation.updated_at = datetime.utcnow()
                db.commit()
            
            logger.info(f"Message added to conversation: {conversation_id}")
            return message
        except Exception as e:
            db.rollback()
            logger.error(f"Error adding message: {str(e)}")
            raise
    
    @staticmethod
    def get_conversation_history(
        db: Session,
        conversation_id: int,
        limit: int = 50
    ) -> List[Message]:
        """Get message history for a conversation."""
        try:
            messages = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(
                Message.created_at.asc()
            ).limit(limit).all()
            return messages
        except Exception as e:
            logger.error(f"Error fetching conversation history: {str(e)}")
            raise
    
    @staticmethod
    def delete_conversation(db: Session, conversation_id: int) -> bool:
        """Delete a conversation and its messages."""
        try:
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if not conversation:
                return False
            
            # Delete all messages in conversation
            db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).delete()
            
            # Delete conversation
            db.delete(conversation)
            db.commit()
            
            logger.info(f"Conversation deleted: {conversation_id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting conversation: {str(e)}")
            raise
