from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.core.security import get_password_hash, verify_password, create_access_token
from app.core.logging_config import get_logger
from app.db.models import User, Session as DBSession
from app.schemas import UserCreate

logger = get_logger(__name__)


class UserService:
    """Service for user management."""
    
    @staticmethod
    def create_user(db: Session, user_create: UserCreate) -> User:
        """Create a new user."""
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(
                (User.username == user_create.username) | 
                (User.email == user_create.email)
            ).first()
            
            if existing_user:
                logger.warning(f"User creation failed: {user_create.username} already exists")
                raise ValueError("Username or email already exists")
            
            # Create new user
            user = User(
                username=user_create.username,
                email=user_create.email,
                hashed_password=get_password_hash(user_create.password)
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            logger.info(f"User created: {user.username}")
            return user
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating user: {str(e)}")
            raise
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            user = db.query(User).filter(User.username == username).first()
            return user
        except Exception as e:
            logger.error(f"Error fetching user: {str(e)}")
            raise
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID."""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            return user
        except Exception as e:
            logger.error(f"Error fetching user: {str(e)}")
            raise
    
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        try:
            user = UserService.get_user_by_username(db, username)
            if not user or not verify_password(password, user.hashed_password):
                logger.warning(f"Authentication failed for user: {username}")
                return None
            
            if not user.is_active:
                logger.warning(f"User inactive: {username}")
                return None
            
            logger.info(f"User authenticated: {username}")
            return user
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            raise
    
    @staticmethod
    def create_user_session(db: Session, user_id: int, token: str) -> DBSession:
        """Create a new user session."""
        try:
            expires_at = datetime.now(timezone.utc) + timedelta(days=7)
            
            session = DBSession(
                user_id=user_id,
                token=token,
                expires_at=expires_at
            )
            
            db.add(session)
            db.commit()
            db.refresh(session)
            
            logger.info(f"Session created for user: {user_id}")
            return session
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating session: {str(e)}")
            raise
    
    @staticmethod
    def get_session_by_token(db: Session, token: str) -> Optional[DBSession]:
        """Get session by token."""
        try:
            session = db.query(DBSession).filter(
                (DBSession.token == token) &
                (DBSession.is_active == True) &
                (DBSession.expires_at > datetime.now(timezone.utc))
            ).first()
            return session
        except Exception as e:
            logger.error(f"Error fetching session: {str(e)}")
            raise
    
    @staticmethod
    def invalidate_session(db: Session, token: str) -> bool:
        """Invalidate a session."""
        try:
            session = db.query(DBSession).filter(DBSession.token == token).first()
            if session:
                session.is_active = False
                db.commit()
                logger.info("Session invalidated")
                return True
            return False
        except Exception as e:
            db.rollback()
            logger.error(f"Error invalidating session: {str(e)}")
            raise
