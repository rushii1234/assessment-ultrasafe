from typing import List, Optional

from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.db.models import Document, DocumentEmbedding
from app.rag.embedding import VectorDatabaseService
from app.schemas import DocumentCreate, DocumentUpdate

logger = get_logger(__name__)


class DocumentService:
    """Service for document management."""
    
    def __init__(self):
        self.vector_db = VectorDatabaseService()
        self.vector_db.get_or_create_collection()
    
    def add_document(
        self,
        db: Session,
        document_create: DocumentCreate
    ) -> Document:
        """Add a new document and its embeddings."""
        try:
            # Create document record
            document = Document(
                title=document_create.title,
                content=document_create.content,
                source=document_create.source,
                metadata_json=document_create.metadata_json
            )
            
            db.add(document)
            db.flush()  # Get document ID without committing
            
            # Add to vector database
            doc_id = f"doc_{document.id}"
            self.vector_db.add_documents(
                ids=[doc_id],
                documents=[document_create.content],
                metadatas=[{
                    "title": document_create.title,
                    "source": document_create.source or "unknown",
                    "doc_id": document.id
                }]
            )
            
            # Create embedding record
            embedding = DocumentEmbedding(
                document_id=document.id,
                chunk_index=0,
                chunk_content=document_create.content,
                vector_id=doc_id
            )
            
            db.add(embedding)
            db.commit()
            db.refresh(document)
            
            logger.info(f"Document added: {document.id} - {document.title}")
            return document
        except Exception as e:
            db.rollback()
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def get_document(db: Session, document_id: int) -> Optional[Document]:
        """Get document by ID."""
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            return document
        except Exception as e:
            logger.error(f"Error fetching document: {str(e)}")
            raise
    
    def list_documents(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
        """List all active documents."""
        try:
            documents = db.query(Document).filter(
                Document.is_active == True
            ).offset(skip).limit(limit).all()
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise
    
    def update_document(
        self,
        db: Session,
        document_id: int,
        document_update: DocumentUpdate
    ) -> Optional[Document]:
        """Update a document."""
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return None
            
            # Update fields if provided
            if document_update.title is not None:
                document.title = document_update.title
            if document_update.content is not None:
                document.content = document_update.content
                # Update in vector database
                doc_id = f"doc_{document.id}"
                self.vector_db.delete_document(doc_id)
                self.vector_db.add_documents(
                    ids=[doc_id],
                    documents=[document_update.content],
                    metadatas=[{
                        "title": document.title,
                        "source": document.source or "unknown",
                        "doc_id": document.id
                    }]
                )
            if document_update.source is not None:
                document.source = document_update.source
            if document_update.metadata_json is not None:
                document.metadata_json = document_update.metadata_json
            if document_update.is_active is not None:
                document.is_active = document_update.is_active
            
            db.commit()
            db.refresh(document)
            
            logger.info(f"Document updated: {document.id}")
            return document
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating document: {str(e)}")
            raise
    
    def delete_document(self, db: Session, document_id: int) -> bool:
        """Delete a document (soft delete)."""
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return False
            
            # Soft delete
            document.is_active = False
            
            # Remove from vector database
            doc_id = f"doc_{document.id}"
            self.vector_db.delete_document(doc_id)
            
            db.commit()
            logger.info(f"Document deleted: {document.id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting document: {str(e)}")
            raise
