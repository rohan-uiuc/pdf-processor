from sqlalchemy import ARRAY, create_engine, Column, Integer, String, DateTime, ForeignKey, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import logging
import os
import time

Base = declarative_base()

logger = logging.getLogger(__name__)

class Document(Base):
    __tablename__ = 'cedar_documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    course_name = Column(String, nullable=True)
    readable_filename = Column(String, nullable=True)
    s3_path = Column(String, nullable=True)
    url = Column(String, nullable=True)
    base_url = Column(String, nullable=True)
    metadata_schema = Column(JSON, nullable=True)  # Only for document schema and metadata extraction patterns
    processing_artifacts = Column(JSON, nullable=True)  # For storing image paths, element lists, and other processing data
    partition_status = Column(String, default='pending')  # Step 1
    chunk_status = Column(String, default='pending')     # Step 2
    table_status = Column(String, default='pending')     # Step 3
    schema_status = Column(String, default='pending')    # Step 4
    metadata_status = Column(String, default='pending')  # Step 5
    db_save_status = Column(String, default='pending')   # Final status
    last_error = Column(String, nullable=True)
    
    segments = relationship("Segment", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    schemas = relationship("DocumentSchema", back_populates="document", cascade="all, delete-orphan")
    metadata_entries = relationship("DocumentMetadata", back_populates="document", cascade="all, delete-orphan")

class Segment(Base):
    __tablename__ = 'cedar_segments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('cedar_documents.id', ondelete='CASCADE'), nullable=False)
    segment_number = Column(Integer, nullable=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    document = relationship("Document", back_populates="segments")
    chunks = relationship("Chunk", back_populates="segment", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = 'cedar_chunks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('cedar_documents.id'), nullable=False)
    segment_id = Column(Integer, ForeignKey('cedar_segments.id', ondelete='CASCADE'), nullable=True)
    chunk_number = Column(Integer, nullable=False)
    chunk_type = Column(String, nullable=False, default='text')  # text, table, image, etc.
    content = Column(Text, nullable=False)
    table_html = Column(Text, nullable=True)  # For table chunks
    table_image_paths = Column(ARRAY(Text), nullable=True)  # List of image paths for table chunks
    table_data = Column(JSON, nullable=True)  # Structured table data
    chunk_metadata = Column(JSON, nullable=True)    # Chunk-specific metadata
    orig_elements = Column(Text, nullable=True)     # Base64 gzipped original elements
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    document = relationship("Document", back_populates="chunks")
    segment = relationship("Segment", back_populates="chunks")

class DocumentSchema(Base):
    __tablename__ = 'cedar_document_schemas'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('cedar_documents.id', ondelete='CASCADE'), nullable=False)
    schema_type = Column(String, nullable=False)  # e.g., 'course', 'syllabus', etc.
    schema_version = Column(String, nullable=False)
    schema_definition = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    document = relationship("Document", back_populates="schemas")

class DocumentMetadata(Base):
    __tablename__ = 'cedar_document_metadata'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('cedar_documents.id', ondelete='CASCADE'), nullable=False)
    segment_id = Column(Integer, ForeignKey('cedar_segments.id', ondelete='CASCADE'), nullable=True)
    chunk_id = Column(Integer, ForeignKey('cedar_chunks.id', ondelete='CASCADE'), nullable=True)
    field_name = Column(String, nullable=False)
    field_value = Column(JSON, nullable=True)
    confidence_score = Column(Integer, nullable=True)  # 0-100
    extraction_method = Column(String, nullable=True)  # e.g., 'gpt-4', 'rule-based', etc.
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    document = relationship("Document", back_populates="metadata_entries")
    segment = relationship("Segment")
    chunk = relationship("Chunk")

def init_db(db_path=None, force_recreate=False, max_retries=5, retry_delay=2):
    """Initialize database with environment variable support and connection retries."""
    if db_path is None:
        # Default to local Supabase PostgreSQL connection
        db_path = os.getenv('DATABASE_URL', 
                           'postgresql://postgres:postgres@db.supabase_network_ai-ta-backend:5432/postgres')
    
    logger.info(f"Initializing database at {db_path}")
    
    for attempt in range(max_retries):
        try:
            engine = create_engine(db_path)
            # Test connection using proper SQLAlchemy syntax
            with engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
                conn.commit()
            
            if force_recreate:
                logger.info("Forcing database recreation...")
                Base.metadata.drop_all(engine)
            
            # Create tables
            Base.metadata.create_all(engine)
            logger.info("Database initialized successfully")
            return engine
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.warning(f"Database connection attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Error initializing database after {max_retries} attempts: {str(e)}")
                raise

def reset_db(engine):
    """Drop all tables and recreate them."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

def get_session(engine):
    """Create a new session."""
    Session = sessionmaker(bind=engine)
    return Session()

def delete_document(session, filename: str) -> tuple[bool, str]:
    """Delete a document and all its related data.
    
    Args:
        session: SQLAlchemy session
        filename: Name of the file to delete
        
    Returns:
        tuple[bool, str]: (success, message)
    """
    try:
        # Find the document
        doc = session.query(Document).filter_by(readable_filename=filename).first()
        if not doc:
            return False, f"Document {filename} not found"
            
        # Due to cascade='all, delete-orphan', this will automatically delete:
        # - All segments
        # - All chunks
        # - All schemas
        # - All metadata entries
        session.delete(doc)
        session.commit()
        
        logger.info(f"Successfully deleted document: {filename}")
        return True, f"Successfully deleted document: {filename}"
        
    except Exception as e:
        session.rollback()
        error_msg = f"Error deleting document {filename}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
