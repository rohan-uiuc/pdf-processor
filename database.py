from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import logging

Base = declarative_base()

logger = logging.getLogger(__name__)

class Document(Base):
    __tablename__ = 'documents_v2'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    course_name = Column(String, nullable=True)
    readable_filename = Column(String, nullable=True)
    s3_path = Column(String, nullable=True)
    url = Column(String, nullable=True)
    base_url = Column(String, nullable=True)
    metadata_schema = Column(JSON, nullable=True)
    partition_status = Column(String, default='pending')
    table_status = Column(String, default='pending')
    chunk_status = Column(String, default='pending')
    db_save_status = Column(String, default='pending')
    last_error = Column(String, nullable=True)
    
    segments = relationship("Segment", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Segment(Base):
    __tablename__ = 'segments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents_v2.id', ondelete='CASCADE'), nullable=False)
    segment_number = Column(Integer, nullable=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    document = relationship("Document", back_populates="segments")
    chunks = relationship("Chunk", back_populates="segment", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = 'chunks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents_v2.id'), nullable=False)
    segment_id = Column(Integer, ForeignKey('segments.id', ondelete='CASCADE'), nullable=True)
    chunk_number = Column(Integer, nullable=False)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    document = relationship("Document", back_populates="chunks")
    segment = relationship("Segment", back_populates="chunks")

def init_db(db_path='sqlite:///pdf_processor.db'):
    logger.info(f"Initializing database at {db_path}")
    try:
        engine = create_engine(db_path)
        # Create tables only if they don't exist
        Base.metadata.create_all(engine)
        logger.info("Database initialized successfully")
        return engine
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

# Keep this function separate in case you need to reset the database manually
def reset_db(engine):
    """Drop all tables and recreate them."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

def get_session(engine):
    """Create a new session."""
    Session = sessionmaker(bind=engine)
    return Session()
