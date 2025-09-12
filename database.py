from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# SQLite database
DATABASE_URL = "sqlite:///./analyzer.db"
engine = create_engine(DATABASE_URL,connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UploadHistory(Base):
    """Upload geçmişi tablosu"""
    __tablename__ = 'upload_history'

    id = Column(Integer, primary_key=True,index=True)
    filename = Column(String(100), nullable=False)
    original_filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False) # csv ,xlsx, json
    file_size = Column(Integer, nullable=False)

    # Analysis
    rows_count = Column(Integer)
    columns_count = Column(Integer)
    analysis_summary = Column(Text) # Json string

    # Timestamps
    uploaded_at = Column(DateTime,default=datetime.now())
    analysis_duration = Column(Float)

def create_tables():
    """Tabloları Oluştur"""
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

