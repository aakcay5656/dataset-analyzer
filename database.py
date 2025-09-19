from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./analyzer.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UploadHistory(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)

    # Analysis results
    rows_count = Column("rows", Integer, nullable=True)  # rows yerine rows_count
    columns_count = Column("columns", Integer, nullable=True)  # columns yerine columns_count
    analysis_summary = Column(Text, nullable=True)

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    analysis_duration = Column(Float, nullable=True)

    # 🆕 Chart data field - EKSİK OLAN BU!
    chart_data = Column(Text, nullable=True)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
