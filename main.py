from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import json
import time
from typing import List

from database import create_tables, get_db, UploadHistory
from file_manager import FileManager
from analyzer import MultiFormatAnalyzer

# Database'i initialize et
create_tables()

app = FastAPI(
    title="Dataset Analyzer v2",
    description="Multi-format dataset analysis with history!",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File manager
file_manager = FileManager()


@app.get("/")
async def root():
    return {
        "message": "Dataset Analyzer v2.0",
        "version": "2.0.0",
        "supported_formats": ["CSV", "Excel (.xlsx, .xls)", "JSON"],
        "endpoints": {
            "analyze": "POST /analyze - Dosya yükle ve analiz et",
            "history": "GET /history - Upload geçmişini görüntüle",
            "analysis/{id}": "GET /analysis/{id} - Belirli analizi getir"
        }
    }


@app.post("/analyze")
async def analyze_dataset(
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    """Multi-format dosya analizi"""
    start_time = time.time()

    try:
        # 1. Dosya format kontrolü
        allowed_extensions = ['.csv', '.xlsx', '.xls', '.json']
        file_extension = None

        for ext in allowed_extensions:
            if file.filename.lower().endswith(ext):
                file_extension = ext
                break

        if not file_extension:
            raise HTTPException(
                status_code=400,
                detail=f"Desteklenmeyen format. İzin verilen: {allowed_extensions}"
            )

        # 2. Dosyayı kaydet
        file_path, unique_filename = await file_manager.save_file(file)
        file_size = file_manager.get_file_size(file_path)

        # 3. Analizi yap
        analyzer = MultiFormatAnalyzer(file_path, file_extension)
        analyzer.load_data()
        analysis_result = analyzer.analyze()

        # 4. Database'e kaydet
        upload_record = UploadHistory(
            filename=unique_filename,
            original_filename=file.filename,
            file_type=file_extension,
            file_size=file_size,
            rows_count=analysis_result["basic_info"]["rows"],
            columns_count=analysis_result["basic_info"]["columns"],
            analysis_summary=json.dumps(analysis_result["insights"]),
            analysis_duration=analysis_result["analysis_duration"]
        )

        db.add(upload_record)
        db.commit()
        db.refresh(upload_record)

        # 5. Response
        total_duration = time.time() - start_time

        return {
            "success": True,
            "upload_id": upload_record.id,
            "filename": file.filename,
            "file_type": file_extension,
            "total_duration": round(total_duration, 3),
            "analysis": analysis_result
        }

    except HTTPException:
        raise
    except Exception as e:
        # Hata durumunda dosyayı temizle
        if 'file_path' in locals():
            file_manager.delete_file(file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Analiz hatası: {str(e)}"
        )


@app.get("/history")
async def get_upload_history(
        limit: int = 10,
        db: Session = Depends(get_db)
):
    """Upload geçmişi"""
    uploads = db.query(UploadHistory).order_by(
        UploadHistory.uploaded_at.desc()
    ).limit(limit).all()

    return {
        "total": len(uploads),
        "uploads": [
            {
                "id": upload.id,
                "original_filename": upload.original_filename,
                "file_type": upload.file_type,
                "file_size": upload.file_size,
                "rows": upload.rows_count,
                "columns": upload.columns_count,
                "uploaded_at": upload.uploaded_at,
                "analysis_duration": upload.analysis_duration
            }
            for upload in uploads
        ]
    }


@app.get("/analysis/{upload_id}")
async def get_analysis(
        upload_id: int,
        db: Session = Depends(get_db)
):
    """Belirli analiz detaylarını getir"""
    upload = db.query(UploadHistory).filter(
        UploadHistory.id == upload_id
    ).first()

    if not upload:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı")

    return {
        "id": upload.id,
        "filename": upload.original_filename,
        "file_type": upload.file_type,
        "rows": upload.rows_count,
        "columns": upload.columns_count,
        "uploaded_at": upload.uploaded_at,
        "insights": json.loads(upload.analysis_summary) if upload.analysis_summary else []
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
