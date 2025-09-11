from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from typing import Dict, Any
from analyzer import DatasetAnalyzer

app = FastAPI(
    title="Dataset Analyzer",
    description="Analyze the dataset",
    version="1.0.0",
)

# cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message":"Dataset Analyzer API",
        "version":"1.0.0",
        "endpoints":{
            "analyze":"POST /analyzer - CSV dosyası yükle ve analiz et"
        }
    }

@app.post("/analyzer")
async def analyzer(file: UploadFile = File(...)):
    """
    CSV dosyası yükle ve analiz et
    """
    try:
        # Dosya validasyonu
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="sadece csv dosyaları destekleniyor"
            )

        # dosya içeriğini oku
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400,detail="Dosya boş")

        # pandas ile csv 'yi oku
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400,detail=f"CSV okuma hatası: {str(e)}")

        # analizi yap
        analyzer = DatasetAnalyzer(df)
        analysis_result = analyzer.analyze()

        # sonucu döndür
        return{
            "success": True,
            "filename": file.filename,
            "analysis": analysis_result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analiz Hatsaı: {str(e)}"
        )


@app.post("/visualizer")
async def visualizer(file: UploadFile = File(...)):


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)