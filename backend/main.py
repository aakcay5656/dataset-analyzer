from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import json
import time
from typing import List, Optional

from database import create_tables, get_db, UploadHistory
from file_manager import FileManager
from analyzer import MultiFormatAnalyzer
from charts import ChartGenerator  # 🆕 Chart import
from streaming_processor import StreamingProcessor
import psutil
import os
# main.py - import bölümüne ekle
import numpy as np
from decimal import Decimal

# JSON serialization helper function ekle
def convert_numpy_types(obj):
    """Numpy tiplerini JSON serializable tiplere dönüştür"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, Decimal):
        return float(obj)
    elif hasattr(obj, 'item'):  # pandas scalars
        return obj.item()
    else:
        return obj
try:
    from enhanced_charts import AdvancedChartGenerator
    ADVANCED_CHARTS_AVAILABLE = True
    print("✅ Advanced Charts modülü yüklendi")
except ImportError as e:
    print(f"❌ Advanced Charts modülü yüklenemedi: {e}")
    ADVANCED_CHARTS_AVAILABLE = False
    # Fallback import
    from charts import ChartGenerator

# Database'i initialize et
create_tables()

app = FastAPI(
    title="Dataset Analyzer v3",
    description="Multi-format dataset analysis with interactive charts!",
    version="3.1.0"
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
file_manager = FileManager('uploads')


@app.get("/")
async def root():
    return {
        "message": "Dataset Analyzer v3",
        "version": "3.1.0",
        "supported_formats": ["CSV", "Excel (.xlsx, .xls)", "JSON"],
        "features": ["Data Analysis", "Interactive Charts", "Outlier Detection", "Correlation Analysis"],
        "endpoints": {
            "analyze": "POST /analyze - Dosya yükle ve analiz et (charts dahil)",
            "charts/{id}": "GET /charts/{id} - Sadece chart'ları getir",
            "history": "GET /history - Upload geçmişini görüntüle",
            "analysis/{id}": "GET /analysis/{id} - Belirli analizi getir"
        }
    }


@app.post("/analyze")
async def analyze_dataset(
        file: UploadFile = File(...),
        include_charts: bool = Query(True),
        detailed_charts: bool = Query(False),
        db: Session = Depends(get_db)
):
    """Dataset analizi - düzeltilmiş chart'larla"""
    start_time = time.time()
    file_path = None

    try:
        print(f"🚀 Analiz başladı: {file.filename}")
        print(f"📊 Charts: {include_charts}, Detailed: {detailed_charts}")
        print(f"🔧 Advanced Charts Available: {ADVANCED_CHARTS_AVAILABLE}")

        # Dosya format kontrolü
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

        # Dosyayı kaydet
        file_path, unique_filename = await file_manager.save_file(file)
        file_size = file_manager.get_file_size(file_path)

        print(f"📁 Dosya kaydedildi: {file_path} ({file_size} bytes)")

        # Analizi yap
        analyzer = MultiFormatAnalyzer(file_path, file_extension)
        analyzer.load_data()
        analysis_result = analyzer.analyze()

        print(f"✅ Temel analiz tamamlandı")
        print(f"📊 DataFrame shape: {analyzer.df.shape if analyzer.df is not None else 'None'}")

        # Chart'ları oluştur
        charts = {}
        chart_count = 0
        chart_mode = "none"

        if include_charts and analyzer.df is not None:
            print(f"🎨 Chart oluşturma başlıyor - Detailed: {detailed_charts}")

            try:
                # Force detailed charts for testing - TEMPORARILY
                if detailed_charts:
                    print(f"🔍 Trying advanced charts - Available: {ADVANCED_CHARTS_AVAILABLE}")

                    if ADVANCED_CHARTS_AVAILABLE:
                        print("🎨 AdvancedChartGenerator çalıştırılıyor...")

                        # Import and create advanced charts
                        from enhanced_charts import AdvancedChartGenerator
                        chart_generator = AdvancedChartGenerator(analyzer.df)

                        print("📊 generate_all_charts() çağırılıyor...")
                        charts = chart_generator.generate_all_charts()

                        print(f"🔍 Generated chart keys: {list(charts.keys()) if charts else 'Empty'}")

                        # Check if charts were actually created
                        if charts and isinstance(charts, dict):
                            # Check format
                            first_key = list(charts.keys())[0] if charts else None
                            if first_key and isinstance(charts[first_key], dict):
                                if 'type' in charts[first_key] and 'charts' in charts[first_key]:
                                    print("✅ Advanced format detected!")
                                    chart_mode = "detailed"

                                    # Count individual charts
                                    chart_count = sum(
                                        len(v.get('charts', {}))
                                        for v in charts.values()
                                        if isinstance(v, dict) and 'charts' in v
                                    )

                                    print(f"📊 Advanced charts: {len(charts)} groups, {chart_count} individual charts")
                                else:
                                    print("⚠️ Charts created but not in advanced format")
                                    chart_mode = "basic"
                                    chart_count = len(charts)
                            else:
                                print("⚠️ Charts created but invalid structure")
                        else:
                            print("❌ No charts generated from AdvancedChartGenerator")
                            charts = {}

                        # 🔧 NUMPY TYPES CONVERT ET
                        if charts:
                            print("🔄 Converting numpy types to JSON...")
                            charts = convert_numpy_types(charts)
                            print("✅ Numpy conversion completed")

                    else:
                        print("❌ Advanced charts not available, falling back to basic")
                        # Fall back to basic charts
                        chart_generator = ChartGenerator(analyzer.df)
                        charts = chart_generator.generate_all_charts()
                        charts = convert_numpy_types(charts)
                        chart_count = len(charts)
                        chart_mode = "basic"

                else:
                    # Basic charts explicitly requested
                    print("📊 Basic charts requested")
                    chart_generator = ChartGenerator(analyzer.df)
                    charts = chart_generator.generate_all_charts()
                    charts = convert_numpy_types(charts)
                    chart_count = len(charts)
                    chart_mode = "basic"

                print(f"🎯 Final chart mode: {chart_mode}, count: {chart_count}")

            except Exception as chart_error:
                print(f"❌ Chart oluşturma hatası: {chart_error}")
                import traceback
                print(f"📋 Chart error stack trace: {traceback.format_exc()}")

                # Try basic charts as fallback
                try:
                    print("🔄 Fallback: Basic charts deneniyor...")
                    chart_generator = ChartGenerator(analyzer.df)
                    charts = chart_generator.generate_all_charts()
                    charts = convert_numpy_types(charts)
                    chart_count = len(charts)
                    chart_mode = "basic_fallback"
                    print(f"✅ Fallback successful: {chart_count} basic charts")
                except Exception as fallback_error:
                    print(f"❌ Fallback de başarısız: {fallback_error}")
                    charts = {}
                    chart_count = 0
                    chart_mode = "failed"

        else:
            print("⚠️ Charts skipped - include_charts=False or DataFrame is None")

        # 🔧 DATABASE SAVE - PROPER INDENTATION
        print("💾 Veritabanına kaydediliyor...")

        try:
            # Analysis results'ı da convert et
            safe_analysis_result = convert_numpy_types(analysis_result)

            upload_record = UploadHistory(
                filename=unique_filename,
                original_filename=file.filename,
                file_type=file_extension,
                file_size=file_size,
                rows_count=int(safe_analysis_result["basic_info"]["rows"]),
                columns_count=int(safe_analysis_result["basic_info"]["columns"]),
                analysis_summary=json.dumps(safe_analysis_result["insights"]),
                analysis_duration=float(safe_analysis_result["analysis_duration"]),
                chart_data=json.dumps(charts) if charts else None
            )

            db.add(upload_record)
            db.commit()
            db.refresh(upload_record)

            print(f"✅ Veritabanına kaydedildi: ID {upload_record.id}")

        except Exception as db_error:
            print(f"❌ Database kayıt hatası: {db_error}")
            import traceback
            print(f"📋 DB error stack trace: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Database kayıt hatası: {str(db_error)}")

        # Dosyayı sil
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ Dosya silindi: {file_path}")

        # Response
        total_duration = time.time() - start_time
        print(f"🎉 Analiz tamamlandı: {total_duration:.2f}s")

        response_data = {
            "success": True,
            "upload_id": upload_record.id,
            "filename": file.filename,
            "file_type": file_extension,
            "total_duration": round(total_duration, 3),
            "analysis": safe_analysis_result,
            "charts": charts,
            "chart_mode": chart_mode,
            "chart_count": chart_count,
            "advanced_charts_available": ADVANCED_CHARTS_AVAILABLE
        }

        return convert_numpy_types(response_data)

    except HTTPException:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Analiz hatası: {str(e)}")
        print(f"📋 Stack trace: {error_trace}")

        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "file_processed": file.filename if file else "unknown"
            }
        )


@app.post("/streaming-analyze")
async def streaming_analyze_dataset(
        file: UploadFile = File(...),
        chunk_size: int = Query(100000),
        include_charts: bool = Query(False),  # 🆕 Chart desteği - default False (performance)
        detailed_charts: bool = Query(False),  # 🆕 Detailed chart desteği
        db: Session = Depends(get_db)
):
    """Streaming analizi - chart desteği ile"""
    file_path = None
    start_time = time.time()

    try:
        print(f"🚀 Streaming analiz başladı: {file.filename}")
        print(f"📊 Charts: {include_charts}, Detailed: {detailed_charts}")
        print(f"🔧 Chunk size: {chunk_size:,}")

        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Streaming analiz sadece CSV için")

        # Dosyayı kaydet
        file_path, unique_filename = await file_manager.save_file(file)
        file_size = file_manager.get_file_size(file_path)

        print(f"📁 Streaming için dosya kaydedildi: {file_path}")

        # Streaming analiz
        processor = StreamingProcessor(file_path, chunk_size)
        analysis_result = processor.stream_analysis()

        print(f"✅ Streaming analiz tamamlandı")
        print(f"📊 Total rows processed: {analysis_result.get('total_rows', 0):,}")

        # 🆕 Chart'ları oluştur (eğer isteniyorsa)
        charts = {}
        chart_count = 0
        chart_mode = "none"

        if include_charts:
            try:
                print("🎨 Streaming sonucu için chart'lar oluşturuluyor...")

                # Streaming'den sonra küçük bir sample DataFrame oluştur
                # Çünkü streaming tüm veriyi memory'de tutmaz
                import pandas as pd

                # StreamingProcessor'dan sample data al (eğer mümkünse)
                sample_df = None

                try:
                    # Dosyayı tekrar oku ama sadece sample için (ilk 10K satır)
                    sample_size = min(10000, analysis_result.get('total_rows', 10000))
                    print(f"📊 Chart için {sample_size:,} satır sample okunuyor...")

                    sample_df = pd.read_csv(file_path, nrows=sample_size)
                    print(f"✅ Sample DataFrame: {sample_df.shape}")

                except Exception as sample_error:
                    print(f"❌ Sample DataFrame oluşturulamadı: {sample_error}")
                    sample_df = None

                # Chart'ları oluştur
                if sample_df is not None and not sample_df.empty:
                    if detailed_charts and ADVANCED_CHARTS_AVAILABLE:
                        print("🎨 Advanced charts (sample data ile)...")

                        from enhanced_charts import AdvancedChartGenerator
                        chart_generator = AdvancedChartGenerator(sample_df)
                        charts = chart_generator.generate_all_charts()

                        # Chart format kontrolü
                        if charts and isinstance(charts, dict):
                            first_key = list(charts.keys())[0] if charts else None
                            if first_key and isinstance(charts[first_key], dict):
                                if 'type' in charts[first_key] and 'charts' in charts[first_key]:
                                    chart_mode = "detailed_sample"
                                    # Count individual charts
                                    chart_count = sum(
                                        len(v.get('charts', {}))
                                        for v in charts.values()
                                        if isinstance(v, dict) and 'charts' in v
                                    )
                                    print(f"✅ Advanced charts: {len(charts)} groups, {chart_count} individual")
                                else:
                                    chart_mode = "basic_sample"
                                    chart_count = len(charts)

                        # Numpy types convert
                        charts = convert_numpy_types(charts)

                    else:
                        print("📊 Basic charts (sample data ile)...")

                        from charts import ChartGenerator
                        chart_generator = ChartGenerator(sample_df)
                        charts = chart_generator.generate_all_charts()
                        charts = convert_numpy_types(charts)
                        chart_count = len(charts)
                        chart_mode = "basic_sample"

                        print(f"✅ Basic charts: {chart_count} charts")

                    # Chart metadata ekle
                    if charts:
                        # Add sample disclaimer to all chart insights
                        sample_disclaimer = f"📊 Charts based on sample of {len(sample_df):,} rows from {analysis_result.get('total_rows', 0):,} total rows"

                        for chart_group_name, chart_group in charts.items():
                            if isinstance(chart_group, dict) and 'insights' in chart_group:
                                if isinstance(chart_group['insights'], list):
                                    chart_group['insights'].insert(0, sample_disclaimer)

                else:
                    print("❌ Sample DataFrame boş - charts oluşturulamadı")

            except Exception as chart_error:
                print(f"❌ Streaming chart hatası: {chart_error}")
                import traceback
                print(f"📋 Chart error: {traceback.format_exc()}")
                charts = {}
                chart_count = 0
                chart_mode = "failed"

        else:
            print("⚠️ Charts atlandı - include_charts=False")

        # Database'e kaydet - UPDATED
        print("💾 Streaming sonuçları veritabanına kaydediliyor...")

        try:
            # Convert analysis result to safe types
            safe_analysis_result = convert_numpy_types(analysis_result)

            upload_record = UploadHistory(
                filename=unique_filename,
                original_filename=file.filename,
                file_type='.csv',
                file_size=file_size,
                rows_count=safe_analysis_result.get('total_rows', 0),
                columns_count=len(safe_analysis_result.get('column_types', {})),
                analysis_summary=json.dumps({
                    "method": "streaming",
                    "performance": safe_analysis_result.get('performance', {}),
                    "chunk_size": chunk_size,
                    "chart_mode": chart_mode,
                    "chart_count": chart_count
                }),
                analysis_duration=safe_analysis_result.get('performance', {}).get('analysis_duration', 0),
                chart_data=json.dumps(charts) if charts else None  # 🆕 Chart data kaydet
            )

            db.add(upload_record)
            db.commit()
            db.refresh(upload_record)

            print(f"✅ Streaming sonuçları kaydedildi: ID {upload_record.id}")

        except Exception as db_error:
            print(f"❌ Database kayıt hatası: {db_error}")
            import traceback
            print(f"📋 DB error: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Database kayıt hatası: {str(db_error)}")

        # Dosyayı sil
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)  # Direct remove yerine
                print(f"🗑️ Streaming dosyası silindi: {file_path}")
            except Exception as delete_error:
                print(f"⚠️ Dosya silme hatası: {delete_error}")

        # Response - ENHANCED
        total_duration = time.time() - start_time

        response_data = {
            "success": True,
            "method": "streaming",
            "upload_id": upload_record.id,
            "filename": file.filename,
            "file_type": ".csv",
            "total_duration": round(total_duration, 3),
            "analysis": safe_analysis_result,
            "charts": charts,  # 🆕 Charts dahil
            "chart_mode": chart_mode,  # 🆕 Chart mode
            "chart_count": chart_count,  # 🆕 Chart count
            "chunk_size": chunk_size,
            "sample_size": len(
                sample_df) if include_charts and 'sample_df' in locals() and sample_df is not None else None,
            "file_deleted": True,
            "advanced_charts_available": ADVANCED_CHARTS_AVAILABLE
        }

        print(f"🎉 Streaming analiz tamamlandı: {total_duration:.2f}s")
        return convert_numpy_types(response_data)

    except HTTPException:
        # HTTP exceptions'ı tekrar raise et
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        raise

    except Exception as e:
        print(f"❌ Streaming analiz hatası: {str(e)}")
        import traceback
        print(f"📋 Stack trace: {traceback.format_exc()}")

        # Hata durumunda da dosyayı sil
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🗑️ Hata sonrası dosya silindi: {file_path}")
            except:
                pass

        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "method": "streaming"
            }
        )


@app.get("/charts/{upload_id}")
async def get_charts(
        upload_id: int,
        chart_type: str = "all",
        db: Session = Depends(get_db)
):
    """Chart'ları getir - geliştirilmiş"""
    try:
        upload = db.query(UploadHistory).filter(UploadHistory.id == upload_id).first()

        if not upload:
            raise HTTPException(status_code=404, detail="Upload not found")

        if not upload.chart_data:
            return {
                "success": False,
                "message": "No charts available for this upload",
                "charts": None
            }

        # Parse chart data
        try:
            charts = json.loads(upload.chart_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid chart data format")

        # Chart info
        chart_info = {
            "upload_id": upload_id,
            "original_filename": upload.original_filename,
            "file_type": upload.file_type,
            "chart_count": len(charts) if isinstance(charts, dict) else 0,
            "chart_format": "unknown"
        }

        # Detect chart format
        if isinstance(charts, dict):
            # Advanced format detection
            is_advanced = any(
                isinstance(v, dict) and 'charts' in v and 'type' in v
                for v in charts.values()
            )

            chart_info["chart_format"] = "advanced" if is_advanced else "basic"

            if is_advanced:
                # Count individual charts in advanced format
                individual_count = sum(
                    len(v.get('charts', {}))
                    for v in charts.values()
                    if isinstance(v, dict) and 'charts' in v
                )
                chart_info["individual_chart_count"] = individual_count

        return {
            "success": True,
            "charts": charts,
            "info": chart_info
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving charts: {str(e)}")


@app.get("/charts/{upload_id}/detailed")
async def get_detailed_charts(
        upload_id: int,
        column: Optional[str] = None,
        chart_type: Optional[str] = None,
        db: Session = Depends(get_db)
):
    """Detaylı chart'ları getir"""
    upload = db.query(UploadHistory).filter(UploadHistory.id == upload_id).first()

    if not upload or not upload.chart_data:
        raise HTTPException(status_code=404, detail="Charts not found")

    charts = json.loads(upload.chart_data)

    # Filter by column
    if column:
        filtered_charts = {k: v for k, v in charts.items() if f'column_{column}' in k}
        return {"column": column, "charts": filtered_charts}

    # Filter by chart type
    if chart_type:
        filtered_charts = {}
        for chart_group_name, chart_group in charts.items():
            if isinstance(chart_group, dict) and 'charts' in chart_group:
                if chart_type in chart_group['charts']:
                    filtered_charts[chart_group_name] = {
                        'type': chart_group.get('type'),
                        'chart': chart_group['charts'][chart_type]
                    }
        return {"chart_type": chart_type, "charts": filtered_charts}

    return {
        "upload_id": upload_id,
        "all_charts": charts,
        "summary": {
            "chart_groups": len(charts),
            "available_columns": [k.replace('column_', '') for k in charts.keys() if k.startswith('column_')],
            "overview_charts": [k for k in charts.keys() if not k.startswith('column_')]
        }
    }


@app.get("/charts/types")
async def get_available_chart_types():
    """Mevcut chart tiplerini listele"""
    return {
        "chart_categories": {
            "dataset_overview": ["column_types", "dataset_shape"],
            "numeric_analysis": ["histogram", "boxplot", "statistics", "density"],
            "categorical_analysis": ["bar_chart", "pie_chart", "frequency_curve"],
            "datetime_analysis": ["time_series", "year_distribution", "month_distribution"],
            "text_analysis": ["length_distribution", "word_count", "character_frequency"],
            "quality_analysis": ["missing_values", "correlation_analysis", "data_quality_dashboard"]
        },
        "column_specific": True,
        "inter_column_analysis": True
    }

@app.get("/history")
async def get_upload_history(limit: int = 10, db: Session = Depends(get_db)):
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
                "rows": upload.rows_count,  # ✅ rows_count kullan
                "columns": upload.columns_count,  # ✅ columns_count kullan
                "uploaded_at": upload.uploaded_at,
                "analysis_duration": upload.analysis_duration,
                "has_charts": upload.chart_data is not None  # ✅ chart_data kontrol
            }
            for upload in uploads
        ]
    }


@app.get("/analysis/{upload_id}")
async def get_analysis(
        upload_id: int,
        db: Session = Depends(get_db)
):
    """Analiz detaylarını getir - geliştirilmiş"""
    try:
        upload = db.query(UploadHistory).filter(UploadHistory.id == upload_id).first()

        if not upload:
            raise HTTPException(status_code=404, detail="Upload not found")

        # Parse analysis summary
        analysis_summary = {}
        if upload.analysis_summary:
            try:
                analysis_summary = json.loads(upload.analysis_summary)
            except json.JSONDecodeError:
                analysis_summary = {"error": "Invalid analysis summary format"}

        # Response with all available data
        response = {
            "upload_id": upload_id,
            "filename": upload.filename,
            "original_filename": upload.original_filename,
            "file_type": upload.file_type,
            "file_size": upload.file_size,
            "rows_count": upload.rows_count,
            "columns_count": upload.columns_count,
            "analysis_duration": upload.analysis_duration,
            "uploaded_at": upload.uploaded_at.isoformat() if upload.uploaded_at else None,
            "insights": analysis_summary,
            "has_charts": upload.chart_data is not None
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analysis: {str(e)}")


@app.get("/system/memory")
async def get_system_memory():
    """🖥️ Sistem memory durumu"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        "memory": {
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "available_gb": round(memory.available / (1024 ** 3), 2),
            "used_percentage": memory.percent,
            "status": "healthy" if memory.percent < 80 else "warning" if memory.percent < 90 else "critical"
        },
        "disk": {
            "total_gb": round(disk.total / (1024 ** 3), 2),
            "free_gb": round(disk.free / (1024 ** 3), 2),
            "used_percentage": round((disk.used / disk.total) * 100, 2)
        }
    }





if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
