import pandas as pd
import numpy as np
import json
import time
import psutil
import os
from typing import Dict, Any, Iterator, Optional


class StreamingProcessor:
    """Büyük CSV dosyaları için memory-efficient streaming processor"""

    def __init__(self, file_path: str, chunk_size: int = 100000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.total_rows = 0
        self.processed_chunks = 0
        self.memory_usage = []

    def get_file_info(self) -> Dict[str, Any]:
        """Dosya hakkında temel bilgiler"""
        file_size = os.path.getsize(self.file_path)

        # İlk chunk'ı okuyarak column bilgisi al
        first_chunk = pd.read_csv(self.file_path, nrows=1000)

        return {
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "estimated_rows": self._estimate_total_rows(),
            "columns": first_chunk.columns.tolist(),
            "column_count": len(first_chunk.columns),
            "recommended_chunk_size": self._calculate_optimal_chunk_size(file_size)
        }

    def _estimate_total_rows(self) -> int:
        """Dosya boyutuna göre toplam satır sayısını tahmin et"""
        try:
            # İlk 10000 satırı oku, ortalama satır boyutunu hesapla
            sample = pd.read_csv(self.file_path, nrows=10000)
            file_size = os.path.getsize(self.file_path)

            # CSV header'ı hariç gerçek veri boyutu
            sample_size = sample.memory_usage(deep=True).sum()
            estimated_total = int((file_size / sample_size) * len(sample))

            return estimated_total
        except:
            return 0

    def _calculate_optimal_chunk_size(self, file_size_bytes: int) -> int:
        """Dosya boyutuna göre optimal chunk size hesapla"""
        file_size_mb = file_size_bytes / (1024 * 1024)

        if file_size_mb < 50:  # 50MB'den küçük
            return 50000
        elif file_size_mb < 500:  # 500MB'den küçük
            return 100000
        elif file_size_mb < 2000:  # 2GB'den küçük
            return 200000
        else:  # 2GB'den büyük
            return 500000

    def stream_analysis(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Ana streaming analiz fonksiyonu"""
        start_time = time.time()

        # Running statistics için container
        running_stats = {
            'total_rows': 0,
            'numeric_stats': {},
            'categorical_stats': {},
            'missing_values': {},
            'column_types': {},
            'memory_snapshots': []
        }

        chunk_count = 0

        print(f"🔄 Streaming analiz başlıyor - chunk size: {self.chunk_size:,}")

        try:
            # CSV'yi chunk'lar halinde oku
            chunk_iterator = pd.read_csv(self.file_path, chunksize=self.chunk_size)

            for chunk_num, chunk in enumerate(chunk_iterator):
                # Memory monitoring
                memory_percent = psutil.virtual_memory().percent
                running_stats['memory_snapshots'].append({
                    'chunk': chunk_num,
                    'memory_percent': memory_percent
                })

                # Chunk analizi
                self._process_chunk(chunk, running_stats)

                chunk_count += 1
                self.processed_chunks = chunk_count

                # Progress callback
                if progress_callback:
                    progress = {
                        'chunks_processed': chunk_count,
                        'rows_processed': running_stats['total_rows'],
                        'memory_usage': memory_percent
                    }
                    progress_callback(progress)

                # Memory kontrolü
                if memory_percent > 85:
                    print(f"⚠️ Yüksek memory kullanımı: %{memory_percent}")

                # Her 10 chunk'ta progress yazdır
                if chunk_count % 10 == 0:
                    print(f"📊 İşlenen chunk: {chunk_count}, satır: {running_stats['total_rows']:,}")

        except Exception as e:
            print(f"❌ Streaming analiz hatası: {str(e)}")
            raise e

        # Final statistics hesapla
        final_stats = self._finalize_statistics(running_stats)

        analysis_duration = time.time() - start_time
        final_stats['performance'] = {
            'analysis_duration': round(analysis_duration, 2),
            'chunks_processed': chunk_count,
            'rows_per_second': round(final_stats['total_rows'] / analysis_duration, 2),
            'avg_memory_usage': round(np.mean([s['memory_percent'] for s in running_stats['memory_snapshots']]), 2),
            'max_memory_usage': max([s['memory_percent'] for s in running_stats['memory_snapshots']])
        }

        print(f"✅ Streaming analiz tamamlandı: {final_stats['total_rows']:,} satır, {analysis_duration:.2f}s")

        return final_stats

    def _process_chunk(self, chunk: pd.DataFrame, running_stats: Dict[str, Any]):
        """Her chunk için analiz işlemi"""
        running_stats['total_rows'] += len(chunk)

        # Column types (ilk chunk'ta belirle)
        if not running_stats['column_types']:
            running_stats['column_types'] = {col: str(dtype) for col, dtype in chunk.dtypes.items()}

        # Numeric columns processing
        for col in chunk.select_dtypes(include=[np.number]).columns:
            col_data = chunk[col].dropna()

            if col not in running_stats['numeric_stats']:
                running_stats['numeric_stats'][col] = {
                    'count': 0,
                    'sum': 0.0,
                    'sum_squares': 0.0,
                    'min': float('inf'),
                    'max': float('-inf')
                }

            stats = running_stats['numeric_stats'][col]
            stats['count'] += len(col_data)
            stats['sum'] += col_data.sum()
            stats['sum_squares'] += (col_data ** 2).sum()
            stats['min'] = min(stats['min'], col_data.min())
            stats['max'] = max(stats['max'], col_data.max())

        # Categorical columns processing (top 100 values only)
        for col in chunk.select_dtypes(include=['object']).columns:
            if col not in running_stats['categorical_stats']:
                running_stats['categorical_stats'][col] = {}

            value_counts = chunk[col].value_counts()

            # Sadece top 100 değeri sakla (memory için)
            for value, count in value_counts.head(100).items():
                running_stats['categorical_stats'][col][value] = \
                    running_stats['categorical_stats'][col].get(value, 0) + count

        # Missing values
        missing_in_chunk = chunk.isnull().sum()
        for col, missing_count in missing_in_chunk.items():
            if col not in running_stats['missing_values']:
                running_stats['missing_values'][col] = 0
            running_stats['missing_values'][col] += missing_count

    def _finalize_statistics(self, running_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Final istatistikleri hesapla"""
        final_stats = {
            'total_rows': running_stats['total_rows'],
            'column_types': running_stats['column_types'],
            'numeric_summary': {},
            'categorical_summary': {},
            'missing_values_summary': {}
        }

        # Numeric statistics finalize
        for col, stats in running_stats['numeric_stats'].items():
            if stats['count'] > 0:
                mean = stats['sum'] / stats['count']
                variance = (stats['sum_squares'] / stats['count']) - (mean ** 2)
                std = np.sqrt(max(0, variance))  # Negative variance koruması

                final_stats['numeric_summary'][col] = {
                    'count': stats['count'],
                    'mean': round(mean, 3),
                    'std': round(std, 3),
                    'min': stats['min'],
                    'max': stats['max'],
                    'range': stats['max'] - stats['min']
                }

        # Categorical statistics finalize (top 10)
        for col, value_counts in running_stats['categorical_stats'].items():
            sorted_counts = dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            total_count = sum(value_counts.values())

            final_stats['categorical_summary'][col] = {
                'unique_count': len(value_counts),
                'top_values': sorted_counts,
                'total_count': total_count
            }

        # Missing values summary
        for col, missing_count in running_stats['missing_values'].items():
            missing_percentage = (missing_count / running_stats['total_rows']) * 100
            final_stats['missing_values_summary'][col] = {
                'missing_count': missing_count,
                'missing_percentage': round(missing_percentage, 2)
            }

        return final_stats
