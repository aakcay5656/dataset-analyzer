import pandas as pd
import numpy as np
from typing import Dict, Any, List

from openpyxl.styles.builtins import total


class DatasetAnalyzer:
    def __init__(self,df:pd.DataFrame):
        self.df = df

    def analyze(self)->Dict[str, Any]:
        """Tam dataset analizi"""
        return {
            "basic_info":self._get_basic_info(),
            "columns_analysis":self._analyze_columns(),
            "missing_values":self._analyzer_missing_values(),
            "statistics":self._get_statistics(),
            "insights":self._generate_insights()
        }

    def _get_basic_info(self) -> Dict[str, Any]:
        """Temel bilgiler"""

        return {
            "rows":len(self.df),
            "columns":len(self.df.columns),
            "columns_names":self.df.columns.tolist(),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
    def _analyze_columns(self)->Dict[str, Any]:
        """Sütun analizi"""
        columns_info = {}

        for col in self.df.columns:
            col_data = self.df[col]

            columns_info[col] = {
                "type":str(col_data.dtype),
                "non_null_count":int(col_data.count()),
                "null_count":int(col_data.isnull().sum()),
                "unique_count":int(col_data.nunique()),
                "sample_values":col_data.dropna().head(3).tolist()
            }

            # Numerik stünlar için ilave info
            if pd.api.types.is_numeric_dtype(col_data):
                columns_info[col].update({
                    "min":float(col_data.min()) if not col_data.empty else None,
                    "max":float(col_data.max()) if not col_data.empty else None,
                    "mean":float(col_data.mean()) if not col_data.empty else None,
                })
        return columns_info
    def _analyzer_missing_values(self)->Dict[str, Any]:
        """Eksik değer analizi"""
        missing_counts = self.df.isnull().sum()
        total_rows = len(self.df)

        missing_info = {}
        for col in self.df.columns:
            missing_count = missing_counts[col]
            missing_percentage = (missing_count / total_rows) * 100

            missing_info[col] = {
                "missing_count":int(missing_count),
                "missing_percentage":round(missing_percentage,2)
            }

        return {
            "total_missing":int(missing_counts.sum()),
            "missing_info":missing_info,
        }

    def _get_statistics(self)->Dict[str, Any]:
        """İstatiksel özet"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return {"message":"Numerik sütun bulunamadı"}

        stats = self.df[numeric_cols].describe()

        # JSON serializeable yap
        stats_dict = {}

        for col in numeric_cols:
            stats_dict[col] = {
                "count":int(stats.loc["count",col]),
                "mean":round(float(stats.loc["mean",col])),
                "std":round(float(stats.loc["std",col])),
                "min":float(stats.loc["min",col]),
                "25%":float(stats.loc["25%",col]),
                "50%":float(stats.loc["50%",col]),
                "75%":float(stats.loc["75%",col]),
                "max":float(stats.loc["max",col]),
            }
        return stats_dict

    def _generate_insights(self) -> List[str]:
        """Basit insights üret"""
        insights = []

        # Veri boyutu insights
        rows, cols = self.df.shape
        if rows > 10000:
            insights.append(f"📊 Büyük dataset: {rows:,} satır ile kapsamlı analiz imkanı")
        elif rows < 100:
            insights.append(f"📊 Küçük dataset: {rows} satır - istatistiksel güvenilirlik sınırlı")

        # Eksik değer insights
        missing_percentage = (self.df.isnull().sum().sum() / (rows * cols)) * 100
        if missing_percentage > 20:
            insights.append(f"⚠️ Yüksek eksik veri oranı: %{missing_percentage:.1f}")
        elif missing_percentage > 0:
            insights.append(f"📈 Düşük eksik veri oranı: %{missing_percentage:.1f}")
        else:
            insights.append("✅ Temiz dataset: Eksik veri yok")

        # Sütun tipi insights
        numeric_cols = len(self.df.select_dtypes(include=[np.number]).columns)
        if numeric_cols == 0:
            insights.append("📝 Sadece kategorik veriler - numerik analiz yapılamaz")
        elif numeric_cols > cols / 2:
            insights.append(f"🔢 Ağırlıklı numerik dataset: {numeric_cols}/{cols} sütun")

        # Unique değer insights
        for col in self.df.columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio > 0.9:
                insights.append(f"🆔 '{col}' muhtemelen ID sütunu (yüksek benzersizlik)")

        return insights