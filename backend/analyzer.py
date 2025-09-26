import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List
import time


class MultiFormatAnalyzer:
    def __init__(self, file_path: str, file_type: str):
        self.file_path = file_path
        self.file_type = file_type.lower()
        self.df = None

    def load_data(self) -> bool:
        """Load data by file type"""
        try:
            if self.file_type == '.csv':
                self.df = pd.read_csv(self.file_path)
            elif self.file_type in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.file_path)
            elif self.file_type == '.json':
                # Convert JSON to DataFrame
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Process according to JSON format
                if isinstance(data, list):
                    self.df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Convert Dict to DataFrame
                    if all(isinstance(v, list) for v in data.values()):
                        self.df = pd.DataFrame(data)
                    else:
                        # Single record dict
                        self.df = pd.DataFrame([data])
                else:
                    raise ValueError("Unsupported JSON format")
            else:
                raise ValueError(f"Unsupported file type: {self.file_type}")

            return True

        except Exception as e:
            raise Exception(f"File upload error ({self.file_type}): {str(e)}")

    def analyze(self) -> Dict[str, Any]:
        """Full analysis - with timing"""
        start_time = time.time()

        if self.df is None:
            raise Exception("Önce load_data() çağırın")

        result = {
            "file_info": self._get_file_info(),
            "basic_info": self._get_basic_info(),
            "columns_analysis": self._analyze_columns(),
            "missing_values": self._analyze_missing_values(),
            "statistics": self._get_statistics(),
            "data_quality": self._assess_data_quality(),
            "insights": self._generate_insights()
        }

        # Add analysis time
        analysis_duration = time.time() - start_time
        result["analysis_duration"] = round(analysis_duration, 3)

        return result

    def _get_file_info(self) -> Dict[str, Any]:
        """File information"""
        return {
            "file_type": self.file_type,
            "file_path": self.file_path,
            "encoding": "utf-8"  # Varsayılan
        }

    def _get_basic_info(self) -> Dict[str, Any]:
        """Basic information"""
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": self.df.columns.tolist(),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }

    def _analyze_columns(self) -> Dict[str, Any]:
        """Advanced column analysis"""
        columns_info = {}

        for col in self.df.columns:
            col_data = self.df[col]

            # Basic information
            info = {
                "type": str(col_data.dtype),
                "non_null_count": int(col_data.count()),
                "null_count": int(col_data.isnull().sum()),
                "unique_count": int(col_data.nunique()),
                "duplicate_count": int(len(col_data) - col_data.nunique()),
                "sample_values": col_data.dropna().head(3).tolist()
            }

            # Additional information for numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                info.update({
                    "min": float(col_data.min()) if not col_data.empty else None,
                    "max": float(col_data.max()) if not col_data.empty else None,
                    "mean": round(float(col_data.mean()), 2) if not col_data.empty else None,
                    "median": round(float(col_data.median()), 2) if not col_data.empty else None,
                    "std": round(float(col_data.std()), 2) if not col_data.empty else None
                })

                # Outlier detection (basic)
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                info["outlier_count"] = len(outliers)

            # For string columns
            elif col_data.dtype == 'object':
                # Most frequent values
                top_values = col_data.value_counts().head(3)
                info["top_values"] = top_values.to_dict()

                # String length analysis
                str_lengths = col_data.astype(str).str.len()
                info.update({
                    "avg_length": round(str_lengths.mean(), 1),
                    "min_length": int(str_lengths.min()),
                    "max_length": int(str_lengths.max())
                })

            columns_info[col] = info

        return columns_info

    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Missing value analysis"""
        missing_counts = self.df.isnull().sum()
        total_rows = len(self.df)

        missing_info = {}
        for col in self.df.columns:
            missing_count = missing_counts[col]
            missing_percentage = (missing_count / total_rows) * 100

            missing_info[col] = {
                "missing_count": int(missing_count),
                "missing_percentage": round(missing_percentage, 2)
            }

        return {
            "total_missing": int(missing_counts.sum()),
            "total_percentage": round((missing_counts.sum() / (total_rows * len(self.df.columns))) * 100, 2),
            "by_column": missing_info
        }

    def _get_statistics(self) -> Dict[str, Any]:
        """Statistical summary"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return {"message": "Numeric column not found"}

        stats = self.df[numeric_cols].describe()

        # Make JSON serializable
        stats_dict = {}
        for col in numeric_cols:
            stats_dict[col] = {
                "count": int(stats.loc['count', col]),
                "mean": round(float(stats.loc['mean', col]), 2),
                "std": round(float(stats.loc['std', col]), 2),
                "min": float(stats.loc['min', col]),
                "25%": float(stats.loc['25%', col]),
                "50%": float(stats.loc['50%', col]),
                "75%": float(stats.loc['75%', col]),
                "max": float(stats.loc['max', col])
            }

        return stats_dict

    def _assess_data_quality(self) -> Dict[str, Any]:
        """Data quality assessment"""
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = self.df.isnull().sum().sum()

        # Completeness score
        completeness = ((total_cells - missing_cells) / total_cells) * 100

        # Uniqueness score (duplicate rows)
        duplicate_rows = len(self.df) - len(self.df.drop_duplicates())
        uniqueness = ((len(self.df) - duplicate_rows) / len(self.df)) * 100

        # Overall quality score
        quality_score = (completeness + uniqueness) / 2

        return {
            "completeness_percentage": round(completeness, 2),
            "uniqueness_percentage": round(uniqueness, 2),
            "duplicate_rows": duplicate_rows,
            "quality_score": round(quality_score, 2)
        }

    def _generate_insights(self) -> List[str]:
        """Advanced insights"""
        insights = []

        rows, cols = self.df.shape

        # Data size
        if rows > 50000:
            insights.append(f"📊 Large dataset: {rows:,} rows - robust analysis opportunity")
        elif rows < 100:
            insights.append(f"📊 Small dataset: {rows} rows - statistical reliability is limited")
        else:
            insights.append(f"📊 Medium size dataset: {rows:,} rows - good analysis possibilities")

        # Data quality
        quality = self._assess_data_quality()
        if quality["quality_score"] > 90:
            insights.append("✅ High quality data – ideal for analysis")
        elif quality["quality_score"] > 70:
            insights.append("⚠️ Medium quality data - may require cleaning")
        else:
            insights.append("❌ Low quality data – serious cleaning required")

        # Numerik vs kategorik
        numeric_cols = len(self.df.select_dtypes(include=[np.number]).columns)
        if numeric_cols == 0:
            insights.append("📝 Purely categorical data - statistical analysis limited")
        elif numeric_cols > cols / 2:
            insights.append(f"🔢 Weighted numeric data: {numeric_cols}/{cols} column")

        # Outlier insights
        outlier_cols = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                outlier_cols.append(col)

        if outlier_cols:
            insights.append(f"🚨 Outlier detected: {', '.join(outlier_cols[:3])}")

        # File format insight
        insights.append(f"📄 {self.file_type.upper()} was successfully analyzed in the format")

        return insights
