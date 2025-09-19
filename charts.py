import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from typing import Dict, Any, List


class ChartGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_all_charts(self) -> Dict[str, Any]:
        """Tüm chart tiplerini oluştur"""
        charts = {}

        # Numeric ve categorical sütunları belirle
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        # 1. Histogram (ilk numeric sütun için)
        if numeric_cols:
            charts['histogram'] = self.create_histogram(numeric_cols[0])

        # 2. Correlation heatmap (2+ numeric sütun varsa)
        if len(numeric_cols) >= 2:
            charts['correlation'] = self.create_correlation_heatmap(numeric_cols[:5])  # İlk 5 sütun

        # 3. Box plot (outlier analizi için)
        if numeric_cols:
            charts['boxplot'] = self.create_boxplot(numeric_cols[0])

        # 4. Kategorik dağılım (ilk categorical sütun için)
        if categorical_cols:
            charts['categorical'] = self.create_categorical_chart(categorical_cols[0])

        return charts

    def create_histogram(self, column: str, bins: int = 30) -> Dict[str, Any]:
        """Histogram oluştur"""
        data = self.df[column].dropna()

        # Plotly histogram
        fig = px.histogram(
            x=data,
            nbins=bins,
            title=f'{column} Histogram',
            labels={'x': column, 'y': 'Frequency'}
        )

        fig.update_layout(
            template='plotly_white',
            height=400,
            showlegend=False
        )

        # İstatistikler
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()

        insights = [
            f"📊 {len(data):,} veri noktası analiz edildi",
            f"📈 Ortalama: {mean_val:.2f}",
            f"📍 Medyan: {median_val:.2f}",
            f"📏 Standart sapma: {std_val:.2f}"
        ]

        return {
            "type": "histogram",
            "column": column,
            "config": json.loads(fig.to_json()),
            "insights": insights,
            "stats": {
                "mean": round(mean_val, 2),
                "median": round(median_val, 2),
                "std": round(std_val, 2),
                "count": len(data)
            }
        }

    def create_correlation_heatmap(self, columns: List[str]) -> Dict[str, Any]:
        """Korelasyon heatmap oluştur"""
        corr_matrix = self.df[columns].corr()

        # Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title="Korelasyon Matrisi",
            template='plotly_white',
            height=500,
            width=500
        )

        # Güçlü korelasyonları bul
        strong_correlations = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    direction = "pozitif" if corr_val > 0 else "negatif"
                    strong_correlations.append(
                        f"🔗 {columns[i]} - {columns[j]}: {direction} korelasyon ({corr_val:.2f})"
                    )

        insights = strong_correlations if strong_correlations else [
            "📊 Güçlü korelasyon bulunamadı (|r| > 0.7)",
            f"🔢 {len(columns)} değişken arasındaki ilişkiler incelendi"
        ]

        return {
            "type": "correlation_heatmap",
            "columns": columns,
            "config": json.loads(fig.to_json()),
            "insights": insights,
            "correlation_matrix": corr_matrix.round(2).to_dict()
        }

    def create_boxplot(self, column: str) -> Dict[str, Any]:
        """Box plot oluştur (outlier analizi)"""
        data = self.df[column].dropna()

        # Plotly box plot
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=data,
            name=column,
            boxmean='sd'  # Show mean and std deviation
        ))

        fig.update_layout(
            title=f'{column} - Outlier Analizi',
            template='plotly_white',
            height=400,
            yaxis_title=column
        )

        # Outlier hesaplama
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_percentage = (len(outliers) / len(data)) * 100

        insights = [
            f"📊 {len(data):,} veri noktası analiz edildi",
            f"📦 Q1: {Q1:.2f}, Q3: {Q3:.2f}",
            f"📏 IQR: {IQR:.2f}",
            f"🚨 {len(outliers)} aykırı değer tespit edildi (%{outlier_percentage:.1f})"
        ]

        return {
            "type": "boxplot",
            "column": column,
            "config": json.loads(fig.to_json()),
            "insights": insights,
            "outlier_stats": {
                "count": len(outliers),
                "percentage": round(outlier_percentage, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2)
            }
        }

    def create_categorical_chart(self, column: str) -> Dict[str, Any]:
        """Kategorik değişken için pie/bar chart"""
        value_counts = self.df[column].value_counts().head(10)  # Top 10

        if len(value_counts) <= 5:
            # Pie chart for few categories
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'{column} Dağılımı'
            )
        else:
            # Bar chart for many categories
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'{column} Dağılımı'
            )
            fig.update_xaxes(tickangle=45)

        fig.update_layout(
            template='plotly_white',
            height=400
        )

        total_count = value_counts.sum()
        top_category = value_counts.index[0]
        top_percentage = (value_counts.iloc[0] / total_count) * 100

        insights = [
            f"📊 {len(value_counts)} farklı kategori",
            f"🏆 En sık: '{top_category}' (%{top_percentage:.1f})",
            f"🔢 Toplam {total_count:,} veri noktası"
        ]

        if top_percentage > 80:
            insights.append("⚠️ Tek kategori dominant - dağılım dengesiz")

        return {
            "type": "categorical",
            "column": column,
            "config": json.loads(fig.to_json()),
            "insights": insights,
            "value_counts": value_counts.to_dict()
        }
