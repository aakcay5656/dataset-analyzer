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
        """Create all chart types"""
        charts = {}

        # Identify numeric and categorical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        # 1.Histogram (for the first numeric column)
        if numeric_cols:
            charts['histogram'] = self.create_histogram(numeric_cols[0])

        # 2. Correlation heatmap (If there are 2+ numeric columns)
        if len(numeric_cols) >= 2:
            charts['correlation'] = self.create_correlation_heatmap(numeric_cols[:5])  # first 5 columns

        # 3. Box plot (for outlier analysis)
        if numeric_cols:
            charts['boxplot'] = self.create_boxplot(numeric_cols[0])

        # 4. Categorical distribution (for the first categorical column
        if categorical_cols:
            charts['categorical'] = self.create_categorical_chart(categorical_cols[0])

        return charts

    def create_histogram(self, column: str, bins: int = 30) -> Dict[str, Any]:
        """Create histogram"""
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

        # Statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()

        insights = [
            f"📊 {len(data):,} data point analyzed",
            f"📈 Average: {mean_val:.2f}",
            f"📍 Median: {median_val:.2f}",
            f"📏 standard deviation: {std_val:.2f}"
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
        """Create correlation heatmap"""
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

        # Find strong correlations
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
            "📊 No strong correlation found (|r| > 0.7)",
            f"🔢 {len(columns)} Relationships between variables were examined"
        ]

        return {
            "type": "correlation_heatmap",
            "columns": columns,
            "config": json.loads(fig.to_json()),
            "insights": insights,
            "correlation_matrix": corr_matrix.round(2).to_dict()
        }

    def create_boxplot(self, column: str) -> Dict[str, Any]:
        """Create a box plot (outlier analysis)"""
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

        # Outlier calculation
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_percentage = (len(outliers) / len(data)) * 100

        insights = [
            f"📊 {len(data):,} data point analyzed",
            f"📦 Q1: {Q1:.2f}, Q3: {Q3:.2f}",
            f"📏 IQR: {IQR:.2f}",
            f"🚨 {len(outliers)} outlier detected (%{outlier_percentage:.1f})"
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
        """Pie/bar chart for categorical variables"""
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
            f"📊 {len(value_counts)} different category",
            f"🏆 most common: '{top_category}' (%{top_percentage:.1f})",
            f"🔢 Toplam {total_count:,} data point"
        ]

        if top_percentage > 80:
            insights.append("⚠️ Single category dominant - distribution unbalanced")

        return {
            "type": "categorical",
            "column": column,
            "config": json.loads(fig.to_json()),
            "insights": insights,
            "value_counts": value_counts.to_dict()
        }
