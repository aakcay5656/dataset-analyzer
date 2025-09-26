import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import json
from typing import Dict, Any, List, Optional
from collections import Counter
import re


class AdvancedChartGenerator:
    """Detailed chart builder for each column"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.column_analysis = self._analyze_columns()
        self.charts = {}

    def _analyze_columns(self) -> Dict[str, Dict]:
        """Analyze each column and determine its type"""
        analysis = {}

        for col in self.df.columns:
            col_data = self.df[col]
            clean_data = col_data.dropna()

            analysis[col] = {
                'dtype': str(col_data.dtype),
                'null_count': int(col_data.isnull().sum()),
                'null_percentage': float((col_data.isnull().sum() / len(col_data)) * 100),
                'unique_count': int(col_data.nunique()),
                'unique_percentage': float((col_data.nunique() / len(col_data)) * 100),
                'sample_values': [str(x) for x in clean_data.head(5).tolist()],
                'is_numeric': bool(pd.api.types.is_numeric_dtype(col_data)),
                'is_datetime': bool(pd.api.types.is_datetime64_any_dtype(col_data)),
                'is_categorical': bool(self._is_categorical(col_data)),
                'is_text': bool(self._is_text(col_data)),
                # 🆕 Object sütunlar için detaylı analiz
                'object_analysis': self._analyze_object_column(col_data) if col_data.dtype == 'object' else None
            }

        return analysis

    def _is_categorical(self, series: pd.Series) -> bool:
        """Determine if column is categorical"""
        if series.dtype == 'object':
            unique_ratio = series.nunique() / len(series)
            return unique_ratio < 0.1  # If there are less than 10% unique values, it is categorical.
        return False

    def _is_text(self, series: pd.Series) -> bool:
        """Determine if column is text"""
        if series.dtype == 'object' and not self._is_categorical(series):
            # Text if average string length is more than 10
            sample = series.dropna().head(100)
            if len(sample) > 0:
                avg_length = sample.astype(str).str.len().mean()
                return avg_length > 10
        return False

    def generate_all_charts(self) -> Dict[str, Any]:
        """Create all charts"""
        print("🎨 Advanced chart creation started...")

        # Dataset overview charts
        self.charts['dataset_overview'] = self._create_dataset_overview()

        # Detailed charts for each column
        for col in self.df.columns:
            col_info = self.column_analysis[col]
            self.charts[f'column_{col}'] = self._create_column_charts(col, col_info)

        # Inter-column analysis
        self.charts['correlation_analysis'] = self._create_correlation_analysis()
        self.charts['missing_values_analysis'] = self._create_missing_values_analysis()
        self.charts['data_quality_dashboard'] = self._create_data_quality_dashboard()

        print(f"✅ {len(self.charts)} chart group created!")
        return self.charts

    def _create_dataset_overview(self) -> Dict[str, Any]:
        """Dataset overview charts"""
        charts = {}

        # 1. Column Types Distribution
        type_counts = {}
        for col, info in self.column_analysis.items():
            if info['is_numeric']:
                chart_type = 'Numeric'
            elif info['is_datetime']:
                chart_type = 'DateTime'
            elif info['is_categorical']:
                chart_type = 'Categorical'
            elif info['is_text']:
                chart_type = 'Text'
            else:
                chart_type = 'Other'

            type_counts[chart_type] = type_counts.get(chart_type, 0) + 1

        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Column Types Distribution"
        )
        charts['column_types'] = {
            'config': json.loads(fig.to_json()),
            'insights': [f"{k}: {v} columns" for k, v in type_counts.items()]
        }

        # 2. Dataset Shape Info
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Rows', 'Columns'],
            y=[len(self.df), len(self.df.columns)],
            marker_color=['lightblue', 'lightgreen']
        ))
        fig.update_layout(title="Dataset Dimensions")

        charts['dataset_shape'] = {
            'config': json.loads(fig.to_json()),
            'insights': [
                f"Dataset has {len(self.df):,} rows and {len(self.df.columns)} columns",
                f"Total data points: {len(self.df) * len(self.df.columns):,}"
            ]
        }

        return {
            'type': 'dataset_overview',
            'charts': charts,
            'insights': [
                f"📊 Dataset contains {len(self.df):,} rows and {len(self.df.columns)} columns",
                f"🔢 Column type breakdown: {type_counts}",
                f"💾 Estimated memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
            ]
        }




    def _create_numeric_charts(self, column: str, data: pd.Series) -> Dict[str, Any]:
        """Charts for numeric columns"""
        charts = {}

        try:
            # 1. Histogram
            fig = px.histogram(x=data, nbins=30, title=f"{column} - Distribution")
            charts['histogram'] = {
                'config': json.loads(fig.to_json()),
                'insights': [f"Distribution shows data spread for {column}"]
            }

            # 2. Box Plot (Outlier Analysis)
            fig = px.box(y=data, title=f"{column} - Outlier Analysis")
            charts['boxplot'] = {
                'config': json.loads(fig.to_json()),
                'insights': self._analyze_outliers(data, column)
            }

            # 3. Descriptive Statistics
            stats = data.describe()

            # Convert pandas Series to dict with safe types
            safe_stats = {
                'mean': float(stats['mean']),
                'median': float(stats['50%']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max'])
            }

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                y=list(safe_stats.values()),
                marker_color=['blue', 'green', 'orange', 'red', 'purple']
            ))
            fig.update_layout(title=f"{column} - Statistical Summary")

            charts['statistics'] = {
                'config': json.loads(fig.to_json()),
                'insights': [
                    f"Mean: {safe_stats['mean']:.2f}",
                    f"Standard deviation: {safe_stats['std']:.2f}",
                    f"Range: {safe_stats['min']:.2f} to {safe_stats['max']:.2f}"
                ],
                'stats': safe_stats  # Safe stats dict
            }

            # 4. Density Plot
            try:
                import plotly.figure_factory as ff
                fig = ff.create_distplot([data.values], [column], show_hist=False)
                fig.update_layout(title=f"{column} - Density Distribution")
                charts['density'] = {
                    'config': json.loads(fig.to_json()),
                    'insights': ["Smooth density curve showing data distribution"]
                }
            except Exception as e:
                print(f"⚠️ Density plot error for {column}: {e}")

            return charts

        except Exception as e:
            print(f"❌ Numeric charts error for {column}: {e}")
            return {
                'error': {
                    'config': None,
                    'insights': [f"Numeric chart generation failed for {column}: {str(e)}"]
                }
            }

    def _create_categorical_charts(self, column: str, data: pd.Series) -> Dict[str, Any]:
        """Categorical column charts"""
        charts = {}

        # Value counts
        value_counts = data.value_counts().head(15)  # Top 15

        # 1. Bar Chart
        fig = px.bar(
            x=value_counts.values,
            y=value_counts.index,
            orientation='h',
            title=f"{column} - Frequency Distribution"
        )
        charts['bar_chart'] = {
            'config': json.loads(fig.to_json()),
            'insights': [f"Top category: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)"]
        }

        # 2. Pie Chart (if <= 8 categories)
        if len(value_counts) <= 8:
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"{column} - Proportion Distribution"
            )
            charts['pie_chart'] = {
                'config': json.loads(fig.to_json()),
                'insights': [
                    f"Most dominant category: {value_counts.index[0]} ({value_counts.iloc[0] / len(data) * 100:.1f}%)"]
            }

        # 3. Frequency Analysis
        total_unique = data.nunique()
        top_5_percent = (value_counts.head(5).sum() / len(data)) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(value_counts))),
            y=value_counts.values,
            mode='lines+markers',
            name='Frequency'
        ))
        fig.update_layout(title=f"{column} - Frequency Curve")
        charts['frequency_curve'] = {
            'config': json.loads(fig.to_json()),
            'insights': [
                f"Total unique values: {total_unique}",
                f"Top 5 categories represent {top_5_percent:.1f}% of data"
            ]
        }

        return charts

    def _create_datetime_charts(self, column: str, data: pd.Series) -> Dict[str, Any]:
        """Charts for DateTime column"""
        charts = {}

        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(data):
                data = pd.to_datetime(data, errors='coerce')

            data = data.dropna()

            # 1. Time Series Line Chart
            daily_counts = data.dt.date.value_counts().sort_index()
            fig = px.line(
                x=daily_counts.index,
                y=daily_counts.values,
                title=f"{column} - Time Series"
            )
            charts['time_series'] = {
                'config': json.loads(fig.to_json()),
                'insights': [f"Time series spanning from {data.min().date()} to {data.max().date()}"]
            }

            # 2. Date Distribution by Components
            components = {
                'Year': data.dt.year,
                'Month': data.dt.month,
                'Day of Week': data.dt.day_name(),
                'Hour': data.dt.hour if data.dt.hour.nunique() > 1 else None
            }

            for comp_name, comp_data in components.items():
                if comp_data is not None and comp_data.nunique() > 1:
                    comp_counts = comp_data.value_counts().sort_index()
                    fig = px.bar(
                        x=comp_counts.index,
                        y=comp_counts.values,
                        title=f"{column} - Distribution by {comp_name}"
                    )
                    charts[f'{comp_name.lower()}_distribution'] = {
                        'config': json.loads(fig.to_json()),
                        'insights': [f"Peak {comp_name}: {comp_counts.idxmax()} ({comp_counts.max()} occurrences)"]
                    }

        except Exception as e:
            charts['error'] = {
                'config': None,
                'insights': [f"DateTime analysis failed: {str(e)}"]
            }

        return charts

    def _create_text_charts(self, column: str, data: pd.Series) -> Dict[str, Any]:
        """Charts for text columns"""
        charts = {}

        text_series = data.astype(str)

        # 1. Text Length Distribution
        lengths = text_series.str.len()
        fig = px.histogram(x=lengths, nbins=20, title=f"{column} - Text Length Distribution")
        charts['length_distribution'] = {
            'config': json.loads(fig.to_json()),
            'insights': [
                f"Average text length: {lengths.mean():.1f} characters",
                f"Length range: {lengths.min()} to {lengths.max()} characters"
            ]
        }

        # 2. Word Count Analysis
        word_counts = text_series.str.split().str.len()
        fig = px.histogram(x=word_counts, nbins=15, title=f"{column} - Word Count Distribution")
        charts['word_count'] = {
            'config': json.loads(fig.to_json()),
            'insights': [f"Average words per entry: {word_counts.mean():.1f}"]
        }

        # 3. Character Analysis
        all_text = ' '.join(text_series.head(1000))
        char_counts = Counter(all_text.lower())
        common_chars = dict(char_counts.most_common(15))

        fig = px.bar(
            x=list(common_chars.keys()),
            y=list(common_chars.values()),
            title=f"{column} - Character Frequency"
        )
        charts['character_frequency'] = {
            'config': json.loads(fig.to_json()),
            'insights': [f"Most common character: '{list(common_chars.keys())[0]}'"]
        }

        return charts

    def _create_missing_chart(self, column: str, col_info: Dict) -> Dict[str, Any]:
        """Missing values chart"""
        missing_count = col_info['null_count']
        total_count = len(self.df)

        fig = px.pie(
            values=[missing_count, total_count - missing_count],
            names=['Missing', 'Present'],
            title=f"{column} - Missing Values",
            color_discrete_map={'Missing': 'red', 'Present': 'green'}
        )

        return {
            'config': json.loads(fig.to_json()),
            'insights': [
                f"{missing_count} missing values ({col_info['null_percentage']:.1f}%)",
                "Consider imputation strategies if missing percentage is high"
            ]
        }

    def _create_correlation_analysis(self) -> Dict[str, Any]:
        """Inter-column correlation analysis"""
        charts = {}

        # Numeric columns only
        numeric_cols = [col for col, info in self.column_analysis.items() if info['is_numeric']]

        if len(numeric_cols) >= 2:
            numeric_df = self.df[numeric_cols]
            correlation_matrix = numeric_df.corr()

            # Correlation Heatmap
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Inter-Column Correlation Matrix"
            )
            charts['correlation_heatmap'] = {
                'config': json.loads(fig.to_json()),
                'insights': self._analyze_correlations(correlation_matrix)
            }

            # Strong correlations scatter plots
            strong_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))

            for col1, col2, corr in strong_pairs[:3]:  # Top 3 correlations
                fig = px.scatter(
                    self.df,
                    x=col1,
                    y=col2,
                    title=f"{col1} vs {col2} (r={corr:.2f})",
                    trendline="ols"
                )
                charts[f'scatter_{col1}_{col2}'] = {
                    'config': json.loads(fig.to_json()),
                    'insights': [f"Strong {'positive' if corr > 0 else 'negative'} correlation: {corr:.3f}"]
                }

        return {
            'type': 'correlation_analysis',
            'charts': charts,
            'insights': [f"Analyzed correlations between {len(numeric_cols)} numeric columns"]
        }

    def _create_missing_values_analysis(self) -> Dict[str, Any]:
        """Missing values general analysis"""
        missing_data = {}
        for col, info in self.column_analysis.items():
            if info['null_count'] > 0:
                missing_data[col] = info['null_count']

        charts = {}

        if missing_data:
            # Missing values bar chart
            fig = px.bar(
                x=list(missing_data.keys()),
                y=list(missing_data.values()),
                title="Missing Values by Column"
            )
            charts['missing_by_column'] = {
                'config': json.loads(fig.to_json()),
                'insights': [
                    f"Columns with missing data: {len(missing_data)}",
                    f"Total missing values: {sum(missing_data.values())}"
                ]
            }

            # Missing values heatmap
            missing_df = self.df[list(missing_data.keys())].isnull()
            if len(missing_df.columns) > 1:
                fig = px.imshow(
                    missing_df.head(100).T,  # Transpose for better view
                    title="Missing Values Pattern (First 100 rows)",
                    color_continuous_scale=['white', 'red']
                )
                charts['missing_pattern'] = {
                    'config': json.loads(fig.to_json()),
                    'insights': ["Red areas show missing values pattern"]
                }

        return {
            'type': 'missing_values_analysis',
            'charts': charts,
            'insights': [
                f"Dataset completeness: {((len(self.df) * len(self.df.columns) - sum(missing_data.values())) / (len(self.df) * len(self.df.columns)) * 100):.1f}%"
            ] if missing_data else ["No missing values found!"]
        }

    def _create_data_quality_dashboard(self) -> Dict[str, Any]:
        """Data quality summary dashboard"""
        quality_metrics = {
            'Total Rows': len(self.df),
            'Total Columns': len(self.df.columns),
            'Numeric Columns': sum(1 for info in self.column_analysis.values() if info['is_numeric']),
            'Categorical Columns': sum(1 for info in self.column_analysis.values() if info['is_categorical']),
            'Text Columns': sum(1 for info in self.column_analysis.values() if info['is_text']),
            'Columns with Missing Values': sum(1 for info in self.column_analysis.values() if info['null_count'] > 0),
            'Total Missing Values': sum(info['null_count'] for info in self.column_analysis.values()),
            'Data Completeness (%)': ((len(self.df) * len(self.df.columns) - sum(
                info['null_count'] for info in self.column_analysis.values())) / (
                                                  len(self.df) * len(self.df.columns)) * 100)
        }

        # Quality metrics bar chart
        fig = px.bar(
            x=list(quality_metrics.keys()),
            y=list(quality_metrics.values()),
            title="Data Quality Metrics"
        )
        fig.update_xaxes(tickangle=45)

        return {
            'type': 'data_quality',
            'charts': {
                'quality_metrics': {
                    'config': json.loads(fig.to_json()),
                    'insights': [
                        f"Dataset is {quality_metrics['Data Completeness (%)']:.1f}% complete",
                        f"Quality score: {'Excellent' if quality_metrics['Data Completeness (%)'] > 95 else 'Good' if quality_metrics['Data Completeness (%)'] > 85 else 'Needs attention'}"
                    ]
                }
            },
            'metrics': quality_metrics,
            'insights': [
                f"📊 Dataset has {quality_metrics['Total Rows']:,} rows and {quality_metrics['Total Columns']} columns",
                f"🔢 Column types: {quality_metrics['Numeric Columns']} numeric, {quality_metrics['Categorical Columns']} categorical, {quality_metrics['Text Columns']} text",
                f"✅ Data completeness: {quality_metrics['Data Completeness (%)']:.1f}%"
            ]
        }

    def _analyze_outliers(self, data: pd.Series, column: str) -> List[str]:
        """Outlier analysis"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_percentage = (len(outliers) / len(data)) * 100

        insights = [
            f"IQR method: {len(outliers)} outliers detected ({outlier_percentage:.1f}%)",
            f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
        ]

        if outlier_percentage > 5:
            insights.append("⚠️ High outlier percentage - consider investigation")
        elif outlier_percentage < 1:
            insights.append("✅ Low outlier percentage - data looks clean")

        return insights

    def _analyze_correlations(self, corr_matrix: pd.DataFrame) -> List[str]:
        """Correlation analysis insights"""
        insights = []

        # Strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))

        if strong_correlations:
            for col1, col2, corr in strong_correlations[:3]:
                direction = "positive" if corr > 0 else "negative"
                insights.append(f"Strong {direction} correlation: {col1} ↔ {col2} (r={corr:.3f})")
        else:
            insights.append("No strong correlations (|r| > 0.7) found")

        return insights

    def get_chart_summary(self) -> Dict[str, Any]:
        """Summary of charts"""
        total_charts = 0
        chart_types = []

        for chart_group in self.charts.values():
            if isinstance(chart_group, dict) and 'charts' in chart_group:
                total_charts += len(chart_group['charts'])
                chart_types.extend(chart_group['charts'].keys())

        return {
            'total_chart_groups': len(self.charts),
            'total_individual_charts': total_charts,
            'chart_types': list(set(chart_types)),
            'columns_analyzed': len([k for k in self.charts.keys() if k.startswith('column_')])
        }



    def _analyze_object_column(self, series: pd.Series) -> Dict[str, Any]:
        """Detailed analysis for the Object column"""
        clean_data = series.dropna()

        if len(clean_data) == 0:
            return {'type': 'empty', 'patterns': {}}

        # Sample for analysis (first 1000 values)
        sample = clean_data.head(1000)

        analysis = {
            'length_stats': {
                'min': int(sample.astype(str).str.len().min()),
                'max': int(sample.astype(str).str.len().max()),
                'mean': float(sample.astype(str).str.len().mean()),
                'median': float(sample.astype(str).str.len().median())
            },
            'patterns': {},
            'data_types': {},
            'suggested_type': 'mixed'
        }

        # Pattern detection
        patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?[\d\s\-\(\)]{10,}$',
            'url': r'^https?://[^\s]+$',
            'date_like': r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}',
            'numeric_string': r'^\d+\.?\d*$',
            'code_like': r'^[A-Z0-9]{2,}-?[A-Z0-9]{2,}$',
            'boolean_like': r'^(true|false|yes|no|1|0|y|n)$'
        }

        for pattern_name, pattern in patterns.items():
            matches = sample.astype(str).str.match(pattern, case=False).sum()
            analysis['patterns'][pattern_name] = {
                'count': int(matches),
                'percentage': float(matches / len(sample) * 100)
            }

        # Data type detection
        numeric_count = 0
        date_count = 0

        for val in sample.astype(str):
            # Try numeric conversion
            try:
                float(val)
                numeric_count += 1
            except:
                pass

            # Try date conversion
            try:
                pd.to_datetime(val, errors='raise')
                date_count += 1
            except:
                pass

        analysis['data_types'] = {
            'numeric_convertible': {
                'count': numeric_count,
                'percentage': float(numeric_count / len(sample) * 100)
            },
            'date_convertible': {
                'count': date_count,
                'percentage': float(date_count / len(sample) * 100)
            }
        }

        # Suggest best type
        if analysis['data_types']['numeric_convertible']['percentage'] > 80:
            analysis['suggested_type'] = 'numeric_string'
        elif analysis['data_types']['date_convertible']['percentage'] > 80:
            analysis['suggested_type'] = 'date_string'
        elif analysis['patterns']['email']['percentage'] > 50:
            analysis['suggested_type'] = 'email'
        elif analysis['patterns']['phone']['percentage'] > 50:
            analysis['suggested_type'] = 'phone'
        elif analysis['patterns']['url']['percentage'] > 50:
            analysis['suggested_type'] = 'url'
        elif analysis['patterns']['code_like']['percentage'] > 50:
            analysis['suggested_type'] = 'identifier'
        elif len(clean_data.unique()) / len(clean_data) < 0.05:  # Less than 5% unique
            analysis['suggested_type'] = 'categorical'
        elif analysis['length_stats']['mean'] > 50:
            analysis['suggested_type'] = 'long_text'
        else:
            analysis['suggested_type'] = 'short_text'

        return analysis

    def _create_column_charts(self, column: str, col_info: Dict) -> Dict[str, Any]:
        """Detailed charts for a specific column"""
        charts = {}
        insights = []

        col_data = self.df[column].dropna()

        try:
            if col_info['is_numeric']:
                # Numeric column charts
                charts.update(self._create_numeric_charts(column, col_data))
                insights.extend([
                    f"📊 {column}: Numeric column with {len(col_data)} values",
                    f"📈 Range: {col_data.min():.2f} to {col_data.max():.2f}",
                    f"📍 Mean: {col_data.mean():.2f}, Median: {col_data.median():.2f}"
                ])

            elif col_info['is_datetime']:
                # DateTime column charts
                charts.update(self._create_datetime_charts(column, col_data))
                insights.extend([
                    f"📅 {column}: DateTime column with {len(col_data)} values",
                    f"⏰ Range: {col_data.min()} to {col_data.max()}"
                ])

            elif col_info['dtype'] == 'object':
                # 🆕 SMART OBJECT COLUMN HANDLING
                charts.update(self._create_object_charts(column, col_data, col_info))

                object_analysis = col_info.get('object_analysis', {})
                suggested_type = object_analysis.get('suggested_type', 'mixed')

                insights.extend([
                    f"📝 {column}: Object column with {len(col_data)} values",
                    f"🎯 Detected as: {suggested_type.replace('_', ' ').title()}",
                    f"📏 Average length: {object_analysis.get('length_stats', {}).get('mean', 0):.1f} characters"
                ])

            else:
                # Fallback for other types
                charts.update(self._create_generic_charts(column, col_data))
                insights.extend([
                    f"❓ {column}: {col_info['dtype']} column with {len(col_data)} values"
                ])

            # Missing values chart (for all columns)
            if col_info['null_count'] > 0:
                charts['missing_values'] = self._create_missing_chart(column, col_info)
                insights.append(f"⚠️ {col_info['null_count']} missing values ({col_info['null_percentage']:.1f}%)")

        except Exception as e:
            print(f"❌ Error creating charts for {column}: {e}")
            charts['error'] = {
                'config': None,
                'insights': [f"Chart generation failed: {str(e)}"]
            }

        return {
            'type': 'column_analysis',
            'column': column,
            'column_info': col_info,
            'charts': charts,
            'insights': insights
        }

    def _create_object_charts(self, column: str, data: pd.Series, col_info: Dict) -> Dict[str, Any]:
        """Smart charts for Object column"""
        charts = {}
        object_analysis = col_info.get('object_analysis', {})
        suggested_type = object_analysis.get('suggested_type', 'mixed')

        try:
            # 1. Data Type Distribution
            type_dist = object_analysis.get('data_types', {})
            if type_dist:
                categories = []
                values = []

                numeric_pct = type_dist.get('numeric_convertible', {}).get('percentage', 0)
                date_pct = type_dist.get('date_convertible', {}).get('percentage', 0)
                other_pct = 100 - numeric_pct - date_pct

                if numeric_pct > 0:
                    categories.append('Numeric-like')
                    values.append(numeric_pct)
                if date_pct > 0:
                    categories.append('Date-like')
                    values.append(date_pct)
                if other_pct > 0:
                    categories.append('Text/Other')
                    values.append(other_pct)

                if categories:
                    fig = px.pie(
                        values=values,
                        names=categories,
                        title=f"{column} - Data Type Distribution"
                    )
                    charts['data_type_distribution'] = {
                        'config': json.loads(fig.to_json()),
                        'insights': [f"Most values are: {categories[values.index(max(values))]} ({max(values):.1f}%)"]
                    }

            # 2. Length Distribution
            length_stats = object_analysis.get('length_stats', {})
            if length_stats:
                lengths = data.astype(str).str.len()
                fig = px.histogram(
                    x=lengths,
                    nbins=20,
                    title=f"{column} - Text Length Distribution"
                )
                charts['length_distribution'] = {
                    'config': json.loads(fig.to_json()),
                    'insights': [
                        f"Average length: {length_stats.get('mean', 0):.1f} characters",
                        f"Range: {length_stats.get('min', 0)} to {length_stats.get('max', 0)} characters"
                    ]
                }

            # 3. Pattern Analysis
            patterns = object_analysis.get('patterns', {})
            significant_patterns = {k: v for k, v in patterns.items() if v.get('percentage', 0) > 5}

            if significant_patterns:
                pattern_names = list(significant_patterns.keys())
                pattern_values = [v['percentage'] for v in significant_patterns.values()]

                fig = px.bar(
                    x=pattern_names,
                    y=pattern_values,
                    title=f"{column} - Pattern Detection",
                    labels={'x': 'Pattern Type', 'y': 'Percentage (%)'}
                )
                charts['pattern_analysis'] = {
                    'config': json.loads(fig.to_json()),
                    'insights': [f"Detected patterns: {', '.join(pattern_names)}"]
                }

            # 4. Value Frequency (for categorical-like data)
            if suggested_type in ['categorical', 'identifier', 'short_text']:
                value_counts = data.value_counts().head(15)

                if len(value_counts) > 1:
                    fig = px.bar(
                        x=value_counts.values,
                        y=value_counts.index,
                        orientation='h',
                        title=f"{column} - Most Frequent Values"
                    )
                    charts['value_frequency'] = {
                        'config': json.loads(fig.to_json()),
                        'insights': [
                            f"Most frequent: '{value_counts.index[0]}' ({value_counts.iloc[0]} times)",
                            f"Total unique values: {len(data.unique())}"
                        ]
                    }

            # 5. Convertibility Analysis (for numeric/date-like data)
            if suggested_type in ['numeric_string', 'date_string']:
                if suggested_type == 'numeric_string':
                    try:
                        # Try converting to numeric and show distribution
                        numeric_converted = pd.to_numeric(data, errors='coerce').dropna()
                        if len(numeric_converted) > 0:
                            fig = px.histogram(
                                x=numeric_converted,
                                nbins=30,
                                title=f"{column} - Numeric Distribution (Converted)"
                            )
                            charts['numeric_converted'] = {
                                'config': json.loads(fig.to_json()),
                                'insights': [
                                    f"Successfully converted {len(numeric_converted)} values to numeric",
                                    f"Range: {numeric_converted.min():.2f} to {numeric_converted.max():.2f}"
                                ]
                            }
                    except Exception as e:
                        print(f"⚠️ Numeric conversion chart error: {e}")

                elif suggested_type == 'date_string':
                    try:
                        # Try converting to dates and show timeline
                        date_converted = pd.to_datetime(data, errors='coerce').dropna()
                        if len(date_converted) > 0:
                            date_counts = date_converted.dt.date.value_counts().sort_index().head(50)
                            fig = px.line(
                                x=date_counts.index,
                                y=date_counts.values,
                                title=f"{column} - Date Timeline (Converted)"
                            )
                            charts['date_converted'] = {
                                'config': json.loads(fig.to_json()),
                                'insights': [
                                    f"Successfully converted {len(date_converted)} values to dates",
                                    f"Date range: {date_converted.min().date()} to {date_converted.max().date()}"
                                ]
                            }
                    except Exception as e:
                        print(f"⚠️ Date conversion chart error: {e}")

            return charts

        except Exception as e:
            print(f"❌ Object charts error for {column}: {e}")
            return {
                'error': {
                    'config': None,
                    'insights': [f"Object chart generation failed: {str(e)}"]
                }
            }

    def _create_generic_charts(self, column: str, data: pd.Series) -> Dict[str, Any]:
        """Generic fallback charts"""
        charts = {}

        try:
            # Basic value counts
            value_counts = data.value_counts().head(20)

            if len(value_counts) > 1:
                fig = px.bar(
                    x=value_counts.values,
                    y=[str(x) for x in value_counts.index],
                    orientation='h',
                    title=f"{column} - Value Distribution"
                )
                charts['value_distribution'] = {
                    'config': json.loads(fig.to_json()),
                    'insights': [
                        f"Data type: {data.dtype}",
                        f"Unique values: {data.nunique()}",
                        f"Most common: {str(value_counts.index[0])}"
                    ]
                }

            return charts

        except Exception as e:
            return {
                'error': {
                    'config': None,
                    'insights': [f"Generic chart error: {str(e)}"]
                }
            }