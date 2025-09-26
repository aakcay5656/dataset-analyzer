from typing import Dict, Any,List

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import time
from datetime import datetime
import os

# Streamlit page config
st.set_page_config(
    page_title="Dataset Analyzer v3.1",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;        
        text-align: center;
        margin-bottom: 1rem;
    }
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .pattern-badge {
        background: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .object-analysis {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Backend API URL
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'upload_history' not in st.session_state:
    st.session_state.upload_history = []


# Helper Functions (same as before)
def safe_format_number(value, default=0):
    if value is None:
        return str(default)
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(default)


def safe_get_value(data, key, default="N/A"):
    if not data or not isinstance(data, dict):
        return default
    value = data.get(key)
    return value if value is not None else default


def call_api(endpoint, method="GET", files=None, params=None):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "POST" and files:
            response = requests.post(url, files=files, params=params, timeout=300)
        elif method == "GET":
            response = requests.get(url, params=params, timeout=30)
        else:
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("🔴 Backend bağlantısı başarısız!")
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ İstek zaman aşımı!")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Hatası: {str(e)}")
        return None


def load_upload_history():
    history = call_api("/history", params={"limit": 20})
    if history:
        st.session_state.upload_history = history.get('uploads', [])


def format_file_size(size_bytes):
    if not size_bytes or size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


# 🆕  Chart Display Functions
def display_advanced_charts(charts_data: Dict[str, Any]):
    """View advanced charts"""
    if not charts_data:
        st.info("📊 No charts available")
        return

    st.subheader("📊 Advanced Data Visualization Dashboard")

    # Chart categories
    chart_categories = {
        '🗂️ Dataset Overview': [k for k in charts_data.keys() if
                                not k.startswith('column_') and k not in
                                ['correlation_analysis', 'missing_values_analysis', 'data_quality_dashboard']],
        '📋 Column Analysis': [k for k in charts_data.keys() if k.startswith('column_')],
        '🔗 Advanced Analysis': [k for k in charts_data.keys() if
                                k in ['correlation_analysis', 'missing_values_analysis', 'data_quality_dashboard']]
    }

    # Remove empty categories
    chart_categories = {k: v for k, v in chart_categories.items() if v}

    if not chart_categories:
        st.warning("No chart categories found")
        return

    # Enhanced navigation with tabs
    tab1, tab2, tab3 = st.tabs(list(chart_categories.keys()))
    tabs = [tab1, tab2, tab3]

    for i, (category_name, available_charts) in enumerate(chart_categories.items()):
        if i < len(tabs):
            with tabs[i]:
                display_chart_category(category_name, available_charts, charts_data)


def display_chart_category(category_name: str, available_charts: List[str], charts_data: Dict[str, Any]):
    """View Chart category"""
    if not available_charts:
        st.info(f"No charts available in {category_name}")
        return

    if category_name == '📋 Column Analysis':
        # Special layout for column analysis
        display_column_analysis_grid(available_charts, charts_data)
    else:
        # Standard layout for other categories
        display_standard_chart_category(available_charts, charts_data)


def display_column_analysis_grid(available_charts: List[str], charts_data: Dict[str, Any]):
    """Grid layout for column analysis"""

    # Column selector
    column_names = [k.replace('column_', '') for k in available_charts]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📋 Select Column")
        selected_column = st.selectbox(
            "Choose column to analyze:",
            column_names,
            key="column_selector"
        )

        if selected_column:
            column_key = f'column_{selected_column}'
            if column_key in charts_data:
                display_column_info_detailed(selected_column, charts_data[column_key])

    with col2:
        if selected_column:
            column_key = f'column_{selected_column}'
            if column_key in charts_data:
                display_column_charts(selected_column, charts_data[column_key])


def display_column_info_detailed(column_name: str, chart_group_data: Dict[str, Any]):
    """Show detailed column information"""

    st.markdown(f"#### 📊 Column: `{column_name}`")

    column_info = chart_group_data.get('column_info', {})
    object_analysis = column_info.get('object_analysis')

    # Basic info
    col_a, col_b = st.columns(2)
    with col_a:
        dtype = column_info.get('dtype', 'Unknown')
        st.metric("Data Type", dtype)

        unique_count = column_info.get('unique_count', 0)
        st.metric("Unique Values", f"{unique_count:,}")

    with col_b:
        null_pct = column_info.get('null_percentage', 0)
        st.metric("Missing %", f"{null_pct:.1f}%")

        unique_pct = column_info.get('unique_percentage', 0)
        st.metric("Uniqueness", f"{unique_pct:.1f}%")

    # 🆕 Object analysis info
    if object_analysis:
        st.markdown("### 🔍 Object Analysis")

        suggested_type = object_analysis.get('suggested_type', 'unknown')
        st.success(f"**Detected Type:** {suggested_type.replace('_', ' ').title()}")

        # Pattern detection
        patterns = object_analysis.get('patterns', {})
        significant_patterns = [k for k, v in patterns.items() if v.get('percentage', 0) > 10]

        if significant_patterns:
            st.markdown("**🎯 Detected Patterns:**")
            for pattern in significant_patterns:
                pct = patterns[pattern]['percentage']
                st.markdown(f'<span class="pattern-badge">{pattern}: {pct:.1f}%</span>',
                            unsafe_allow_html=True)

        # Length stats
        length_stats = object_analysis.get('length_stats', {})
        if length_stats:
            st.markdown("**📏 Text Length:**")
            st.write(f"Min: {length_stats.get('min', 0)}, "
                     f"Max: {length_stats.get('max', 0)}, "
                     f"Avg: {length_stats.get('mean', 0):.1f}")

    # Type indicators
    type_indicators = []
    if column_info.get('is_numeric'): type_indicators.append("🔢 Numeric")
    if column_info.get('is_categorical'): type_indicators.append("🏷️ Categorical")
    if column_info.get('is_datetime'): type_indicators.append("📅 DateTime")
    if column_info.get('is_text'): type_indicators.append("📝 Text")

    if type_indicators:
        st.markdown("**🎯 Type Indicators:**")
        for indicator in type_indicators:
            st.markdown(f"- {indicator}")


def display_column_charts(column_name: str, chart_group_data: Dict[str, Any]):
    """Show bar charts"""

    charts = chart_group_data.get('charts', {})
    if not charts:
        st.warning("No charts available for this column")
        return

    st.markdown(f"### 📈 Charts for `{column_name}`")

    # Chart selector
    chart_names = list(charts.keys())
    chart_names = [name for name in chart_names if name != 'error']  # Filter out error charts

    if not chart_names:
        st.info("No valid charts to display")
        return

    selected_chart = st.selectbox(
        "Select Chart:",
        chart_names,
        format_func=lambda x: format_individual_chart_name(x),
        key=f"chart_selector_{column_name}"
    )

    if selected_chart and selected_chart in charts:
        chart_data = charts[selected_chart]

        # Display chart with enhanced styling
        with st.container():
            st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)

            display_single_chart_enhanced(selected_chart, chart_data)

            st.markdown('</div>', unsafe_allow_html=True)


def display_single_chart_enhanced(chart_name: str, chart_data: Dict[str, Any]):
    """Enhanced single chart display"""
    if not chart_data:
        st.warning(f"No data available for chart: {chart_name}")
        return

    # Chart title with icon
    chart_title = format_individual_chart_name(chart_name)
    st.markdown(f"#### {chart_title}")

    # Chart config
    if 'config' in chart_data and chart_data['config']:
        try:
            config = chart_data['config']
            if isinstance(config, str):
                config = json.loads(config)

            if isinstance(config, dict) and 'data' in config and 'layout' in config:
                # Create Plotly figure
                fig = go.Figure(data=config['data'], layout=config['layout'])

                # Enhanced layout for Streamlit
                fig.update_layout(
                    height=450,
                    margin=dict(l=10, r=10, t=50, b=10),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12)
                )

                # Display chart
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                })

            else:
                st.error("Invalid chart configuration format")
                with st.expander("🔍 Debug - Chart Config"):
                    st.json(config)

        except Exception as e:
            st.error(f"Error displaying chart: {str(e)}")
            with st.expander("🔍 Debug - Raw Chart Data"):
                st.json(chart_data)

    # Enhanced insights display
    if 'insights' in chart_data and chart_data['insights']:
        st.markdown("#### 💡 Chart Insights")
        insights_container = st.container()
        with insights_container:
            for i, insight in enumerate(chart_data['insights']):
                if insight:
                    # Different styling for different insight types
                    if 'pattern' in insight.lower() or 'detected' in insight.lower():
                        st.info(f"🎯 {insight}")
                    elif 'sample' in insight.lower():
                        st.warning(f"📊 {insight}")
                    else:
                        st.success(f"✨ {insight}")

    # Enhanced statistics display
    if 'stats' in chart_data and chart_data['stats']:
        st.markdown("#### 📊 Statistics")
        stats = chart_data['stats']

        # Create columns for stats
        num_cols = min(len(stats), 4)
        cols = st.columns(num_cols)

        for i, (stat_name, stat_value) in enumerate(stats.items()):
            with cols[i % num_cols]:
                formatted_value = format_stat_value(stat_value)
                st.metric(
                    label=stat_name.replace('_', ' ').title(),
                    value=formatted_value
                )


def format_individual_chart_name(chart_name: str) -> str:
    """Enhanced chart name formatting"""
    name_mappings = {
        'histogram': '📊 Histogram',
        'boxplot': '📦 Box Plot',
        'statistics': '📈 Statistics',
        'density': '🌊 Density Plot',
        'bar_chart': '📊 Bar Chart',
        'pie_chart': '🥧 Pie Chart',
        'frequency_curve': '📈 Frequency Curve',
        'time_series': '📅 Time Series',
        'missing_values': '⚠️ Missing Values',
        'correlation_heatmap': '🔥 Correlation Heatmap',
        'length_distribution': '📏 Text Length',
        'word_count': '📝 Word Count',
        'character_frequency': '🔤 Character Frequency',
        'column_types': '🏷️ Column Types',
        'dataset_shape': '📐 Dataset Shape',
        'quality_metrics': '✅ Quality Metrics',
        # 🆕 New object chart types
        'data_type_distribution': '🎯 Data Type Analysis',
        'pattern_analysis': '🔍 Pattern Detection',
        'value_frequency': '📊 Value Frequency',
        'numeric_converted': '🔢 Numeric Conversion',
        'date_converted': '📅 Date Conversion',
        'value_distribution': '📊 Value Distribution'
    }

    return name_mappings.get(chart_name, chart_name.replace('_', ' ').title())


def display_standard_chart_category(available_charts: List[str], charts_data: Dict[str, Any]):
    """Standard chart category display"""
    selected_chart_group = st.selectbox(
        "Select Analysis:",
        available_charts,
        format_func=lambda x: format_chart_name(x),
        key=f"standard_chart_selector_{len(available_charts)}"
    )

    if selected_chart_group and selected_chart_group in charts_data:
        chart_group_data = charts_data[selected_chart_group]

        col1, col2 = st.columns([3, 1])

        with col1:
            display_chart_group(selected_chart_group, chart_group_data)

        with col2:
            display_chart_info_panel(selected_chart_group, chart_group_data)


def format_chart_name(chart_key: str) -> str:
    """Convert chart name to user-friendly format"""
    if chart_key.startswith('column_'):
        column_name = chart_key.replace('column_', '')
        return f"📋 {column_name}"

    name_mappings = {
        'dataset_overview': '🗂️ Dataset Overview',
        'correlation_analysis': '🔗 Correlation Analysis',
        'missing_values_analysis': '⚠️ Missing Values Analysis',
        'data_quality_dashboard': '✅ Data Quality Dashboard'
    }

    return name_mappings.get(chart_key, chart_key.replace('_', ' ').title())


def display_chart_group(chart_group_name: str, chart_group_data: Dict[str, Any]):
    """View chart group"""
    if not isinstance(chart_group_data, dict):
        st.error("Invalid chart group data format")
        return

    # Group insights with enhanced styling
    if 'insights' in chart_group_data and chart_group_data['insights']:
        with st.expander("💡 Key Insights", expanded=True):
            for insight in chart_group_data['insights']:
                if insight:
                    st.markdown(f'<div class="insight-box">💡 {insight}</div>',
                                unsafe_allow_html=True)

    # Group charts
    if 'charts' in chart_group_data:
        charts = chart_group_data['charts']

        if not charts:
            st.warning("No charts available in this group")
            return

        chart_names = list(charts.keys())
        if len(chart_names) == 1:
            display_single_chart_enhanced(chart_names[0], charts[chart_names[0]])
        else:
            selected_chart = st.selectbox(
                "Select Chart:",
                chart_names,
                format_func=lambda x: format_individual_chart_name(x),
                key=f"chart_selector_{chart_group_name}"
            )

            if selected_chart:
                display_single_chart_enhanced(selected_chart, charts[selected_chart])


def display_chart_info_panel(chart_group_name: str, chart_group_data: Dict[str, Any]):
    """Enhanced chart information panel"""
    with st.container():
        st.markdown("### 📊 Chart Info")

        # Chart group type
        if 'type' in chart_group_data:
            chart_type = chart_group_data['type'].replace('_', ' ').title()
            st.info(f"**Type:** {chart_type}")

        # Chart count
        if 'charts' in chart_group_data:
            chart_count = len(chart_group_data['charts'])
            st.metric("Available Charts", chart_count)

        # column specific info
        if chart_group_name.startswith('column_'):
            column_name = chart_group_name.replace('column_', '')
            st.success(f"**Column:** `{column_name}`")

            if 'column_info' in chart_group_data:
                column_info = chart_group_data['column_info']

                # Quick stats
                st.markdown("**📊 Quick Stats:**")
                col_a, col_b = st.columns(2)

                with col_a:
                    unique_count = column_info.get('unique_count', 0)
                    st.metric("Unique", f"{unique_count:,}")

                with col_b:
                    null_pct = column_info.get('null_percentage', 0)
                    if null_pct == 0:
                        st.success("✅ Complete")
                    elif null_pct < 5:
                        st.info(f"ℹ️ {null_pct:.1f}% missing")
                    else:
                        st.warning(f"⚠️ {null_pct:.1f}% missing")


def format_stat_value(value):
    """Format statistics value"""
    if isinstance(value, (int, float)):
        if abs(value) >= 1000000:
            return f"{value / 1000000:.1f}M"
        elif abs(value) >= 1000:
            return f"{value / 1000:.1f}K"
        elif isinstance(value, float):
            return f"{value:.2f}"
        else:
            return f"{value:,}"
    else:
        return str(value)


def create_chart_summary_dashboard(charts_data: Dict[str, Any]):
    """Enhanced chart summary dashboard"""
    st.subheader("📊 Chart Analysis Summary")

    # Calculate enhanced summary statistics
    total_chart_groups = len(charts_data)
    total_individual_charts = 0
    column_analyses = 0
    overview_charts = 0
    object_analyses = 0

    for key, value in charts_data.items():
        if isinstance(value, dict) and 'charts' in value:
            total_individual_charts += len(value['charts'])

            if key.startswith('column_'):
                column_analyses += 1
                # Check if this is an object column with pattern analysis
                column_info = value.get('column_info', {})
                if column_info.get('object_analysis'):
                    object_analyses += 1
            else:
                overview_charts += 1

    # Display  metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Chart Groups", total_chart_groups)

    with col2:
        st.metric("Individual Charts", total_individual_charts)

    with col3:
        st.metric("Column Analyses", column_analyses)

    with col4:
        st.metric("Object Analyses", object_analyses, delta=f"+{object_analyses} smart")

    # Enhanced chart type breakdown
    chart_types = {}
    object_chart_types = {}

    for chart_group in charts_data.values():
        if isinstance(chart_group, dict) and 'charts' in chart_group:
            for chart_name in chart_group['charts'].keys():
                chart_types[chart_name] = chart_types.get(chart_name, 0) + 1

                # Track object-specific chart types
                if chart_name in ['data_type_distribution', 'pattern_analysis', 'numeric_converted', 'date_converted']:
                    object_chart_types[chart_name] = object_chart_types.get(chart_name, 0) + 1

    if chart_types:
        st.markdown("### 📈 Chart Type Distribution")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 All Chart Types**")
            sorted_types = sorted(chart_types.items(), key=lambda x: x[1], reverse=True)[:6]

            for chart_type, count in sorted_types:
                st.metric(format_individual_chart_name(chart_type), count)

        with col2:
            if object_chart_types:
                st.markdown("**🎯 Object Analysis Charts**")
                for chart_type, count in object_chart_types.items():
                    st.metric(format_individual_chart_name(chart_type), count, delta="Smart")


# Rest of the functions remain the same...
def display_analysis_results_with_advanced_charts(result):
    """Show analysis results with advanced charts"""

    # Debug info with enhanced toggle
    debug_enabled = st.sidebar.checkbox("🔍 Debug Mode", help="Show technical information")

    if debug_enabled:
        with st.expander("🐛 Debug Information", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.json({
                    "data_source": result.get('data_source', 'direct'),
                    "chart_mode": result.get('chart_mode', 'unknown'),
                    "chart_count": result.get('chart_count', 0),
                    "has_charts": 'charts' in result and bool(result['charts']),
                    "sample_size": result.get('sample_size')
                })

            with col2:
                if 'charts' in result and result['charts']:
                    chart_keys = list(result['charts'].keys())[:5]
                    st.json({
                        "chart_keys_sample": chart_keys,
                        "total_keys": len(result['charts']),
                        "first_chart_structure": {
                            k: type(v).__name__ for k, v in
                            (result['charts'][chart_keys[0]].items() if chart_keys and isinstance(
                                result['charts'][chart_keys[0]], dict) else {})
                        } if chart_keys else "No charts"
                    })

    # Enhanced basic info metrics
    if 'analysis' in result and 'basic_info' in result['analysis']:
        basic_info = result['analysis']['basic_info']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            rows_value = safe_get_value(basic_info, 'rows', 0)
            st.metric(
                label="📊 Total Rows",
                value=safe_format_number(rows_value),
                help="Number of data rows in the dataset"
            )

        with col2:
            columns_value = safe_get_value(basic_info, 'columns', 0)
            st.metric(
                label="📋 Columns",
                value=str(columns_value),
                help="Number of columns/features"
            )

        with col3:
            duration = result.get('total_duration')
            duration_str = f"{duration:.2f}s" if duration else "N/A"
            st.metric(
                label="⏱️ Processing Time",
                value=duration_str,
                help="Time taken for analysis"
            )

        with col4:
            chart_mode = result.get('chart_mode', 'unknown')
            chart_count = result.get('chart_count', 0)
            data_source = result.get('data_source', 'direct')

            mode_emoji_and_text = {
                'detailed': ('🎯', 'Enhanced'),
                'detailed_sample': ('🎯📊', 'Enhanced Sample'),
                'detailed_loaded': ('🎯', 'Loaded'),
                'basic': ('📊', 'Basic'),
                'basic_sample': ('📊📋', 'Basic Sample'),
                'basic_loaded': ('📊📂', 'Basic Loaded'),
                'basic_fallback': ('📊⚠️', 'Basic Fallback'),
                'none': ('❌', 'None'),
                'failed': ('🚨', 'Failed'),
                'unknown': ('❓', 'Unknown'),
                'loaded': ('📂', 'Loaded')
            }

            emoji, display_text = mode_emoji_and_text.get(chart_mode, ('❓', chart_mode.title()))

            st.metric(
                label="📊 Chart Analysis",
                value=f"{emoji} {display_text} ({chart_count})",
                help=f"Chart mode: {display_text} • Count: {chart_count} • Source: {data_source}"
            )

    # Analysis Insights with enhanced display
    if 'analysis' in result and 'insights' in result['analysis']:
        insights = result['analysis']['insights']
        if insights and isinstance(insights, list):
            st.subheader("🔍 Analysis Insights")

            # Group insights by category
            general_insights = []
            data_quality_insights = []

            for insight in insights:
                if insight:
                    if any(word in insight.lower() for word in ['quality', 'missing', 'outlier']):
                        data_quality_insights.append(insight)
                    else:
                        general_insights.append(insight)

            col1, col2 = st.columns(2)

            with col1:
                if general_insights:
                    st.markdown("**📊 General Insights**")
                    for insight in general_insights:
                        st.info(f"💡 {insight}")

            with col2:
                if data_quality_insights:
                    st.markdown("**✅ Data Quality**")
                    for insight in data_quality_insights:
                        st.success(f"🎯 {insight}")

    # Charts Section
    if 'charts' in result and result['charts']:
        charts = result['charts']

        st.markdown("---")

        # Sample disclaimer for streaming
        if result.get('sample_size'):
            st.info(f"📊 Charts based on sample of {result['sample_size']:,} rows from total dataset")

        # Chart format detection
        is_advanced = False
        if isinstance(charts, dict):
            advanced_indicators = [
                any('column_' in k for k in charts.keys()),
                any(isinstance(v, dict) and 'charts' in v and 'type' in v for v in charts.values()),
                any(isinstance(v, dict) and 'insights' in v for v in charts.values())
            ]
            is_advanced = any(advanced_indicators)

        if is_advanced:
            st.success("🎯 Advanced charts detected - Enhanced analysis mode")
            create_chart_summary_dashboard(charts)
            display_advanced_charts(charts)
        else:
            st.info("📊 Basic charts detected - Standard mode")
            display_basic_charts_improved(charts)
    else:
        st.warning("📊 No charts available for this analysis")


def display_basic_charts_improved(charts: Dict[str, Any]):
    """Enhanced basic chart display"""
    st.subheader("📈 Interactive Charts")

    if not charts:
        st.info("No charts to display")
        return

    chart_names = list(charts.keys())
    tabs = st.tabs([format_individual_chart_name(name) for name in chart_names])

    for i, (chart_name, chart_data) in enumerate(charts.items()):
        with tabs[i]:
            if isinstance(chart_data, dict):
                with st.container():
                    st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
                    display_single_chart_enhanced(chart_name, chart_data)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"Invalid chart data format for {chart_name}")


# Header with enhanced styling
st.markdown('<h1 class="main-header">📊 Dataset Analyzer v3.1</h1>', unsafe_allow_html=True)
st.markdown("---")

# Enhanced sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")

    # Backend Status Check
    with st.container():
        st.subheader("🔗 Backend Status")
        if st.button("Check Status", type="secondary"):
            status = call_api("/")
            if status:
                st.success("✅ Backend is running!")
                if status.get('version'):
                    st.code(f"Version: {status['version']}")
                st.json(status)
            else:
                st.error("❌ Backend not responding")

    st.markdown("---")

    # Enhanced File Upload Section
    st.subheader("📁 Dataset Upload")

    uploaded_file = st.file_uploader(
        "Select your dataset",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Maximum file size: 500MB"
    )

    # Analysis Method Selection
    analysis_method = st.radio(
        "Analysis Method",
        ["Normal Analysis", "Streaming Analysis"],
        help="Streaming analysis is recommended for large files (>50MB) - Supports CSV, JSON, Excel"
    )

    # Enhanced Chart Options
    st.markdown("**📊 Visualization Options**")
    include_charts = st.checkbox("Generate Charts", value=True)

    if include_charts:
        chart_mode = st.radio(
            "Chart Detail Level:",
            ["Basic Charts", "Enhanced Analysis"],
            help="Enhanced analysis creates smart charts for every column including object pattern detection"
        )

        detailed_charts = (chart_mode == "Enhanced Analysis")

        if detailed_charts:
            st.info("🎯 Enhanced mode includes smart object analysis, pattern detection, and convertibility analysis")
        else:
            st.info("📊 Basic mode provides standard statistical charts")

        # Chart performance note for streaming
        if analysis_method == "Streaming Analysis" and include_charts:
            st.info("📊 Streaming charts are based on data samples • Supports CSV, JSON, Excel formats")

    else:
        detailed_charts = False

    # File info and upload
    if uploaded_file is not None:
        # Enhanced file info
        st.markdown("### 📄 File Information")
        file_info_container = st.container()

        with file_info_container:
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Filename",
                          uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
                st.metric("Size", format_file_size(uploaded_file.size))

            with col_b:
                st.metric("Type", uploaded_file.type)

                # Size recommendation
                size_mb = uploaded_file.size / (1024 * 1024)
                if size_mb > 50:
                    st.warning(f"⚠️ Large file ({size_mb:.1f}MB) - Consider streaming analysis")
                else:
                    st.success(f"✅ Good size ({size_mb:.1f}MB)")

        # Analysis button
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing your dataset..."):
                # Determine endpoint and parameters
                if analysis_method == "Streaming Analysis":
                    endpoint = "/streaming-analyze"
                    params = {
                        "chunk_size": 100000,
                        "include_charts": include_charts,
                        "detailed_charts": detailed_charts
                    }
                else:
                    endpoint = "/analyze"
                    params = {
                        "include_charts": include_charts,
                        "detailed_charts": detailed_charts
                    }

                # Prepare file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                # Make API call
                result = call_api(endpoint, method="POST", files=files, params=params)

                if result and result.get('success'):
                    st.session_state.analysis_results = result
                    st.success("✅ Analysis completed successfully!")
                    load_upload_history()
                    st.rerun()
                else:
                    st.error("❌ Analysis failed!")

    st.markdown("---")

    # Enhanced History Section
    st.subheader("📚 Analysis History")

    if st.button("🔄 Refresh History"):
        load_upload_history()

    # Load history on first run
    if not st.session_state.upload_history:
        load_upload_history()

    # Display enhanced history
    if st.session_state.upload_history:
        for upload in st.session_state.upload_history[:5]:
            with st.container():
                # Enhanced history card
                st.markdown(
                    f"**#{upload['id']}** `{upload['original_filename'][:25]}{'...' if len(upload['original_filename']) > 25 else ''}`")

                col1, col2 = st.columns(2)
                with col1:
                    rows = safe_get_value(upload, 'rows', 'N/A')
                    if rows != 'N/A':
                        st.caption(f"📊 {safe_format_number(rows)} rows")
                    else:
                        st.caption("⏳ Processing...")

                with col2:
                    st.caption(f"💾 {format_file_size(upload.get('file_size', 0))}")

                # Enhanced load button
                if st.button(f"📂 Load #{upload['id']}", key=f"load_{upload['id']}", use_container_width=True):
                    with st.spinner(f"📊 Loading analysis #{upload['id']}..."):
                        # Load previous analysis
                        analysis_data = call_api(f"/analysis/{upload['id']}")

                        if analysis_data:
                            # Try to get charts
                            charts_data = call_api(f"/charts/{upload['id']}")

                            # Parse charts
                            parsed_charts = None
                            chart_count = 0
                            chart_mode = "loaded"

                            if charts_data and 'charts' in charts_data:
                                parsed_charts = charts_data['charts']

                                # Enhanced chart format detection
                                if isinstance(parsed_charts, dict):
                                    is_advanced = any(
                                        isinstance(v, dict) and 'charts' in v and 'type' in v
                                        for v in parsed_charts.values()
                                    )

                                    if is_advanced:
                                        chart_mode = "detailed_loaded"
                                        chart_count = sum(
                                            len(v.get('charts', {}))
                                            for v in parsed_charts.values()
                                            if isinstance(v, dict) and 'charts' in v
                                        )
                                    else:
                                        chart_mode = "basic_loaded"
                                        chart_count = len(parsed_charts)
                                else:
                                    chart_mode = "loaded"
                                    chart_count = 0

                            # Reconstruct analysis result format
                            st.session_state.analysis_results = {
                                'success': True,
                                'upload_id': upload['id'],
                                'filename': analysis_data.get('original_filename', 'Unknown'),
                                'file_type': analysis_data.get('file_type', 'unknown'),
                                'total_duration': analysis_data.get('analysis_duration'),
                                'analysis': {
                                    'basic_info': {
                                        'rows': analysis_data.get('rows_count') or analysis_data.get('rows'),
                                        'columns': analysis_data.get('columns_count') or analysis_data.get('columns'),
                                        'memory_usage_mb': None
                                    },
                                    'insights': analysis_data.get('insights', [])
                                },
                                'charts': parsed_charts,
                                'chart_mode': chart_mode,
                                'chart_count': chart_count,
                                'data_source': 'history'
                            }

                            st.success(f"✅ Analysis #{upload['id']} loaded ({chart_mode}, {chart_count} charts)")
                            st.rerun()
                        else:
                            st.error(f"❌ Could not load analysis #{upload['id']}")

                st.markdown("---")

    else:
        st.info("No analysis history yet")

# Main Content Area
if st.session_state.analysis_results:
    result = st.session_state.analysis_results

    filename = result.get('filename', 'Unknown file')
    chart_mode = result.get('chart_mode', 'unknown')
    chart_count = result.get('chart_count', 0)

    # Success emoji and text mapping
    success_mapping = {
        'detailed': ('🎯', 'Enhanced Analysis'),
        'detailed_sample': ('🎯📊', 'Enhanced Sample Analysis'),
        'detailed_loaded': ('🎯📂', 'Enhanced Analysis (Loaded)'),  # ✅ FIX
        'basic': ('📊', 'Basic Analysis'),
        'basic_sample': ('📊📋', 'Basic Sample Analysis'),
        'basic_loaded': ('📊📂', 'Basic Analysis (Loaded)'),  # ✅ FIX
        'basic_fallback': ('📊⚠️', 'Basic Analysis (Fallback)'),
        'none': ('❌', 'No Charts'),
        'failed': ('🚨', 'Chart Generation Failed'),
        'loaded': ('📂', 'Analysis Loaded')  # ✅ FIX
    }

    emoji, mode_text = success_mapping.get(chart_mode, ('✅', chart_mode.replace('_', ' ').title()))

    st.success(f"{emoji} Analysis completed: **{filename}** • {mode_text} ({chart_count} charts)")

    # Display results
    display_analysis_results_with_advanced_charts(result)

    # Raw data preview
    with st.expander("🔍 Raw Analysis Data (for debugging)"):
        st.json(result)

else:
    # Enhanced Welcome Screen
    st.markdown("""
    ## 🎉 Welcome to Dataset Analyzer v3.1!

    ### 🚀 Get Started in 3 Steps:

    1. **📁 Upload** your dataset (CSV, Excel, or JSON)
    2. **⚙️ Choose** analysis method and visualization level
    3. **📊 Explore** automated insights and smart visualizations

    ### ✨ New Features in v3.1:

    - 🎯 **Smart Object Analysis**: Automatic pattern detection in text columns
    - 🔍 **Enhanced Pattern Recognition**: Email, phone, URL, and code detection  
    - 📊 **Convertibility Analysis**: Automatic detection of numeric/date strings
    - 🎨 **Improved Visualizations**: Better charts for all data types
    - 📱 **Enhanced UI**: Improved navigation and chart display

    ### 🎯 Chart Options:

    - **Basic Charts**: Standard statistical visualizations
    - **Enhanced Analysis**: Smart object analysis with pattern detection and convertibility analysis
    """)

    # Enhanced feature showcase
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="chart-container">
        <h4>🎯 Smart Analysis</h4>
        <p>Automatic pattern detection, data type suggestions, and convertibility analysis for object columns</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="chart-container">
        <h4>📊 Enhanced Charts</h4>
        <p>Pattern analysis, length distribution, convertibility charts, and smart frequency analysis</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="chart-container">
        <h4>⚡ High Performance</h4>
        <p>Streaming processing for large files with sample-based smart analysis</p>
        </div>
        """, unsafe_allow_html=True)

    # system status
    st.markdown("---")
    st.subheader("🔧 System Status")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Check Backend Status", use_container_width=True):
            status = call_api("/")
            if status:
                st.success("✅ Backend API is running!")
                version = safe_get_value(status, 'version', 'Unknown')
                st.code(f"Version: {version}")

                # Show available features
                features = status.get('features', [])
                if features:
                    st.write("**Available Features:**")
                    for feature in features:
                        st.write(f"• {feature}")
            else:
                st.error("❌ Backend API not responding")

    with col2:
        if st.button("Check System Memory", use_container_width=True):
            memory_info = call_api("/system/memory")
            if memory_info and 'memory' in memory_info:
                memory = memory_info['memory']
                usage = safe_get_value(memory, 'used_percentage', 0)

                if usage < 70:
                    st.success(f"💾 Memory Usage: {usage}% - Good")
                elif usage < 85:
                    st.warning(f"💾 Memory Usage: {usage}% - Moderate")
                else:
                    st.error(f"💾 Memory Usage: {usage}% - High")

                try:
                    st.progress(float(usage) / 100)
                except (ValueError, TypeError):
                    st.progress(0)
            else:
                st.error("❌ Could not fetch memory information")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px; background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px;'>
    <p><strong>🚀 Dataset Analyzer v3.1</strong> | Built with Streamlit & FastAPI</p>
    <p>💡 <em>Pro tip: Use Enhanced Analysis mode for smart object column analysis</em></p>
    <p>🎯 <em>New: Pattern detection, convertibility analysis, and enhanced visualizations</em></p>
</div>
""", unsafe_allow_html=True)
