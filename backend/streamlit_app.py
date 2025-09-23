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
    page_title="Dataset Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
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
if 'current_charts' not in st.session_state:
    st.session_state.current_charts = None


# Helper Functions
def safe_format_number(value, default=0):
    """Safely format a number, handling None values"""
    if value is None:
        return str(default)
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(default)


def safe_get_value(data, key, default="N/A"):
    """Safely get value from dict, handling None and missing keys"""
    if not data or not isinstance(data, dict):
        return default
    value = data.get(key)
    return value if value is not None else default


def call_api(endpoint, method="GET", files=None, params=None):
    """Generic API caller with error handling"""
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
        st.error(
            "🔴 Backend bağlantısı başarısız! FastAPI sunucusunun http://localhost:8000 adresinde çalıştığından emin olun")
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ İstek zaman aşımı! Büyük dosyalar daha uzun sürebilir.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Hatası: {str(e)}")
        return None


def load_upload_history():
    """Load upload history from backend"""
    history = call_api("/history", params={"limit": 20})
    if history:
        st.session_state.upload_history = history.get('uploads', [])


def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if not size_bytes or size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


# Header
st.markdown('<h1 class="main-header">📊 Dataset Analyzer</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🎛️ Kontrol Paneli")

    # Backend Status Check
    with st.container():
        st.subheader("🔗 Backend Durumu")
        if st.button("Durumu Kontrol Et", type="secondary"):
            status = call_api("/")
            if status:
                st.success("✅ Backend çalışıyor!")
                st.json(status)
            else:
                st.error("❌ Backend çalışmıyor")

    st.markdown("---")

    # File Upload Section
    st.subheader("📁 Dosya Yükleme")

    uploaded_file = st.file_uploader(
        "Veri setinizi seçin",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Maksimum dosya boyutu: 500MB"
    )

    # Analysis Method Selection
    analysis_method = st.radio(
        "Analiz Yöntemi",
        ["Normal Analiz", "Streaming Analiz"],
        help="Büyük dosyalar için streaming analiz önerilir (>50MB)"
    )

    include_charts = st.checkbox("Grafikler Oluştur", value=True)

    if uploaded_file is not None:
        # File info
        st.info(f"**Dosya:** {uploaded_file.name}")
        st.info(f"**Boyut:** {format_file_size(uploaded_file.size)}")
        st.info(f"**Tip:** {uploaded_file.type}")

        # Analysis button
        if st.button("🚀 Analizi Başlat", type="primary"):
            with st.spinner("🔄 Veri setiniz analiz ediliyor..."):
                # Determine endpoint based on method
                if analysis_method == "Streaming Analiz" and uploaded_file.name.lower().endswith('.csv'):
                    endpoint = "/streaming-analyze"
                    params = {"chunk_size": 100000}
                else:
                    endpoint = "/analyze"
                    params = {"include_charts": include_charts}

                # Prepare file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                # Make API call
                result = call_api(endpoint, method="POST", files=files, params=params)

                if result and result.get('success'):
                    st.session_state.analysis_results = result
                    st.session_state.current_charts = result.get('charts')
                    st.success("✅ Analiz başarıyla tamamlandı!")

                    # Refresh history
                    load_upload_history()

                    # Force rerun to update main content
                    st.rerun()
                else:
                    st.error("❌ Analiz başarısız!")

    st.markdown("---")

    # History Section
    st.subheader("📚 Yükleme Geçmişi")

    if st.button("🔄 Geçmişi Yenile"):
        load_upload_history()

    # Load history on first run
    if not st.session_state.upload_history:
        load_upload_history()

    # Display history
    if st.session_state.upload_history:
        for upload in st.session_state.upload_history[:5]:  # Show last 5
            with st.container():
                st.write(f"**#{upload['id']}** {upload['original_filename']}")
                col1, col2 = st.columns(2)
                with col1:
                    rows = safe_get_value(upload, 'rows', 'N/A')
                    if rows != 'N/A':
                        st.caption(f"Satır: {safe_format_number(rows)}")
                    else:
                        st.caption("İşleniyor...")
                with col2:
                    st.caption(f"Boyut: {format_file_size(upload.get('file_size', 0))}")

                if st.button(f"Yükle #{upload['id']}", key=f"load_{upload['id']}"):
                    # Load previous analysis
                    analysis_data = call_api(f"/analysis/{upload['id']}")
                    charts_data = call_api(f"/charts/{upload['id']}")

                    if analysis_data:
                        # Reconstruct analysis result format
                        st.session_state.analysis_results = {
                            'success': True,
                            'upload_id': upload['id'],
                            'filename': analysis_data.get('filename', 'Unknown'),
                            'analysis': {
                                'basic_info': {
                                    'rows': safe_get_value(analysis_data, 'rows'),
                                    'columns': safe_get_value(analysis_data, 'columns')
                                },
                                'insights': analysis_data.get('insights', [])
                            },
                            'total_duration': None
                        }

                        if charts_data and charts_data.get('charts'):
                            st.session_state.current_charts = charts_data['charts']

                        st.success(f"✅ #{upload['id']} numaralı analiz yüklendi")
                        st.rerun()

                st.markdown("---")
    else:
        st.info("Henüz yükleme geçmişi yok")

# Main Content Area
if st.session_state.analysis_results:
    result = st.session_state.analysis_results

    # Success message
    st.success(f"✅ Analiz tamamlandı: **{result.get('filename', 'Bilinmeyen dosya')}**")

    # Basic Info Metrics
    if 'analysis' in result and 'basic_info' in result['analysis']:
        basic_info = result['analysis']['basic_info']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            rows_value = safe_get_value(basic_info, 'rows', 0)
            st.metric(
                label="📊 Toplam Satır",
                value=safe_format_number(rows_value),
                help="Veri setindeki satır sayısı"
            )

        with col2:
            columns_value = safe_get_value(basic_info, 'columns', 0)
            st.metric(
                label="📋 Sütun Sayısı",
                value=str(columns_value),
                help="Sütun/özellik sayısı"
            )

        with col3:
            if 'memory_usage_mb' in basic_info and basic_info['memory_usage_mb']:
                st.metric(
                    label="💾 Bellek Kullanımı",
                    value=f"{basic_info['memory_usage_mb']} MB",
                    help="Veri seti tarafından kullanılan bellek"
                )
            else:
                duration = result.get('total_duration')
                duration_str = f"{duration:.2f}s" if duration else "N/A"
                st.metric(
                    label="⏱️ İşlem Süresi",
                    value=duration_str,
                    help="Analiz için geçen süre"
                )

        with col4:
            st.metric(
                label="🎯 Upload ID",
                value=f"#{result.get('upload_id', 'N/A')}",
                help="Bu analiz için benzersiz kimlik"
            )

    # Analysis Insights
    if 'analysis' in result and 'insights' in result['analysis']:
        insights = result['analysis']['insights']
        if insights and isinstance(insights, list):
            st.subheader("🔍 Analiz İçgörüleri")
            for insight in insights:
                if insight:  # Check if insight is not empty
                    st.info(f"💡 {insight}")

    # Charts Section
    if st.session_state.current_charts:
        st.subheader("📈 İnteraktif Görselleştirmeler")

        charts = st.session_state.current_charts
        chart_names = list(charts.keys())

        if chart_names:
            selected_chart = st.selectbox(
                "Grafik Seçin:",
                chart_names,
                format_func=lambda x: x.replace('_', ' ').title()
            )

            if selected_chart and selected_chart in charts:
                chart_data = charts[selected_chart]

                st.subheader(f"{selected_chart.replace('_', ' ').title()}")

                # Display chart if config exists
                if 'config' in chart_data and chart_data['config']:
                    try:
                        # Handle different config formats
                        config = chart_data['config']
                        if isinstance(config, str):
                            config = json.loads(config)

                        # Create Plotly figure
                        if isinstance(config, dict) and 'data' in config and 'layout' in config:
                            fig = go.Figure(data=config['data'], layout=config['layout'])

                            # Update layout for Streamlit
                            fig.update_layout(
                                height=500,
                                margin=dict(l=0, r=0, t=50, b=0)
                            )

                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Grafik konfigürasyonu tanınmadı")

                    except Exception as e:
                        st.error(f"Grafik görüntüleme hatası: {str(e)}")
                        with st.expander("Ham grafik verisi (debug için)"):
                            st.json(chart_data)

                # Chart insights
                if 'insights' in chart_data and chart_data['insights']:
                    st.subheader("📋 Grafik İçgörüleri")
                    for insight in chart_data['insights']:
                        if insight:
                            st.success(f"✨ {insight}")

                # Chart statistics
                if 'stats' in chart_data and chart_data['stats']:
                    st.subheader("📊 İstatistikler")
                    stats_cols = st.columns(min(len(chart_data['stats']), 4))
                    for j, (stat_name, stat_value) in enumerate(chart_data['stats'].items()):
                        if j < len(stats_cols):
                            with stats_cols[j]:
                                formatted_value = safe_format_number(stat_value) if isinstance(stat_value,
                                                                                               (int, float)) else str(
                                    stat_value)
                                st.metric(
                                    label=stat_name.replace('_', ' ').title(),
                                    value=formatted_value
                                )

    # Raw Data Preview (if available)
    with st.expander("🔍 Ham Analiz Verisi (debug için)"):
        st.json(result)

else:
    # Welcome Screen
    st.markdown("""
    ## 🎉 Dataset Analyzer'a Hoş Geldiniz!

    ### 🚀 3 kolay adımda başlayın:

    1. **📁 Yükleyin** veri setinizi (CSV, Excel, veya JSON)
    2. **⚙️ Seçin** analiz yöntemini (Normal veya Streaming)
    3. **📊 Görüntüleyin** otomatik içgörüleri ve interaktif grafikleri

    ### ✨ Özellikler:

    - 🤖 **Otomatik Analiz**: Verileriniz hakkında anında içgörüler
    - 📈 **İnteraktif Grafikler**: Plotly destekli görselleştirmeler  
    - 🔄 **Büyük Dosya Desteği**: Büyük veri setleri için streaming analiz
    - 📚 **Geçmiş Takibi**: Önceki analizlere erişim
    - 💾 **Çoklu Format**: CSV, Excel (.xlsx, .xls), JSON

    ### 🎯 Şunlar için mükemmel:

    - Veri keşfi ve profilleme
    - Hızlı istatistiksel özetler  
    - Veri kalitesi değerlendirmesi
    - Görselleştirme oluşturma
    - Büyük veri seti işleme
    """)

    # Feature showcase
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🔍 **Akıllı Analiz**\n\nOtomatik veri tipi tespiti, istatistiksel özetler ve kalite içgörüleri")

    with col2:
        st.info(
            "📊 **Zengin Görselleştirmeler**\n\nHistogramlar, korelasyonlar, kutu grafikleri ve kategorik dağılımlar")

    with col3:
        st.info("⚡ **Yüksek Performans**\n\nBellek optimizasyonu ile büyük dosyalar için streaming işleme")

    # System status
    st.markdown("---")
    st.subheader("🔧 Sistem Durumu")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Backend Durumunu Kontrol Et"):
            status = call_api("/")
            if status:
                st.success("✅ Backend API çalışıyor!")
                version = safe_get_value(status, 'version', 'Bilinmiyor')
                st.code(f"Versiyon: {version}", language="text")
            else:
                st.error("❌ Backend API yanıt vermiyor")

    with col2:
        if st.button("Sistem Belleğini Kontrol Et"):
            memory_info = call_api("/system/memory")
            if memory_info and 'memory' in memory_info:
                memory = memory_info['memory']
                usage = safe_get_value(memory, 'used_percentage', 0)
                st.success(f"💾 Bellek Kullanımı: {usage}%")
                try:
                    st.progress(float(usage) / 100)
                except (ValueError, TypeError):
                    st.progress(0)
            else:
                st.error("❌ Bellek bilgisi alınamadı")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 Dataset Analyzer v3.0 | Streamlit & FastAPI ile yapıldı</p>
    <p>💡 İpucu: 50MB'dan büyük dosyalar için streaming analiz kullanın</p>
</div>
""", unsafe_allow_html=True)
