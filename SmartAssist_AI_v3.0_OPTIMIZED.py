# 🎯 SmartAssist AI v3.0 - OPTIMIZED EXPERT SYSTEM
# Integrated ML + Fuzzy Logic + NLP in a Side Panel UI
# Refined: Geo Map, Customer Types, Deep Dark Mode, Blue Buttons

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import re
import warnings

warnings.filterwarnings('ignore')

def format_currency(value):
    return f"Rp {value/1_000_000:,.1f} JT"

# ═══════════════════════════════════════════════════════════════
# 🎨 PAGE CONFIG & THEME
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SmartAssist AI: Intelligent Sales Analytics and Decision Support Platform for Empowering UMKM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "show_expert_panel" not in st.session_state:
    st.session_state.show_expert_panel = False 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "Halo! Saya Assistant AI Keratonian. Tanyakan tentang 'produk terlaris', 'analisis stok', atau 'strategi marketing'."}
    ]

# THEME DEFINITIONS
THEMES = {
    "dark": {
        "bg": "#0E1117",
        "card": "#161B22",
        "text": "#E6EDF3",
        "accent": "#58A6FF",
        "secondary": "#238636",
        "border": "rgba(48, 54, 61, 0.7)",
        "plot_bg": "#0E1117" 
    },
    "light": {
        "bg": "#FFFFFF",
        "card": "#F8F9FA",
        "text": "#24292F",
        "accent": "#0969DA",
        "secondary": "#1A7F37",
        "border": "rgba(208, 215, 222, 0.7)",
        "plot_bg": "#FFFFFF"
    }
}

current_theme = THEMES[st.session_state.theme]

# CSS FOR PREMIUM UI & DARK MODE FIXES
st.markdown(f"""
<style>
    /* GLOBAL RESET & DARK MODE FIXES */
    .stApp, .stApp > header, .stApp > div {{
        background-color: {current_theme['bg']} !important;
        color: {current_theme['text']} !important;
    }}
    
    /* REMOVE WHITE PATCHES IN DARK MODE */
    div[data-testid="stSidebar"], div[data-testid="stHeader"] {{
        background-color: {current_theme['bg']};
    }}
    
    /* UNIVERSAL TEXT COLOR FIX */
    p, h1, h2, h3, h4, span, div, label, .stMarkdown {{
        color: {current_theme['text']} !important;
    }}
    
    /* CUSTOM CARD STYLING */
    .css-card {{
        background-color: {current_theme['card']};
        border: 1px solid {current_theme['border']};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        color: {current_theme['text']};
    }}
    
    /* EXPERT AI BUTTON - FORCE BLUE */
    div.stButton > button[kind="primary"] {{
        background-color: #007BFF !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }}
    div.stButton > button[kind="primary"] * {{
        color: white !important;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: #0056b3 !important;
    }}

    /* CHAT BUBBLES */
    .chat-user {{
        background: #007BFF;
        color: white !important;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 5px 0;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
    }}
    
    .chat-ai {{
        background: {current_theme['card']};
        border: 1px solid {current_theme['border']};
        color: {current_theme['text']} !important;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 5px 0;
        text-align: left;
        max-width: 90%;
    }}
    
    [data-testid="stMetricValue"] {{
        color: {current_theme['accent']} !important;
    }}
    
    /* TABLE HEADER FIX */
    [data-testid="stDataFrame"] th {{
        background-color: {current_theme['card']} !important;
        color: {current_theme['text']} !important;
    }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 🧠 INTELLIGENT BACKEND
# ═══════════════════════════════════════════════════════════════

class ExpertSystem:
    def __init__(self, df):
        self.df = df
        # Normalize Column Names
        self.df.columns = self.df.columns.str.strip()
        self.vectorizer = TfidfVectorizer()
        # Knowledge Base
        self.intents = {
            "bestsellers": ["produk terlaris", "paling laku", "best seller", "top produk"],
            "performance": ["performa", "penjualan", "omset", "revenue", "pendapatan"],
            "stock": ["stok", "inventory", "habis", "sisa", "slow moving", "dead stock"],
            "marketing": ["marketing", "promosi", "strategi", "iklan"],
            "customers": ["customer", "pelanggan", "pembeli", "konsumen"]
        }
    
    def fuzzy_decision_engine(self, row):
        sales_score = row['norm_sales']
        if sales_score > 0.8:
            return "🔥 STAR PRODUCT (Push Ads)", "high"
        elif sales_score > 0.4:
            return "✅ STABLE (Maintain)", "medium"
        else:
            return "⚠️ SLOW MOVING (Diskon/Bundle)", "critical"

    def generate_detailed_insights(self):
        insights = []
        total_sales = self.df['Total Penjualan'].sum()
        insights.append({"type": "info", "text": f"Revenue: **Rp {total_sales:,.0f}**"})
        
        product_stats = self.df.groupby('Pesanan').agg({
            'Total Penjualan': 'sum',
            'Qty': 'sum'
        }).reset_index()
        
        if not product_stats.empty:
            scaler = StandardScaler()
            if len(product_stats) > 1:
                sales_scaled = scaler.fit_transform(product_stats[['Total Penjualan']])
                product_stats['norm_sales'] = 1 / (1 + np.exp(-sales_scaled))
            else:
                product_stats['norm_sales'] = 0.5
            
            product_stats['Recommendation'], product_stats['Priority'] = zip(*product_stats.apply(self.fuzzy_decision_engine, axis=1))

            slow_movers = product_stats[product_stats['Priority'] == 'critical']
            if not slow_movers.empty:
                count = len(slow_movers)
                top_slow = slow_movers.sort_values('Qty').head(1)['Pesanan'].values[0]
                insights.append({
                    "type": "warning",
                    "text": f"⚠️ **{count} Produk Slow Moving** (e.g., {top_slow}). Buat promo bundle!"
                })
        else:
            product_stats = pd.DataFrame()

        return insights, product_stats

    def nlp_processor(self, user_query):
        user_query = user_query.lower()
        detected_intent = "unknown"
        for intent, keywords in self.intents.items():
            for kw in keywords:
                if kw in user_query:
                    detected_intent = intent
                    break
        
        if detected_intent == "bestsellers":
            top_5 = self.df.groupby('Pesanan')['Total Penjualan'].sum().nlargest(5)
            return "Top 5 Produk:\n" + "\n".join([f"- {i}. {idx}" for i, (idx, val) in enumerate(top_5.items(), 1)])
        elif detected_intent == "stock":
            return "Analisis Fuzzy Logic menunjukkan beberapa item 'Slow Moving'. Cek panel Stock Recs."
        else:
            return "Topik tidak dikenali. Coba 'produk terlaris' atau 'stok'."

# ═══════════════════════════════════════════════════════════════
# 📊 UTILITY ANALYZERS
# ═══════════════════════════════════════════════════════════════

class SalesForecaster:
    def __init__(self, df):
        self.df = df.copy()
        if 'Tanggal Order' in self.df.columns:
            self.df['Tanggal Order'] = pd.to_datetime(self.df['Tanggal Order'], errors='coerce')

    def predict_until_date(self, target_date="2025-12-31"):
        if 'Tanggal Order' not in self.df.columns: return None
        monthly = self.df.set_index('Tanggal Order').resample('M')['Total Penjualan'].sum()
        if len(monthly) < 2: return None
        
        last_date = monthly.index[-1]
        target_dt = pd.to_datetime(target_date)
        delta = (target_dt.year - last_date.year) * 12 + (target_dt.month - last_date.month)
        if delta <= 0: delta = 12 
        
        X = np.arange(len(monthly)).reshape(-1, 1)
        y = monthly.values
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(monthly), len(monthly) + delta).reshape(-1, 1)
        forecast = model.predict(future_X)
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, delta + 1)]
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': forecast, 'Type': 'Prediksi'})
        history_df = pd.DataFrame({'Date': monthly.index, 'Predicted Sales': monthly.values, 'Type': 'Riwayat'})
        return pd.concat([history_df, forecast_df])

# ═══════════════════════════════════════════════════════════════
# 🌍 GEOGRAPHY UTILS
# ═══════════════════════════════════════════════════════════════
INDONESIA_CITIES = {
    "Jakarta": {"lat": -6.2088, "lon": 106.8456},
    "Bandung": {"lat": -6.9175, "lon": 107.6191},
    "Surabaya": {"lat": -7.2575, "lon": 112.7521},
    "Semarang": {"lat": -6.9667, "lon": 110.4167},
    "Yogyakarta": {"lat": -7.7956, "lon": 110.3695},
    "Medan": {"lat": 3.5952, "lon": 98.6722},
    "Makassar": {"lat": -5.1477, "lon": 119.4328},
    "Bali": {"lat": -8.4095, "lon": 115.1889},
    "Denpasar": {"lat": -8.6705, "lon": 115.2126},
    "Malang": {"lat": -7.9666, "lon": 112.6326},
    "Solo": {"lat": -7.5755, "lon": 110.8243},
    "Surakarta": {"lat": -7.5755, "lon": 110.8243},
    "Bekasi": {"lat": -6.2383, "lon": 106.9756},
    "Tangerang": {"lat": -6.1730, "lon": 106.6358},
    "Depok": {"lat": -6.4025, "lon": 106.7942},
    "Bogor": {"lat": -6.5971, "lon": 106.8060}
}

def get_coordinates(city_name):
    """Simple exact match lookup for coordinates"""
    if not isinstance(city_name, str): return None, None
    for city, coords in INDONESIA_CITIES.items():
        if city.lower() in city_name.lower():
            return coords['lat'], coords['lon']
    return None, None

# ═══════════════════════════════════════════════════════════════
# 📱 MAIN APPLICATION UI
# ═══════════════════════════════════════════════════════════════

# DATA LOADING (Moved up for Dynamic Header)
uploaded_file = st.sidebar.file_uploader("📂 Unggah Data", type=['xlsx', 'csv'])

# HEADER
c1, c2 = st.columns([7, 3]) 
with c1:
    if uploaded_file:
        st.title("🎯 SmartAssist AI: Intelligent Sales Analytics and Decision Support Platform for Empowering UMKM")
        st.caption("Dashboard Analisis Terintegrasi")
    else:
        st.title("HOME")

with c2:
    # GROUPED BUTTONS
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🌓 Theme", use_container_width=True):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
    with btn_col2:
        if st.button("🤖 Expert AI", type="primary", use_container_width=True):
            st.session_state.show_expert_panel = not st.session_state.show_expert_panel
            st.rerun()

# LAYOUT
if st.session_state.show_expert_panel:
    main_col, side_col = st.columns([3, 1.2]) 
else:
    main_col = st.container()

# DATA LOADING
# uploaded_file already defined above

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Global Cleanup
    df.columns = df.columns.str.strip()
    
    expert = ExpertSystem(df)
    forecaster = SalesForecaster(df)
    
    # --------------------------
    # LEFT: MAIN DASHBOARD TABS
    # --------------------------
    with main_col:
        # ✅ UPDATED: Removed 'Marketing' Tab
        tab_titles = [
            "📊 Overview", "🏆 Best Seller", "⏰ Prime Time", "👥 Customers", 
            "🌍 Geography (Map)", "💰 Prediction", "📦 Stock Recomendation"
        ]
        tabs = st.tabs(tab_titles)
        
        # TAB 1: OVERVIEW
        with tabs[0]:
            st.markdown(f'<div class="css-card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Pendapatan", format_currency(df['Total Penjualan'].sum()))
            c2.metric("Total Pesanan", len(df))
            c3.metric("Total Produk", df['Pesanan'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)
            
            if 'Tanggal Order' in df.columns:
                 df['Tanggal Order'] = pd.to_datetime(df['Tanggal Order'], errors='coerce')
                 daily = df.groupby('Tanggal Order')['Total Penjualan'].sum()
                 st.area_chart(daily, color=current_theme['accent'])

        # TAB 2: BEST SELLERS
        with tabs[1]:
            st.subheader("Analisis Produk Unggulan")
            # ✅ UPDATED: Added 'Hampers' & 'Tatakan Dupa' Filters
            # Toggle between Product View and Category View
            view_mode = st.radio("Mode Tampilan:", ["Detail Produk (per Kategori)", "Peringkat Kategori Global"], horizontal=True, key="bs_view_mode")

            if view_mode == "Detail Produk (per Kategori)":
                # ✅ EXISTING: Dynamic Filtering based on 'Kategori' Column
                cat_options = ["Semua Item"]
                if 'Kategori' in df.columns:
                    unique_cats = sorted(df['Kategori'].dropna().unique().tolist())
                    cat_options += unique_cats
                
                filter_mode = st.selectbox("Pilih Kategori:", cat_options, key="bs_filter_dynamic")
                
                df_display = df.copy()
                if filter_mode != "Semua Item":
                    if 'Kategori' in df.columns:
                        df_display = df[df['Kategori'] == filter_mode]
                    else:
                        st.warning("Kolom 'Kategori' tidak ditemukan dalam dataset!")
                
                if len(df_display) > 0:
                    bs = df_display.groupby('Pesanan').agg({'Qty':'sum','Total Penjualan':'sum'}).reset_index().sort_values('Total Penjualan', ascending=False).head(15)
                    # Convert to JT for display
                    bs['Total Penjualan JT'] = bs['Total Penjualan'] / 1_000_000
                    fig = px.bar(bs, x='Total Penjualan JT', y='Pesanan', orientation='h', color='Pesanan', title=f"Top 15 Produk - {filter_mode}", text_auto=True)
                    fig.update_layout(showlegend=False, xaxis_title="Total Penjualan (JT Rupiah)", yaxis_title="Pesanan", paper_bgcolor=current_theme['card'], plot_bgcolor=current_theme['card'], font=dict(color=current_theme['text']))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Tidak ada produk ditemukan untuk kategori: {filter_mode}")
            
            else:
                # ✅ NEW: Category Ranking View
                if 'Kategori' in df.columns:
                    cat_stats = df.groupby('Kategori').agg({'Qty':'sum', 'Total Penjualan':'sum'}).reset_index().sort_values('Total Penjualan', ascending=False)
                    cat_stats['Total Penjualan JT'] = cat_stats['Total Penjualan'] / 1_000_000
                    
                    fig_cat = px.bar(cat_stats, x='Total Penjualan JT', y='Kategori', orientation='h', color='Kategori', title="Peringkat Penjualan per Kategori", text_auto=True)
                    fig_cat.update_layout(showlegend=False, xaxis_title="Total Penjualan (JT Rupiah)", yaxis_title="Kategori", paper_bgcolor=current_theme['card'], plot_bgcolor=current_theme['card'], font=dict(color=current_theme['text']))
                    st.plotly_chart(fig_cat, use_container_width=True)
                else:
                    st.error("Kolom 'Kategori' tidak ditemukan dalam dataset.")

        # TAB 3: PRIME TIME
        with tabs[2]:
            if 'Tanggal Order' in df.columns:
                df['Month'] = df['Tanggal Order'].dt.month_name()
                monthly = df.groupby('Month')['Total Penjualan'].sum()
                st.line_chart(monthly, color=current_theme['accent'])

        # TAB 4: CUSTOMERS (ENHANCED)
        with tabs[3]:
            st.subheader("Intelijen Pelanggan")
            if 'Cust' in df.columns:
                # Group by Customer
                cust_stats = df.groupby('Cust').agg({
                    'Total Penjualan': 'sum',
                    'Qty': 'sum',
                    'Pesanan': 'count' # Frequency
                }).rename(columns={'Pesanan': 'Frequency'}).sort_values('Total Penjualan', ascending=False)
                
                # Logic: Reseller vs End User
                # Rule: High Qty (>10) OR High Frequency (>5) = Reseller (Approximation)
                cust_stats['Tipe Customer'] = cust_stats.apply(
                    lambda x: '💎 Reseller' if (x['Qty'] > 10 or x['Frequency'] > 5) else '👤 End User',axis=1
                )
                
                st.dataframe(cust_stats.style.format({'Total Penjualan': lambda x: format_currency(x)}), use_container_width=True)

        # TAB 5: GEOGRAPHY (MAPS)
        with tabs[4]:
            st.subheader("Peta Distribusi")
            
            # 1. Filter Logic
            geo_filter = st.radio("Filter Peta:", ["Semua Item", "DKK (Keratonian)", "DKS (Standard)"], horizontal=True, key="geo_filter")
            
            df_geo = df.copy()
            if 'Daerah' in df_geo.columns:
                if geo_filter == "DKK (Keratonian)":
                    df_geo = df_geo[df_geo['Pesanan'].str.contains('DKK', na=False, case=False)]
                elif geo_filter == "DKS (Standard)":
                    df_geo = df_geo[df_geo['Pesanan'].str.contains('DKS', na=False, case=False)]

                # 2. Coordinate Mapping
                geo_agg = df_geo.groupby('Daerah').agg({'Total Penjualan': 'sum', 'Qty': 'sum'}).reset_index()
                geo_agg['lat'], geo_agg['lon'] = zip(*geo_agg['Daerah'].apply(get_coordinates))
                geo_agg = geo_agg.dropna(subset=['lat', 'lon']) # Remove unmapped cities

                # 3. Coloring Logic (DKK = Pekat/Dark, DKS = Terang/Light)
                # We achieve this by picking different color scales
                # DKK -> Dark Blue to Purple. DKS -> Light Blue to Cyan.
                if geo_filter == "DKK (Keratonian)":
                    color_continuous_scale = px.colors.sequential.Viridis # Darker palette
                elif geo_filter == "DKS (Standard)":
                    color_continuous_scale = px.colors.sequential.PuBu # Lighter palette
                else:
                    color_continuous_scale = px.colors.sequential.Plasma

                if not geo_agg.empty:
                    fig_map = px.scatter_mapbox(
                        geo_agg, 
                        lat="lat", lon="lon", 
                        size="Total Penjualan", 
                        color="Total Penjualan",
                        hover_name="Daerah", 
                        hover_data={"Total Penjualan": True, "Qty": True, "lat": False, "lon": False},
                        zoom=4, 
                        mapbox_style="carto-positron",
                        color_continuous_scale=color_continuous_scale,
                        title=f"Distribusi Penjualan - {geo_filter}"
                    )
                    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("Data lokasi tidak tersedia. Cek kolom 'Daerah'.")
            else:
                st.error("Kolom 'Daerah' tidak ditemukan.")

        # TAB 6: FORECASTING (2025)
        with tabs[5]:
            st.subheader("🔮 Prediksi Penjualan (Hingga Des 2025)")
            forecast_df = forecaster.predict_until_date("2025-12-31")
            
            if forecast_df is not None:
                st.line_chart(forecast_df.set_index('Date')['Predicted Sales'], color=current_theme['accent'])
                future_only = forecast_df[forecast_df['Type'] == 'Prediksi']
                st.write(f"Nilai Prediksi ({len(future_only)} Bulan Berikutnya):")
                st.dataframe(future_only.style.format({"Predicted Sales": lambda x: format_currency(x)}), use_container_width=True)

        # TAB 7: STOCK RECS
        with tabs[6]:
            _, stats = expert.generate_detailed_insights()
            if not stats.empty:
                st.subheader("Stock Health")
                
                # ✅ UPDATED: Added Glossary/Legend
                with st.expander("ℹ️ Glossary: ​​What does this status mean?"):
                    st.markdown("""
                    **Level Prioritas (Fuzzy Logic):**
                    - **🔥 High (Star Product)**: Penjualan sangat tinggi. **Rekomendasi**: Dorong iklan dan jaga stok.
                    - **✅ Medium (Stable)**: Penjualan konsisten. **Rekomendasi**: Pertahankan strategi saat ini.
                    - **⚠️ Critical (Slow Moving)**: Inventaris tinggi tapi penjualan rendah. **Rekomendasi**: Buat bundle promo atau diskon untuk menghabiskan stok.
                    """)
                
                st.dataframe(stats[['Pesanan', 'Qty', 'Recommendation', 'Priority']].style.applymap(lambda x: 'color: red; font-weight: bold;' if x == 'critical' else '', subset=['Priority']), use_container_width=True)

    # --------------------------
    # RIGHT: EXPERT AI PANEL
    # --------------------------
    if st.session_state.show_expert_panel:
        with side_col:
            st.markdown(f"""
            <div style="background-color: {current_theme['card']}; padding: 15px; border-radius: 10px; border: 1px solid {current_theme['accent']}; margin-bottom: 20px;">
                <h3 style="margin:0; color:{current_theme['accent']}">🤖 Expert Recomendation (AI)</h3>
            </div>
            """, unsafe_allow_html=True)
            
            insights, _ = expert.generate_detailed_insights()
            for insight in insights:
                icon = "ℹ️"
                if insight['type'] == 'success': icon = "✅"
                elif insight['type'] == 'warning': icon = "⚠️"
                
                # Fix bold rendering in HTML: replace **text** with <b>text</b>
                text_content = insight["text"]
                if "**" in text_content:
                    text_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text_content)
                
                st.markdown(f'<div style="background:{current_theme["card"]}; border-left: 3px solid {current_theme["accent"]}; padding: 10px; margin-bottom: 5px;"><b>{icon}</b> {text_content}</div>', unsafe_allow_html=True)
            
            st.divider()
            st.markdown("##### 💬 Asisten Chat")
            for msg in st.session_state.chat_history:
                role_class = "chat-user" if msg['role'] == 'user' else "chat-ai"
                prefix = "👤" if msg['role'] == 'user' else "🤖"
                st.markdown(f'<div class="{role_class}">{prefix} {msg["content"]}</div>', unsafe_allow_html=True)
            
            with st.form("chat_form", clear_on_submit=True):
                q = st.text_input("Tanya sesuatu...", placeholder="Apa strategi DKK?")
                if st.form_submit_button("Kirim") and q:
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    ans = expert.nlp_processor(q)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    st.rerun()

else:
    # LANDING PAGE
    st.info("👆 Silakan unggah data untuk memulai analisis.")

    gif_url = "https://raw.githubusercontent.com/USER/REPO/branch/path/to/SmartAssist.gif"

    st.markdown(
        f'<div style="text-align: center;"><img src="{gif_url}" '
        'alt="Smart Assist AI" style="max-width: 100%;"></div>',
        unsafe_allow_html=True,
    )
