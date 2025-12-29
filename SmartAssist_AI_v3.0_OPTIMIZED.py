# 🎯 SmartAssist AI v3.0 - OPTIMIZED EXPERT SYSTEM
# Integrated ML + Fuzzy Logic + NLP in a Side Panel UI
# Refined: Geo Map, Customer Types, Deep Dark Mode, Blue Buttons
# ✅ Integrated with PHP Chat API (Groq): https://arifproject.my.id/chat/api/chat.php

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px
import re
import warnings
import requests

# Optional (Gemini). Keep disabled by default.
# import google.generativeai as genai

warnings.filterwarnings('ignore')

# ============================================================
# ✅ PHP CHAT API CONFIG
# ============================================================
PHP_CHAT_URL = "https://arifproject.my.id/chat/api/chat.php"  # your hosted PHP endpoint

def call_php_chat(user_text: str, chat_history: list[dict]) -> str:
    """
    Send message + history to your PHP endpoint (Groq) and return assistant reply.
    chat_history format: [{"role":"user|assistant","content":"..."}]
    """
    history = []
    for m in chat_history:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ["user", "assistant"] and isinstance(content, str):
            history.append({"role": role, "content": content})

    payload = {"message": user_text, "history": history}
    r = requests.post(PHP_CHAT_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("reply", "")


# ============================================================
# 🔑 OPTIONAL: GEMINI (DISABLED DEFAULT)
# ============================================================
# ⚠️ Jangan hardcode API Key di source publik.
# Gemini dipakai hanya kalau kamu mau fallback selain PHP chat.
USE_GEMINI_FALLBACK = False
GEMINI_API_KEY = ""  # isi via sidebar kalau mau

def format_currency(value):
    try:
        value = float(value)
    except:
        value = 0.0
    if value >= 1_000_000:
        return f"Rp {value/1_000_000:,.1f} JT"
    return f"Rp {value:,.0f}"


# ============================================================
# 🎨 PAGE CONFIG & THEME
# ============================================================
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
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = GEMINI_API_KEY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content":
            "Halo! 👋 Saya Asisten AI untuk UMKM yang siap bantu kamu.\n"
            "Aku bisa bantu:\n"
            "✅ Produk best seller\n"
            "✅ Prime time jualan\n"
            "✅ Customer setia\n"
            "✅ Prediksi penjualan\n"
            "✅ Rekomendasi stok\n\n"
            "Tanya aja ya!"
        }
    ]

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
expert_btn_color = "#02356C" if st.session_state.show_expert_panel else "#007BFF"

# CSS UI
st.markdown(f"""
<style>
    .stApp, .stApp > header, .stApp > div {{
        background-color: {current_theme['bg']} !important;
        color: {current_theme['text']} !important;
    }}
    div[data-testid="stSidebar"], div[data-testid="stHeader"] {{
        background-color: {current_theme['bg']};
    }}
    p, h1, h2, h3, h4, span, div, label, .stMarkdown {{
        color: {current_theme['text']} !important;
    }}
    .css-card {{
        background-color: {current_theme['card']};
        border: 1px solid {current_theme['border']};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        color: {current_theme['text']};
    }}
    div.stButton > button[kind="primary"] {{
        background-color: {expert_btn_color} !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }}
    div.stButton > button[kind="primary"] * {{
        color: white !important;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: #004494 !important;
    }}
    .chat-user {{
        background: {expert_btn_color};
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
    [data-testid="stDataFrame"] th {{
        background-color: {current_theme['card']} !important;
        color: {current_theme['text']} !important;
    }}
    [data-testid="column"]:nth-of-type(2) {{
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 50px !important;
        align-self: flex-start !important;
        max-height: 95vh !important;
        padding-right: 10px !important;
    }}
    .chat-container {{
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid {current_theme['border']};
        border-radius: 10px;
        background: {current_theme['bg']};
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 🧠 BACKEND CLASSES
# ============================================================
class ExpertSystem:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip()

    def fuzzy_decision_engine(self, row):
        sales_score = row.get('norm_sales', 0)
        if sales_score > 0.8:
            return "🔥 STAR PRODUCT (Push Ads)", "high"
        elif sales_score > 0.4:
            return "✅ STABLE (Maintain)", "medium"
        return "⚠️ SLOW MOVING (Diskon/Bundle)", "critical"

    def generate_detailed_insights(self):
        insights = []
        if 'Total Penjualan' not in self.df.columns or 'Pesanan' not in self.df.columns or 'Qty' not in self.df.columns:
            return insights, pd.DataFrame()

        total_sales = float(self.df['Total Penjualan'].sum())
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

            rec_pri = product_stats.apply(self.fuzzy_decision_engine, axis=1)
            product_stats['Recommendation'] = rec_pri.apply(lambda x: x[0])
            product_stats['Priority'] = rec_pri.apply(lambda x: x[1])

            slow_movers = product_stats[product_stats['Priority'] == 'critical']
            if not slow_movers.empty:
                count = len(slow_movers)
                top_slow = slow_movers.sort_values('Qty').head(1)['Pesanan'].values[0]
                insights.append({
                    "type": "warning",
                    "text": f"⚠️ **{count} Produk Slow Moving** (e.g., {top_slow}). Buat promo bundle!"
                })

        return insights, product_stats

    def analyze_overview(self):
        if 'Total Penjualan' not in self.df.columns or 'Pesanan' not in self.df.columns:
            return "Kolom wajib tidak lengkap untuk ringkasan."
        total_rev = float(self.df['Total Penjualan'].sum())
        total_order = len(self.df)
        total_prod = self.df['Pesanan'].nunique()
        return (
            f"📊 **Ringkasan Bisnis**:\n"
            f"- Total Pendapatan: {format_currency(total_rev)}\n"
            f"- Total Transaksi: {total_order}\n"
            f"- Variasi Produk: {total_prod} Item\n\n"
            f"Bisnis Anda berjalan aktif! Cek tab lain untuk detail lebih dalam."
        )

    def analyze_bestsellers(self):
        if 'Pesanan' not in self.df.columns or 'Total Penjualan' not in self.df.columns:
            return "Data tidak lengkap untuk best seller."
        top_prod = self.df.groupby('Pesanan')['Total Penjualan'].sum().nlargest(3)
        msg = "🏆 Produk Unggulan (Top 3):\n"
        for i, (name, val) in enumerate(top_prod.items(), 1):
            msg += f"{i}. {name} ({format_currency(val)})\n"
        msg += "\n💡 **Saran**: Pastikan stok produk ini selalu aman karena berkontribusi terbesar pada omset."
        return msg

    def analyze_primetime(self):
        if 'Tanggal Order' not in self.df.columns or 'Total Penjualan' not in self.df.columns:
            return "Data Tanggal Order tidak tersedia untuk analisis waktu."
        df_time = self.df.copy()
        df_time['Tanggal Order'] = pd.to_datetime(df_time['Tanggal Order'], errors='coerce')

        valid_dates = df_time['Tanggal Order'].dropna()
        if valid_dates.empty:
            return "Data tanggal tidak valid."
        min_year = valid_dates.dt.year.min()
        max_year = valid_dates.dt.year.max()
        year_range = f"{int(min_year)} - {int(max_year)}" if min_year != max_year else f"{int(min_year)}"

        monthly = df_time.groupby(df_time['Tanggal Order'].dt.month_name())['Total Penjualan'].sum()
        if monthly.empty:
            return "Data waktu tidak cukup."

        peak_month = monthly.idxmax()
        peak_val = monthly.max()
        return (
            f"⏰ **Analisis Prime Time ({year_range})**:\n"
            f"- Bulan Puncak: {peak_month} (Tertinggi: {format_currency(peak_val)})\n\n"
            f"💡 Data ini akumulasi penjualan rentang tahun {year_range} untuk melihat tren musiman."
        )

    def analyze_customers(self, metric='Total Penjualan', top=True):
        if 'Cust' not in self.df.columns:
            return "Data Customer tidak tersedia."
        if 'Total Penjualan' not in self.df.columns or 'Qty' not in self.df.columns:
            return "Data tidak lengkap untuk analisis pelanggan."

        cust_stats = self.df.groupby('Cust').agg({
            'Total Penjualan': 'sum',
            'Qty': 'sum'
        })

        if cust_stats.empty:
            return "Data customer kosong."

        sort_col = 'Qty' if metric == 'Qty' else 'Total Penjualan'
        target_row = cust_stats.nlargest(1, sort_col) if top else cust_stats.nsmallest(1, sort_col)
        label = "tertinggi" if top else "terendah"

        target_cust = target_row.index[0]
        target_val = target_row[sort_col].values[0]

        metric_label = "Jumlah Barang (Qty)" if sort_col == 'Qty' else "Total Pendapatan"
        val_display = f"{int(target_val):,} unit" if sort_col == 'Qty' else format_currency(target_val)

        return (
            f"👥 **Analisis Pelanggan ({label.capitalize()})**:\n"
            f"- Berdasarkan: {metric_label}\n"
            f"- Nama Customer: **{target_cust}**\n"
            f"- Nilai: {val_display}\n\n"
            f"💡 Total pelanggan unik: {len(cust_stats)} orang."
        )

    def analyze_geography(self):
        if 'Daerah' not in self.df.columns or 'Total Penjualan' not in self.df.columns:
            return "Data Daerah tidak tersedia."
        top_city = self.df.groupby('Daerah')['Total Penjualan'].sum().idxmax()
        uni_city = self.df['Daerah'].nunique()
        return (
            f"🌍 Distribusi Geografis:\n"
            f"- Jangkauan: {uni_city} Kota/Daerah\n"
            f"- Pasar Terbesar: {top_city}\n\n"
            f"💡 Ekspansi: pertimbangkan subsidi ongkir atau cari reseller di kota baru."
        )

    def analyze_stock(self):
        _, stats = self.generate_detailed_insights()
        if stats.empty:
            return "📦 Data stok tidak cukup untuk analisis (produk_stats kosong)."

        critical = stats[stats['Priority'] == 'critical']
        high = stats[stats['Priority'] == 'high']

        msg = "📦 Kesehatan Stok & Rekomendasi:\n"
        if not high.empty:
            msg += f"- 🔥 {len(high)} Produk Star: jaga stok jangan sampai kosong.\n"
        if not critical.empty:
            msg += f"- ⚠️ {len(critical)} Produk Slow Moving: buat diskon/bundle.\n"
        if high.empty and critical.empty:
            msg += "- ✅ Mayoritas produk stabil.\n"
        return msg


class SalesForecaster:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if 'Tanggal Order' in self.df.columns:
            self.df['Tanggal Order'] = pd.to_datetime(self.df['Tanggal Order'], errors='coerce')

    def predict_until_date(self, target_date="2026-12-31"):
        if 'Tanggal Order' not in self.df.columns or 'Total Penjualan' not in self.df.columns:
            return None
        monthly = self.df.set_index('Tanggal Order').resample('M')['Total Penjualan'].sum()
        if len(monthly) < 2:
            return None

        last_date = monthly.index[-1]
        target_dt = pd.to_datetime(target_date)
        delta = (target_dt.year - last_date.year) * 12 + (target_dt.month - last_date.month)
        if delta <= 0:
            delta = 12

        X = np.arange(len(monthly)).reshape(-1, 1)
        y = monthly.values
        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(monthly), len(monthly) + delta).reshape(-1, 1)
        forecast = model.predict(future_X)

        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, delta + 1)]
        future_dates = [d.date() for d in future_dates]
        monthly_index_date = [d.date() for d in monthly.index]

        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': forecast, 'Type': 'Prediksi'})
        history_df = pd.DataFrame({'Date': monthly_index_date, 'Predicted Sales': monthly.values, 'Type': 'Riwayat'})
        return pd.concat([history_df, forecast_df], ignore_index=True)


# ============================================================
# 🌍 GEOGRAPHY UTILS
# ============================================================
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
    if not isinstance(city_name, str):
        return None, None
    for city, coords in INDONESIA_CITIES.items():
        if city.lower() in city_name.lower():
            return coords['lat'], coords['lon']
    return None, None


# ============================================================
# 📱 SIDEBAR
# ============================================================
uploaded_file = st.sidebar.file_uploader("📂 Unggah Data", type=['xlsx', 'csv'])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Chat Backend")
st.sidebar.write("Chat panel menggunakan PHP+Groq endpoint:")
st.sidebar.code(PHP_CHAT_URL)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Gemini (Opsional)")
st.session_state.gemini_api_key = st.sidebar.text_input(
    "Gemini API Key (Optional)",
    value=st.session_state.gemini_api_key,
    type="password"
)
USE_GEMINI_FALLBACK = st.sidebar.checkbox("Gunakan Gemini fallback jika PHP gagal", value=False)


# ============================================================
# HEADER
# ============================================================
c1, c2 = st.columns([7, 3])
with c1:
    if uploaded_file:
        st.title("🎯 SmartAssist AI: Intelligent Sales Analytics and Decision Support Platform for Empowering UMKM")
        st.caption("Dashboard Analisis Terintegrasi")
    else:
        st.title("HOME")

with c2:
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🌓 Theme", use_container_width=True):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
    with btn_col2:
        if st.button("🤖 Expert AI", type="primary", use_container_width=True):
            st.session_state.show_expert_panel = not st.session_state.show_expert_panel
            st.rerun()

# Layout
if st.session_state.show_expert_panel:
    main_col, side_col = st.columns([3, 1.2])
else:
    main_col = st.container()


# ============================================================
# MAIN
# ============================================================
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip()

    expert = ExpertSystem(df)
    forecaster = SalesForecaster(df)

    with main_col:
        tab_titles = [
            "📊 Overview", "🏆 Best Seller", "⏰ Prime Time",
            "👥 Customers", "🌍 Geography (Map)", "💰 Prediction", "📦 Stock Recommendation"
        ]
        tabs = st.tabs(tab_titles)

        # TAB 1
        with tabs[0]:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            cA, cB, cC = st.columns(3)
            cA.metric("Pendapatan", format_currency(df['Total Penjualan'].sum()) if 'Total Penjualan' in df.columns else "N/A")
            cB.metric("Total Pesanan", len(df))
            cC.metric("Total Produk", df['Pesanan'].nunique() if 'Pesanan' in df.columns else 0)
            st.markdown('</div>', unsafe_allow_html=True)

            if 'Tanggal Order' in df.columns and 'Total Penjualan' in df.columns:
                dfx = df.copy()
                dfx['Tanggal Order'] = pd.to_datetime(dfx['Tanggal Order'], errors='coerce')
                daily = dfx.groupby('Tanggal Order')['Total Penjualan'].sum()
                st.area_chart(daily)

        # TAB 2
        with tabs[1]:
            st.subheader("Analisis Produk Unggulan")
            view_mode = st.radio(
                "Mode Tampilan:",
                ["Detail Produk (per Kategori)", "Peringkat Kategori Global"],
                horizontal=True,
                key="bs_view_mode"
            )

            if view_mode == "Detail Produk (per Kategori)":
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
                        st.warning("Kolom 'Kategori' tidak ditemukan.")

                if len(df_display) > 0 and 'Pesanan' in df_display.columns and 'Total Penjualan' in df_display.columns and 'Qty' in df_display.columns:
                    bs = df_display.groupby('Pesanan').agg({'Qty': 'sum', 'Total Penjualan': 'sum'}).reset_index()
                    bs = bs.sort_values('Total Penjualan', ascending=False).head(15)
                    bs['Total Penjualan JT'] = bs['Total Penjualan'] / 1_000_000

                    fig = px.bar(bs, x='Total Penjualan JT', y='Pesanan', orientation='h', text_auto=True,
                                 title=f"Top Produk - {filter_mode}")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Data tidak cukup untuk menampilkan chart (butuh Pesanan, Qty, Total Penjualan).")

            else:
                if 'Kategori' in df.columns and 'Total Penjualan' in df.columns:
                    cat_stats = df.groupby('Kategori').agg({'Qty': 'sum', 'Total Penjualan': 'sum'}).reset_index()
                    cat_stats = cat_stats.sort_values('Total Penjualan', ascending=False)
                    cat_stats['Total Penjualan JT'] = cat_stats['Total Penjualan'] / 1_000_000

                    fig_cat = px.bar(cat_stats, x='Total Penjualan JT', y='Kategori', orientation='h', text_auto=True,
                                     title="Peringkat Penjualan per Kategori")
                    fig_cat.update_layout(showlegend=False)
                    st.plotly_chart(fig_cat, use_container_width=True)
                else:
                    st.error("Kolom 'Kategori' dan/atau 'Total Penjualan' tidak ditemukan.")

        # TAB 3
        with tabs[2]:
            if 'Tanggal Order' in df.columns and 'Total Penjualan' in df.columns:
                dft = df.copy()
                dft['Tanggal Order'] = pd.to_datetime(dft['Tanggal Order'], errors='coerce')
                valid_dates = dft['Tanggal Order'].dropna()
                if not valid_dates.empty:
                    min_year = valid_dates.dt.year.min()
                    max_year = valid_dates.dt.year.max()
                    year_range = f"{int(min_year)} - {int(max_year)}" if min_year != max_year else f"{int(min_year)}"
                    st.info(f"💡 Grafik akumulasi penjualan bulanan rentang tahun **{year_range}** untuk melihat pola musiman.")

                dft['Month'] = dft['Tanggal Order'].dt.month_name()
                monthly = dft.groupby('Month')['Total Penjualan'].sum()
                st.line_chart(monthly)
            else:
                st.warning("Butuh kolom 'Tanggal Order' dan 'Total Penjualan'.")

        # TAB 4
        with tabs[3]:
            st.subheader("Intelijen Pelanggan")
            if all(c in df.columns for c in ['Cust', 'Total Penjualan', 'Qty', 'Pesanan']):
                cust_stats = df.groupby('Cust').agg({
                    'Total Penjualan': 'sum',
                    'Qty': 'sum',
                    'Pesanan': 'count'
                }).rename(columns={'Pesanan': 'Frequency'}).sort_values('Total Penjualan', ascending=False)

                cust_stats['Tipe Customer'] = cust_stats.apply(
                    lambda x: '💎 Reseller' if (x['Qty'] > 10 or x['Frequency'] > 5) else '👤 End User',
                    axis=1
                )
                st.dataframe(cust_stats, use_container_width=True)
            else:
                st.warning("Butuh kolom: Cust, Total Penjualan, Qty, Pesanan.")

        # TAB 5
        with tabs[4]:
            st.subheader("Peta Distribusi")

            geo_filter = st.radio("Filter Peta:", ["Semua Item", "DKK (Kerucut)", "DKS (Stik)"],
                                  horizontal=True, key="geo_filter")

            if 'Daerah' in df.columns and 'Pesanan' in df.columns and 'Total Penjualan' in df.columns and 'Qty' in df.columns:
                df_geo = df.copy()
                if geo_filter == "DKK (Kerucut)":
                    df_geo = df_geo[df_geo['Pesanan'].astype(str).str.contains('DKK', na=False, case=False)]
                    color_scale = px.colors.sequential.Viridis
                elif geo_filter == "DKS (Stik)":
                    df_geo = df_geo[df_geo['Pesanan'].astype(str).str.contains('DKS', na=False, case=False)]
                    color_scale = px.colors.sequential.PuBu
                else:
                    color_scale = px.colors.sequential.Plasma

                geo_agg = df_geo.groupby('Daerah').agg({'Total Penjualan': 'sum', 'Qty': 'sum'}).reset_index()
                geo_agg['lat'], geo_agg['lon'] = zip(*geo_agg['Daerah'].apply(get_coordinates))
                geo_agg = geo_agg.dropna(subset=['lat', 'lon'])

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
                        color_continuous_scale=color_scale,
                        title=f"Distribusi Penjualan - {geo_filter}"
                    )
                    fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("Tidak ada kota yang ter-mapping (cek nama daerah).")
            else:
                st.warning("Butuh kolom: Daerah, Pesanan, Total Penjualan, Qty.")

        # TAB 6
        with tabs[5]:
            st.subheader("🔮 Prediksi Penjualan (Hingga Des 2026)")
            forecast_df = forecaster.predict_until_date("2026-12-31")
            if forecast_df is not None:
                st.line_chart(forecast_df.set_index('Date')['Predicted Sales'])
                future_only = forecast_df[forecast_df['Type'] == 'Prediksi']
                st.write(f"Nilai Prediksi ({len(future_only)} Bulan Berikutnya):")
                st.dataframe(future_only, use_container_width=True)
            else:
                st.warning("Data tidak cukup untuk forecast (butuh tanggal dan penjualan minimal 2 bulan).")

        # TAB 7
        with tabs[6]:
            _, stats = expert.generate_detailed_insights()
            if not stats.empty:
                st.subheader("Stock Health")
                with st.expander("ℹ️ Glossary: What does this status mean?"):
                    st.markdown("""
                    **Level Prioritas:**
                    - **🔥 High (Star Product)**: Penjualan sangat tinggi. Dorong iklan & jaga stok.
                    - **✅ Medium (Stable)**: Penjualan konsisten. Pertahankan strategi.
                    - **⚠️ Critical (Slow Moving)**: Penjualan rendah. Buat bundle/diskon.
                    """)
                st.dataframe(stats[['Pesanan', 'Qty', 'Recommendation', 'Priority']], use_container_width=True)
            else:
                st.warning("Stats belum terbentuk (pastikan kolom Pesanan/Qty/Total Penjualan ada).")

    # ============================================================
    # RIGHT SIDE PANEL (CHAT)
    # ============================================================
    if st.session_state.show_expert_panel:
        with side_col:
            st.markdown(f"""
            <div style="background-color: {current_theme['card']}; padding: 15px; border-radius: 10px; border: 1px solid {current_theme['accent']}; margin-bottom: 20px;">
                <h3 style="margin:0; color:{current_theme['accent']}">🤖 Expert Recommendation (AI)</h3>
            </div>
            """, unsafe_allow_html=True)

            insights, _ = expert.generate_detailed_insights()
            for insight in insights:
                icon = "ℹ️"
                if insight.get('type') == 'warning':
                    icon = "⚠️"
                text_content = insight.get("text", "")
                if "**" in text_content:
                    text_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text_content)
                st.markdown(
                    f'<div style="background:{current_theme["card"]}; border-left: 3px solid {current_theme["accent"]}; padding: 10px; margin-bottom: 5px;"><b>{icon}</b> {text_content}</div>',
                    unsafe_allow_html=True
                )

            st.divider()
            st.markdown("##### 💬 Asisten Chat (via PHP/Groq)")

            chat_html = '<div class="chat-container">'
            for msg in st.session_state.chat_history:
                role_class = "chat-user" if msg['role'] == 'user' else "chat-ai"
                prefix = "👤" if msg['role'] == 'user' else "🤖"
                content_display = msg.get("content", "")
                if "**" in content_display:
                    content_display = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content_display)
                chat_html += f'<div class="{role_class}">{prefix} {content_display}</div>'
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)

            with st.form("chat_form", clear_on_submit=True):
                q = st.text_input("Tanya sesuatu...", placeholder="Apa strategi DKK?")
                if st.form_submit_button("Kirim") and q:
                    st.session_state.chat_history.append({"role": "user", "content": q})

                    # ✅ primary: PHP/Groq
                    try:
                        ans = call_php_chat(q, st.session_state.chat_history)
                        if not ans:
                            ans = "(empty reply)"
                    except Exception as e:
                        ans = f"❌ Gagal menghubungi PHP Chat API: {e}"

                        # Optional fallback to Gemini if enabled and key provided
                        if USE_GEMINI_FALLBACK and st.session_state.gemini_api_key:
                            ans += "\n\n(Opsional) Aktifkan Gemini fallback dengan implementasi terpisah."

                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    st.rerun()

else:
    st.info("👆 Silakan unggah data untuk memulai analisis.")

    gif_url = "https://raw.githubusercontent.com/friskaa28/dashboard-AISmartAssistUMKM/master/Smart%20Assist%20ai%20(2)%20dengan%20latar%20belakang%20putih.gif"
    st.markdown(
        f'<div style="text-align: center;"><img src="{gif_url}" alt="Smart Assist AI" style="max-width: 100%;"></div>',
        unsafe_allow_html=True,
    )
