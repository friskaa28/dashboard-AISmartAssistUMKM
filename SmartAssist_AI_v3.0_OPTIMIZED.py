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
import re
import google.generativeai as genai
import warnings

warnings.filterwarnings('ignore')

# 🔑 CONFIGURATION (Bisa diisi langsung di sini atau via UI)
GEMINI_API_KEY = "AIzaSyC8WHWXUPO9ZiYoHB7rBSuUQ8IspvjgIM4" # Masukkan API Key Anda di sini

def format_currency(value):
    if value >= 1_000_000:
        return f"Rp {value/1_000_000:,.1f} JT"
    else:
        return f"Rp {value:,.0f}"

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
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = GEMINI_API_KEY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Halo! 👋 Saya Asisten AI untuk UMKM yang siap bantu kamu.\nAku bisa bantu jelasin:\n✅ Produk apa yang lagi best seller\n✅ Kapan waktu terbaik buat jualan (Prime Time)\n✅ Siapa aja pelanggan setiamu\n✅ Prediksi penjualan ke depan\n✅ Kondisi stok gudang\n\nTanya aja apa yang mau kamu tau!"}
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

# EXPERT AI BUTTON STATE COLOR
expert_btn_color = "#02356C" if st.session_state.show_expert_panel else "#007BFF"

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
    
    /* EXPERT AI BUTTON - DYNAMIC COLOR */
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

    /* CHAT BUBBLES */
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
    
    /* TABLE HEADER FIX */
    [data-testid="stDataFrame"] th {{
        background-color: {current_theme['card']} !important;
        color: {current_theme['text']} !important;
    }}

    /* STICKY EXPERT AI COLUMN */
    [data-testid="column"]:nth-of-type(2) {{
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 50px !important;
        align-self: flex-start !important;
        max-height: 95vh !important;
        padding-right: 10px !important;
    }}

    /* INTERNAL CHAT SCROLL */
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


# ═══════════════════════════════════════════════════════════════
# 🧠 INTELLIGENT BACKEND
# ═══════════════════════════════════════════════════════════════

class ExpertSystem:
    def __init__(self, df):
        self.df = df
        # Normalize Column Names
        self.df.columns = self.df.columns.str.strip()

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
        if 'Total Penjualan' not in self.df.columns:
            return insights, pd.DataFrame()
            
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

    def analyze_overview(self):
        total_rev = self.df['Total Penjualan'].sum()
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
        if 'Pesanan' not in self.df.columns:
            return "Data Pesanan tidak tersedia."
        top_prod = self.df.groupby('Pesanan')['Total Penjualan'].sum().nlargest(3)
        msg = "🏆 Produk Unggulan (Top 3):\n"
        for i, (name, val) in enumerate(top_prod.items(), 1):
            msg += f"{i}. {name} ({format_currency(val)})\n"
        msg += "\n💡 **Saran**: Pastikan stok produk ini selalu aman karena berkontribusi terbesar pada omset."
        return msg

    def analyze_primetime(self):
        if 'Tanggal Order' not in self.df.columns:
            return "Data Tanggal Order tidak tersedia untuk analisis waktu."
        df_time = self.df.copy()
        df_time['Tanggal Order'] = pd.to_datetime(df_time['Tanggal Order'], errors='coerce')

        monthly = df_time.groupby(df_time['Tanggal Order'].dt.month_name())['Total Penjualan'].sum()
        if monthly.empty:
            return "Data waktu tidak cukup."

        peak_month_en = monthly.idxmax()
        peak_val = monthly.max()
        peak_month_id = self._get_indonesian_month(peak_month_en)
        return (
            f"⏰ Analisis Waktu (Prime Time):\n"
            f"- Bulan Puncak: {peak_month_id} (Tertinggi: {format_currency(peak_val)})\n\n"
            f"💡 Strategi: Siapkan stok & promo menjelang bulan {peak_month_id}."
        )

    def analyze_customers(self):
        if 'Cust' not in self.df.columns:
            return "Data Customer tidak tersedia."
        cust_counts = self.df['Cust'].value_counts()
        top_cust = self.df.groupby('Cust')['Total Penjualan'].sum().nlargest(1).index[0]
        return (
            f"👥 Profil Pelanggan:\n"
            f"- Total Pelanggan Unik: {len(cust_counts)}\n"
            f"- Top Customer: {top_cust}\n\n"
            f"💡 Tips: Cek tab Customer untuk lihat reseller (Qty > 10 atau Frequency > 5)."
        )

    def analyze_geography(self):
        if 'Daerah' not in self.df.columns:
            return "Data Daerah tidak tersedia."
        top_city = self.df.groupby('Daerah')['Total Penjualan'].sum().idxmax()
        uni_city = self.df['Daerah'].nunique()
        return (
            f"🌍 Distribusi Geografis:\n"
            f"- Jangkauan: {uni_city} Kota/Daerah\n"
            f"- Pasar Terbesar: {top_city}\n\n"
            f"💡 Ekspansi: Pertimbangkan subsidi ongkir atau cari reseller di kota baru."
        )

    def analyze_forecast(self):
        return (
            "💰 Prediksi Masa Depan:\n"
            "Sistem menggunakan Regresi Linear untuk memprediksi penjualan hingga akhir 2026.\n"
            "Cek tab 'Prediction' untuk grafik tren."
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

    # ==========================
    # ✅ NEW HELPERS (for 50+ Q)
    # ==========================

    def _clean_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def _has_cols(self, *cols):
        return all(c in self.df.columns for c in cols)

    def _top_star_products(self, n=5):
        _, stats = self.generate_detailed_insights()
        if stats.empty: 
            return None
        high = stats[stats["Priority"] == "high"].sort_values("Total Penjualan" if "Total Penjualan" in stats.columns else "Qty", ascending=False)
        if high.empty:
            # If no "high", take top by sales as "candidate"
            base = self.df.groupby("Pesanan")["Total Penjualan"].sum().sort_values(ascending=False).head(n)
            return ("candidate", base.index.tolist())
        return ("high", high["Pesanan"].head(n).tolist())

    def _slow_movers(self, n=5):
        _, stats = self.generate_detailed_insights()
        if stats.empty:
            return []
        slow = stats[stats["Priority"] == "critical"].sort_values("Qty", ascending=True)
        return slow["Pesanan"].head(n).tolist()

    def _category_ranking(self, n=5):
        if not self._has_cols("Kategori", "Total Penjualan"):
            return None
        cat = self.df.groupby("Kategori")["Total Penjualan"].sum().sort_values(ascending=False).head(n)
        return list(cat.items())

    def _get_indonesian_month(self, month_name_en):
        months = {
            'January': 'Januari', 'February': 'Februari', 'March': 'Maret',
            'April': 'April', 'May': 'Mei', 'June': 'Juni',
            'July': 'Juli', 'August': 'Agustus', 'September': 'September',
            'October': 'Oktober', 'November': 'November', 'December': 'Desember'
        }
        return months.get(month_name_en, month_name_en)

    def _worst_month(self):
        if not self._has_cols("Tanggal Order", "Total Penjualan"):
            return None
        dft = self.df.copy()
        dft["Tanggal Order"] = pd.to_datetime(dft["Tanggal Order"], errors="coerce")
        monthly = dft.groupby(dft["Tanggal Order"].dt.month_name())["Total Penjualan"].sum()
        if monthly.empty:
            return None
        worst_en = monthly.idxmin()
        worst_val = monthly.min()
        return self._get_indonesian_month(worst_en), worst_val

    def _top_city(self):
        if not self._has_cols("Daerah", "Total Penjualan"):
            return None
        s = self.df.groupby("Daerah")["Total Penjualan"].sum().sort_values(ascending=False)
        if s.empty:
            return None
        return s.index[0], s.iloc[0]

    def _growth_trend_simple(self):
        """Simple slope check on monthly revenue."""
        if not self._has_cols("Tanggal Order", "Total Penjualan"):
            return None
        dft = self.df.copy()
        dft["Tanggal Order"] = pd.to_datetime(dft["Tanggal Order"], errors="coerce")
        monthly = dft.set_index("Tanggal Order").resample("M")["Total Penjualan"].sum().dropna()
        if len(monthly) < 3:
            return None
        y = monthly.values
        x = np.arange(len(y))
        # slope
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def _help_menu(self):
        return (
            "Aku bisa jawab banyak hal, misalnya:\n"
            "- 'ringkasan bisnis'\n"
            "- 'produk terlaris'\n"
            "- 'produk yang harus diiklankan'\n"
            "- 'slow moving apa saja'\n"
            "- 'bulan terbaik / terburuk'\n"
            "- 'kota pasar terbesar'\n"
            "- 'ranking kategori'\n"
            "- 'prediksi penjualan'\n"
            "- 'customer terbaik / reseller'\n"
        )

    # ==========================
    # ✅ NEW: 50+ QUESTIONS NLP + GEMINI
    # ==========================
    def nlp_processor(self, user_query: str, api_key: str = None) -> str:
        q = self._clean_text(user_query)

        # ---- Rule Based Logic (Fallback/Hybrid) ----
        # ---- greetings / identity ----
        if re.search(r'\b(halo|hai|hi|hello|pagi|siang|sore|malam|siapa kamu|kamu siapa)\b', q):
            return (
                "Halo! 👋 Saya Asisten AI untuk UMKM.\n"
                "Saya bisa bantu analisis:\n"
                "✅ Ringkasan bisnis & tren\n"
                "✅ Produk terlaris & rekomendasi iklan\n"
                "✅ Prime time penjualan\n"
                "✅ Customer & reseller\n"
                "✅ Peta distribusi\n"
                "✅ Prediksi penjualan\n"
                "✅ Produk slow moving & strategi promo\n\n"
                "Tanya aja ya!"
            )

        # ---- help / examples ----
        if re.search(r'\b(bisa tanya apa|contoh pertanyaan|help|menu|bantuan|fitur)\b', q):
            return self._help_menu()

        # ---- overview / revenue / transactions ----
        if re.search(r'\b(ringkasan|overview|summary|gambaran|total|omset|omzet|pendapatan|revenue|penjualan|transaksi|berapa transaksi)\b', q):
            if re.search(r'\b(naik|turun|trend|tren|perkembangan|growth)\b', q):
                slope = self._growth_trend_simple()
                if slope is None:
                    return "Aku butuh kolom **Tanggal Order** dan data minimal beberapa bulan untuk analisis tren."
                if slope > 0:
                    return "📈 Tren penjualan cenderung **naik** berdasarkan akumulasi bulanan. Pertahankan strategi dan siapkan stok lebih baik."
                elif slope < 0:
                    return "📉 Tren penjualan cenderung **turun** berdasarkan akumulasi bulanan. Pertimbangkan promo, bundling, atau fokus ke produk star."
                else:
                    return "➖ Tren penjualan relatif **stabil**."
            return self.analyze_overview()

        # ---- best sellers / top products ----
        if re.search(r'\b(produk|barang|item|terlaris|unggulan|top|best seller|best|paling laku|paling menghasilkan|ranking produk)\b', q):
            if re.search(r'\b(iklankan|ads|push|promosi|promo iklan|boost)\b', q):
                res = self._top_star_products(5)
                if not res:
                    return "Aku belum bisa menentukan produk iklan karena data produk belum cukup."
                label, items = res
                if label == "high":
                    return "🎯 Produk yang paling layak diiklankan (Star Product):\n- " + "\n- ".join(items) + "\n\n💡 Fokus: iklan + jaga stok."
                return "🎯 Kandidat terbaik untuk iklan (berdasarkan omset tertinggi):\n- " + "\n- ".join(items)
            return self.analyze_bestsellers()

        # ---- prime time / month peak / worst month ----
        if re.search(r'\b(bulan terburuk|bulan sepi|worst month|paling rendah|paling dikit|penjualan rendah)\b', q):
            worst = self._worst_month()
            if not worst:
                return "Aku butuh kolom **Tanggal Order** untuk menentukan bulan terburuk."
            m, v = worst
            return f"📉 Bulan terburuk: **{m}** dengan penjualan sekitar {format_currency(v)}.\n\n💡 Saran: siapkan promo khusus di bulan ini untuk menaikkan minat pembeli."

        if re.search(r'\b(prime|waktu terbaik|kapan|bulan terbaik|bulan ramai|musim|peak|puncak)\b', q):
            return self.analyze_primetime()

        # ---- customers ----
        if re.search(r'\b(customer|cust|pelanggan|pembeli|reseller|loyal|terbaik|paling sering)\b', q):
            if re.search(r'\b(reseller)\b', q) and 'Cust' in self.df.columns:
                cust_stats = self.df.groupby('Cust').agg({'Total Penjualan': 'sum','Qty': 'sum','Pesanan': 'count'}).rename(columns={'Pesanan':'Frequency'})
                reseller = cust_stats[(cust_stats['Qty'] > 10) | (cust_stats['Frequency'] > 5)].sort_values('Total Penjualan', ascending=False)
                if reseller.empty:
                    return "Belum terlihat reseller kuat dari aturan sederhana (Qty>10 atau Frequency>5)."
                top = reseller.head(5).index.tolist()
                return "💎 Reseller potensial (Top 5):\n- " + "\n- ".join(top) + "\n\n💡 Beri harga khusus / paket grosir."
            return self.analyze_customers()

        # ---- geography ----
        if re.search(r'\b(peta|lokasi|daerah|kota|wilayah|geo|geografi|pasar terbesar)\b', q):
            if re.search(r'\b(pasar terbesar|kota terbaik|kota mana)\b', q):
                top = self._top_city()
                if not top:
                    return "Aku butuh kolom **Daerah** untuk menentukan pasar terbesar."
                city, val = top
                return f"🏙️ Pasar terbesar saat ini: **{city}** dengan penjualan {format_currency(val)}.\n\n💡 Bisa coba cari reseller di {city}."
            return self.analyze_geography()

        # ---- stock / slow moving / bundle / discount ----
        if re.search(r'\b(stok|stock|inventory|gudang|slow moving|menumpuk|diskon|bundle|habiskan|clearance)\b', q):
            if re.search(r'\b(slow moving|menumpuk|kurang laku|habiskan|diskon|bundle)\b', q):
                slow = self._slow_movers(5)
                if not slow:
                    return "Belum ada indikasi slow moving dari data saat ini."
                return "⚠️ Produk slow moving (contoh 5 teratas):\n- " + "\n- ".join(slow) + "\n\n💡 Saran: bundling, diskon terbatas, atau bonus ongkir."
            return self.analyze_stock()

        # ---- category ranking ----
        if re.search(r'\b(kategori|category|ranking kategori|peringkat kategori)\b', q):
            cat = self._category_ranking(5)
            if not cat:
                return "Aku butuh kolom **Kategori** untuk membuat ranking kategori."
            lines = [f"- {k}: {format_currency(v)}" for k, v in cat]
            return "📌 Peringkat Kategori (Top 5):\n" + "\n".join(lines)

        # ---- forecast ----
        if re.search(r'\b(prediksi|forecast|proyeksi|masa depan|tahun depan|2026|trend ke depan|ramalan)\b', q):
            return self.analyze_forecast()

        # ---- DKK / DKS specific strategies ----
        if re.search(r'\b(dkk|keratonian)\b', q):
            return (
                "🛡️ **Strategi DKK (Keratonian)**:\n"
                "Produk ini adalah premium target. Pastikan visual iklan terlihat eksklusif.\n"
                "💡 Saran: Gunakan influencer niche atau branding yang menekankan kualitas bahan."
            )
            
        if re.search(r'\b(dks|standar)\b', q):
            return (
                "📦 **Strategi DKS (Standard)**:\n"
                "Produk ini bersifat mass-market. Fokus pada volume penjualan.\n"
                "💡 Saran: Buat promo 'Beli 3 Lebih Hemat' atau bundling dengan produk best seller lainnya."
            )

        # ---- qty / volume ----
        if re.search(r'\b(berapa banyak|jumlah barang|qty|volume terjual|total unit)\b', q):
            total_qty = self.df['Qty'].sum()
            return f"📦 Total volume barang terjual mencapai **{total_qty:,} unit**. Cek tab Best Seller untuk detail per item."

        # ---- marketing strategy ----
        if re.search(r'\b(marketing|pemasaran|promosi|iklan|ads|jualan|strategi bisnis|cara laku)\b', q):
            # Check trend
            slope = self._growth_trend_simple()
            trend_msg = ""
            if slope is not None:
                if slope > 0: trend_msg = "Tren Anda sedang **naik**, saatnya ekspansi iklan ke wilayah baru."
                else: trend_msg = "Tren Anda sedang **turun**, fokus pada retensi pelanggan setia dan promo bundle."
            
            # Check stock
            _, stats = self.generate_detailed_insights()
            stock_msg = ""
            if not stats.empty:
                slow = len(stats[stats['Priority'] == 'critical'])
                if slow > 0: stock_msg = f"Terdapat {slow} produk slow moving. Segera buat **promo cuci gudang** atau **bundle hemat**."
            
            return (
                "📢 **Rekomendasi Strategi Marketing**:\n"
                f"1. **Tren**: {trend_msg if trend_msg else 'Jaga konsistensi konten media sosial.'}\n"
                f"2. **Stok**: {stock_msg if stock_msg else 'Stok aman, fokus pada produk Star (terlaris).'}\n"
                "3. **Channel**: Optimalkan Marketplace Ads di jam-jam ramai (Prime Time).\n"
                "4. **Loyalty**: Beri voucher khusus untuk Top Customer agar mereka repeat order sebagai Reseller."
            )

        # ---- GEMINI LLM integration ----
        if api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # Context building
                context = (
                    f"Anda adalah analis bisnis UMKM. Anda memiliki data dashboard dengan ringkasan:\n"
                    f"{self.analyze_overview()}\n"
                    f"{self.analyze_stock()}\n"
                    f"Gunakan data ini untuk menjawab pertanyaan user secara profesional, singkat, dan solutif."
                )
                
                response = model.generate_content(f"{context}\n\nPertanyaan User: {user_query}")
                return response.text
                
            except Exception as e:
                return f"❌ Terjadi kesalahan saat menghubungi Gemini: {str(e)}\n\nCoba periksa API Key Anda atau gunakan pertanyaan yang dipahami sistem rule-based."

        # ---- fallback ----
        return (
            "Maaf, aku belum paham pertanyaannya.\n\n"
            "Coba tanya dengan contoh:\n"
            "- 'ringkasan bisnis'\n"
            "- 'produk terlaris'\n"
            "- 'produk yang harus diiklankan'\n"
            "- 'slow moving apa'\n"
            "- 'bulan terbaik / terburuk'\n"
            "- 'pasar terbesar kota mana'\n"
            "- 'ranking kategori'\n"
            "- 'prediksi penjualan'\n"
        )

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
        
        # ✅ FIX: Format date to remove time component
        future_dates = [d.date() for d in future_dates]
        monthly_index_date = [d.date() for d in monthly.index]
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': forecast, 'Type': 'Prediksi'})
        history_df = pd.DataFrame({'Date': monthly_index_date, 'Predicted Sales': monthly.values, 'Type': 'Riwayat'})
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

# DATA LOADING & SIDEBAR
uploaded_file = st.sidebar.file_uploader("📂 Unggah Data", type=['xlsx', 'csv'])

# API KEY SECTION
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 API Configuration")
st.session_state.gemini_api_key = st.sidebar.text_input(
    "Gemini API Key", 
    value=st.session_state.gemini_api_key, 
    type="password",
    help="Dapatkan di: https://makersuite.google.com/app/apikey"
)
if st.session_state.gemini_api_key:
    st.sidebar.success("✅ Gemini AI Ready")
else:
    st.sidebar.info("💡 Masukkan API Key untuk fitur AI yang lebih cerdas.")

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
            st.subheader("🔮 Prediksi Penjualan (Hingga Des 2026)")
            forecast_df = forecaster.predict_until_date("2026-12-31")
            
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
                    **Level Prioritas:**
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
            
            # --- INTERNAL SCROLLABLE CHAT AREA ---
            chat_html = '<div class="chat-container">'
            for msg in st.session_state.chat_history:
                role_class = "chat-user" if msg['role'] == 'user' else "chat-ai"
                prefix = "👤" if msg['role'] == 'user' else "🤖"
                
                # Formatter: Convert Markdown Bold to HTML Bold
                content_display = msg["content"]
                if "**" in content_display:
                    import re
                    content_display = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content_display)
                
                chat_html += f'<div class="{role_class}">{prefix} {content_display}</div>'
            chat_html += '</div>'
            
            st.markdown(chat_html, unsafe_allow_html=True)
            
            with st.form("chat_form", clear_on_submit=True):
                q = st.text_input("Tanya sesuatu...", placeholder="Apa strategi DKK?")
                if st.form_submit_button("Kirim") and q:
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    # Pass API key to enable Gemini if available
                    ans = expert.nlp_processor(q, api_key=st.session_state.gemini_api_key)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    st.rerun()

else:
    # LANDING PAGE
    st.info("👆 Silakan unggah data untuk memulai analisis.")

    gif_url = "https://raw.githubusercontent.com/friskaa28/dashboard-AISmartAssistUMKM/master/Smart%20Assist%20ai%20(2)%20dengan%20latar%20belakang%20putih.gif"

    st.markdown(
        f'<div style="text-align: center;"><img src="{gif_url}" '
        'alt="Smart Assist AI" style="max-width: 100%;"></div>',
        unsafe_allow_html=True,
    )
