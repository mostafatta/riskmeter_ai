import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import time

# === Fix Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

try:
    from src.data_loader import TadawulDataLoader
    from src.calculations import RiskCalculator
    from src.risk_labeler import RiskLabeler
except ModuleNotFoundError:
    st.error("⚠️ Error: 'src' folder not found. Make sure app.py is next to the src folder.")
    st.stop()

# === UI Configuration ===
st.set_page_config(page_title="Tadawul Risk AI", page_icon="📈", layout="wide")

# ============================================================
# 🔥 CACHING LAYER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_ai_model():
    model_path = os.path.join(BASE_DIR, "models", "risk_classifier.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource(show_spinner=False)
def load_metadata():
    meta_path = os.path.join(BASE_DIR, 'data', 'raw', "stocks_metadata.csv")
    if os.path.exists(meta_path):
        return pd.read_csv(meta_path).set_index("Ticker")
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_and_calculate(tickers_tuple, weights_tuple):
    tickers = list(tickers_tuple)
    weights = list(weights_tuple)

    data_directory = os.path.join(BASE_DIR, 'data', 'raw')
    os.makedirs(data_directory, exist_ok=True)

    loader = TadawulDataLoader(tickers=tickers, data_dir=data_directory)
    loader.fetch_stock_data()
    loader.fetch_market_data()

    meta_path = os.path.join(loader.data_dir, "stocks_metadata.csv")
    if not os.path.exists(meta_path):
        loader.fetch_metadata()
    meta_df = pd.read_csv(meta_path).set_index("Ticker")

    calc = RiskCalculator(data_dir=data_directory)
    calc.load_data()
    calc.calculate_daily_returns()

    metrics = calc.calculate_portfolio_risk(weights)
    vol = metrics['Portfolio_Volatility_Percentage']
    beta = metrics['Portfolio_Beta']

    div_index = 1.0 - np.sum(np.array(weights) ** 2)

    portfolio_sectors = {}
    port_cap_score = 0.0

    for t, w in zip(tickers, weights):
        score = meta_df.loc[t, "Market_Cap_Score"] if t in meta_df.index else 2.0
        port_cap_score += w * score
        sector = meta_df.loc[t, "Sector"] if (
            t in meta_df.index and "Sector" in meta_df.columns
        ) else loader.sector_map.get(t, "Unknown")
        portfolio_sectors[sector] = portfolio_sectors.get(sector, 0.0) + w

    weighted_sector_vol = 0.0
    weighted_sector_beta = 0.0

    for sec, sec_weight in portfolio_sectors.items():
        sec_tickers = [tk for tk, s in loader.sector_map.items() if s == sec]
        s_vol, s_beta = calc.calculate_sector_metrics(sec_tickers)
        weighted_sector_vol += sec_weight * s_vol
        weighted_sector_beta += sec_weight * s_beta

    labeler = RiskLabeler()
    score_result = labeler.calculate_final_score(
        port_q_pct=vol, port_b=beta,
        sector_q=weighted_sector_vol, sector_b=weighted_sector_beta
    )

    return {
        'vol': vol,
        'beta': beta,
        'div_index': div_index,
        'port_cap_score': port_cap_score,
        'portfolio_sectors': portfolio_sectors,
        'weighted_sector_vol': weighted_sector_vol,
        'weighted_sector_beta': weighted_sector_beta,
        'score_result': score_result,
        'meta_df': meta_df,
        'sector_map': loader.sector_map,
    }


# ============================================================
# 🎨 CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #1b2838 100%);
    }

    .hero-container {
        background: linear-gradient(135deg, rgba(30,60,114,0.4), rgba(42,82,152,0.3));
        border: 1px solid rgba(100,150,255,0.15);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(100,150,255,0.05) 0%, transparent 60%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100% { transform: scale(1); opacity: .5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #60a5fa);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        margin-bottom: .5rem;
        position: relative; z-index: 1;
    }
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        color: rgba(200,210,230,0.8);
        font-size: 1.1rem;
        font-weight: 300;
        position: relative; z-index: 1;
        letter-spacing: .3px;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(96,165,250,0.2), rgba(167,139,250,0.2));
        border: 1px solid rgba(96,165,250,0.3);
        border-radius: 50px;
        padding: .3rem 1rem;
        font-size: .8rem;
        color: #93c5fd;
        margin-top: 1rem;
        position: relative; z-index: 1;
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1520, #111827) !important;
        border-right: 1px solid rgba(100,150,255,0.1);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }
    .sidebar-header {
        background: linear-gradient(135deg, rgba(30,60,114,0.5), rgba(42,82,152,0.3));
        border: 1px solid rgba(100,150,255,0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .sidebar-header h2 { font-family:'Inter',sans-serif; font-size:1.4rem; color:#93c5fd !important; margin:0; }
    .sidebar-header p  { font-family:'Inter',sans-serif; color:rgba(200,210,230,.6); font-size:.85rem; margin:.5rem 0 0 0; }

    .result-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.6), rgba(30,41,59,0.3));
        border: 1px solid rgba(100,150,255,0.12);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        transition: all .3s ease;
        position: relative;
        overflow: hidden;
    }
    .result-card:hover {
        border-color: rgba(100,150,255,0.3);
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(0,0,0,.3);
    }
    .result-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 20px 20px 0 0;
    }
    .result-card.math-card::after { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
    .result-card.ai-card::after   { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }

    .card-label {
        font-family:'Inter',sans-serif;
        color:rgba(200,210,230,.6);
        font-size:.85rem;
        text-transform:uppercase;
        letter-spacing:1.5px;
        font-weight:500;
        margin-bottom:.3rem;
    }
    .card-title {
        font-family:'Inter',sans-serif;
        color:#e2e8f0;
        font-size:1.3rem;
        font-weight:700;
        margin-bottom:1.5rem;
    }

    .score-circle {
        width:140px; height:140px;
        border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        flex-direction:column;
        margin:1rem auto;
        position:relative;
    }
    .score-circle.low    { background:radial-gradient(circle,rgba(34,197,94,.15),transparent 70%); border:3px solid rgba(34,197,94,.4);  box-shadow:0 0 30px rgba(34,197,94,.15); }
    .score-circle.medium { background:radial-gradient(circle,rgba(234,179,8,.15),transparent 70%); border:3px solid rgba(234,179,8,.4);  box-shadow:0 0 30px rgba(234,179,8,.15); }
    .score-circle.high   { background:radial-gradient(circle,rgba(239,68,68,.15),transparent 70%); border:3px solid rgba(239,68,68,.4);  box-shadow:0 0 30px rgba(239,68,68,.15); }

    .score-value { font-family:'Inter',sans-serif; font-size:2.5rem; font-weight:800; }
    .score-value.low    { color:#22c55e; }
    .score-value.medium { color:#eab308; }
    .score-value.high   { color:#ef4444; }
    .score-label-small  { font-family:'Inter',sans-serif; font-size:.7rem; color:rgba(200,210,230,.5); text-transform:uppercase; letter-spacing:1px; }

    .risk-badge {
        display:inline-block;
        padding:.5rem 1.5rem;
        border-radius:50px;
        font-family:'Inter',sans-serif;
        font-weight:600;
        font-size:.95rem;
        text-align:center;
        margin-top:.5rem;
    }
    .risk-badge.low    { background:rgba(34,197,94,.15);  border:1px solid rgba(34,197,94,.4);  color:#4ade80; }
    .risk-badge.medium { background:rgba(234,179,8,.15);  border:1px solid rgba(234,179,8,.4);  color:#facc15; }
    .risk-badge.high   { background:rgba(239,68,68,.15);  border:1px solid rgba(239,68,68,.4);  color:#f87171; }

    .metric-card {
        background: linear-gradient(135deg, rgba(30,41,59,.5), rgba(30,41,59,.2));
        border: 1px solid rgba(100,150,255,.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all .3s ease;
    }
    .metric-card:hover { border-color:rgba(100,150,255,.25); transform:translateY(-2px); }
    .metric-icon  { font-size:1.8rem; margin-bottom:.5rem; }
    .metric-value { font-family:'Inter',sans-serif; font-size:1.8rem; font-weight:700; color:#e2e8f0; }
    .metric-name  { font-family:'Inter',sans-serif; font-size:.8rem; color:rgba(200,210,230,.5); text-transform:uppercase; letter-spacing:1px; margin-top:.3rem; }

    .section-header {
        font-family:'Inter',sans-serif;
        color:#e2e8f0;
        font-size:1.5rem;
        font-weight:700;
        margin:2rem 0 1rem 0;
        display:flex;
        align-items:center;
        gap:.5rem;
    }
    .section-divider { height:1px; background:linear-gradient(90deg,transparent,rgba(100,150,255,.2),transparent); margin:1.5rem 0; }

    .sector-table { width:100%; border-collapse:separate; border-spacing:0; border-radius:12px; overflow:hidden; font-family:'Inter',sans-serif; }
    .sector-table thead th {
        background:rgba(30,41,59,.8); color:#93c5fd; padding:.8rem 1rem;
        font-size:.8rem; text-transform:uppercase; letter-spacing:1px; font-weight:600;
        border-bottom:1px solid rgba(100,150,255,.15);
    }
    .sector-table tbody td {
        padding:.7rem 1rem; color:#cbd5e1; font-size:.9rem;
        border-bottom:1px solid rgba(100,150,255,.05);
        background:rgba(15,23,42,.3);
    }
    .sector-table tbody tr:hover td { background:rgba(30,41,59,.5); }

    /* ---- Probability Bars (updated) ---- */
    .prob-section { padding: 0.5rem 0; }
    .prob-section-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 500;
        color: rgba(200,210,230,0.5);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }
    .prob-row { margin-bottom: 14px; }
    .prob-meta { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 5px; }
    .prob-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #cbd5e1;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .prob-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; display: inline-block; }
    .prob-dot.low    { background: #22c55e; }
    .prob-dot.medium { background: #eab308; }
    .prob-dot.high   { background: #ef4444; }
    .prob-pct { font-family: 'Inter', sans-serif; font-size: 0.9rem; font-weight: 600; }
    .prob-pct.low    { color: #4ade80; }
    .prob-pct.medium { color: #facc15; }
    .prob-pct.high   { color: #f87171; }
    .prob-track { height: 6px; background: rgba(30,41,59,0.8); border-radius: 3px; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 3px; transition: width 1s cubic-bezier(0.4,0,0.2,1); }
    .prob-fill.low    { background: linear-gradient(90deg, #22c55e, #4ade80); }
    .prob-fill.medium { background: linear-gradient(90deg, #eab308, #facc15); }
    .prob-fill.high   { background: linear-gradient(90deg, #ef4444, #f87171); }
    .prob-sublabel { font-family: 'Inter', sans-serif; font-size: 0.72rem; color: rgba(200,210,230,0.35); margin-top: 3px; }

    .holdings-container {
        background:linear-gradient(135deg,rgba(30,41,59,.5),rgba(30,41,59,.2));
        border:1px solid rgba(100,150,255,.1);
        border-radius:16px;
        padding:1.5rem;
        margin-top:1rem;
    }

    .stButton > button {
        background:linear-gradient(135deg,#3b82f6,#8b5cf6) !important;
        color:white !important;
        border:none !important;
        border-radius:12px !important;
        padding:.75rem 2rem !important;
        font-family:'Inter',sans-serif !important;
        font-weight:600 !important;
        font-size:1rem !important;
        transition:all .3s ease !important;
        box-shadow:0 4px 15px rgba(59,130,246,.3) !important;
    }
    .stButton > button:hover {
        transform:translateY(-2px) !important;
        box-shadow:0 6px 25px rgba(59,130,246,.5) !important;
    }

    [data-testid="stMetric"] {
        background:rgba(30,41,59,.4);
        border:1px solid rgba(100,150,255,.1);
        border-radius:12px; padding:1rem;
    }
    [data-testid="stMetricLabel"] { font-family:'Inter',sans-serif !important; }

    .success-banner {
        background:linear-gradient(135deg,rgba(34,197,94,.15),rgba(34,197,94,.05));
        border:1px solid rgba(34,197,94,.3);
        border-radius:12px;
        padding:1rem 1.5rem;
        display:flex;
        align-items:center;
        gap:.75rem;
        font-family:'Inter',sans-serif;
        color:#4ade80;
        font-weight:500;
        margin-bottom:1.5rem;
    }

    .speed-chip {
        display:inline-flex;
        align-items:center;
        gap:.4rem;
        background:rgba(139,92,246,.1);
        border:1px solid rgba(139,92,246,.25);
        border-radius:50px;
        padding:.3rem .8rem;
        font-family:'Inter',sans-serif;
        font-size:.75rem;
        color:#a78bfa;
    }

    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 🏠 HERO HEADER
# ============================================================

st.markdown("""
<div class="hero-container">
    <div class="hero-title">📈 Tadawul Portfolio Risk Analyzer</div>
    <div class="hero-subtitle">
        Predict the risk of any Saudi Stock Market portfolio using <strong>Mathematics</strong> & <strong>Artificial Intelligence</strong>
    </div>
    <div class="hero-badge">🔬 Powered by Machine Learning & Quantitative Analysis</div>
</div>
""", unsafe_allow_html=True)

# Preload model silently
load_ai_model()


# ============================================================
# 📋 SIDEBAR — Portfolio Builder
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>💼 Portfolio Builder</h2>
        <p>Configure your stock portfolio below</p>
    </div>
    """, unsafe_allow_html=True)

    num_stocks = st.number_input(
        "Number of Stocks", min_value=1, max_value=10, value=1,
        help="Select how many stocks to include in your portfolio"
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tickers = []
    weights = []

    for i in range(num_stocks):
        st.markdown(f"""
        <div style="
            font-family:'Inter',sans-serif;
            color:#93c5fd;
            font-size:.85rem;
            font-weight:600;
            margin-bottom:.3rem;
            letter-spacing:.5px;
        ">STOCK {i+1}</div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            ticker = st.text_input(
                "Ticker", value="2222.SR" if i == 0 else "",
                key=f"t_{i}", placeholder="e.g., 2222.SR",
                label_visibility="collapsed"
            )
        with col2:
            weight = st.number_input(
                "Weight %",
                min_value=1.0, max_value=100.0,
                value=round(100.0 / num_stocks, 1),
                key=f"w_{i}",
                label_visibility="collapsed"
            )

        if ticker:
            t_upper = ticker.upper().strip()
            if not t_upper.endswith('.SR'):
                t_upper += '.SR'
            tickers.append(t_upper)
            weights.append(weight / 100.0)

    total_weight = sum(weights) * 100 if weights else 0
    is_valid = abs(total_weight - 100) < 1
    weight_color = "#4ade80" if is_valid else "#f87171"
    check_icon = "✓" if is_valid else "✗"

    st.markdown(f"""
    <div style="
        background:rgba(30,41,59,.5);
        border:1px solid {'rgba(34,197,94,.3)' if is_valid else 'rgba(239,68,68,.3)'};
        border-radius:10px;
        padding:.8rem;
        text-align:center;
        margin:1rem 0;
        font-family:'Inter',sans-serif;
        transition: all .3s ease;
    ">
        <span style="color:rgba(200,210,230,.5); font-size:.75rem; text-transform:uppercase; letter-spacing:1px;">
            Total Weight {check_icon}
        </span><br>
        <span style="color:{weight_color}; font-size:1.5rem; font-weight:700;">
            {total_weight:.1f}%
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    analyze_button = st.button("🚀 Analyze Portfolio Risk", use_container_width=True)

    st.markdown("""
    <div style="
        margin-top:2rem; padding-top:1rem;
        border-top:1px solid rgba(100,150,255,.1);
        text-align:center;
        font-family:'Inter',sans-serif;
        color:rgba(200,210,230,.3);
        font-size:.7rem;
    ">
        Tadawul Risk AI v2.0<br>
        Built with ❤️ using Streamlit<br>
        <span style="color:rgba(139,92,246,.5);">⚡ Cache-accelerated</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 🔧 HELPER FUNCTIONS
# ============================================================

def get_risk_class(category):
    cat = str(category)
    if "Low" in cat: return "low"
    if "Medium" in cat: return "medium"
    return "high"

def get_risk_emoji(category):
    cat = str(category)
    if "Low" in cat: return "🟢"
    if "Medium" in cat: return "🟡"
    return "🔴"

def get_risk_description(category):
    cat = str(category)
    if "Low" in cat:
        return "This portfolio shows conservative risk characteristics. Suitable for risk-averse investors."
    if "Medium" in cat:
        return "This portfolio has moderate risk exposure. A balanced approach for most investors."
    return "This portfolio exhibits high risk levels. Only suitable for aggressive, experienced investors."

def get_prob_sublabel(cat):
    cat = str(cat)
    if "Low" in cat:
        return "High confidence — conservative portfolio characteristics"
    if "Medium" in cat:
        return "Marginal — moderate exposure detected"
    return "Elevated — aggressive risk signals present"


# ============================================================
# 🚀 MAIN ANALYSIS LOGIC
# ============================================================

if analyze_button:
    if abs(sum(weights) - 1.0) > 0.01:
        st.sidebar.error("⚠️ Total weights must equal 100%!")
    elif len(tickers) == 0:
        st.sidebar.error("⚠️ Please enter at least one stock.")
    else:
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.markdown("""
            <div style="font-family:'Inter',sans-serif; color:#93c5fd; font-size:.95rem; padding:.5rem;">
                ⏳ Fetching market data from Tadawul...
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(15)

            results = fetch_and_calculate(tuple(tickers), tuple(weights))

            progress_bar.progress(60)

            vol = results['vol']
            beta = results['beta']
            div_index = results['div_index']
            port_cap_score = results['port_cap_score']
            portfolio_sectors = results['portfolio_sectors']
            weighted_sector_vol = results['weighted_sector_vol']
            weighted_sector_beta = results['weighted_sector_beta']
            score_result = results['score_result']
            meta_df = results['meta_df']
            sector_map = results['sector_map']

            status_text.markdown("""
            <div style="font-family:'Inter',sans-serif; color:#a78bfa; font-size:.95rem; padding:.5rem;">
                🤖 Running AI prediction...
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(80)

            model = load_ai_model()
            if model is not None:
                input_df = pd.DataFrame(
                    [[vol, beta, weighted_sector_vol * 100, weighted_sector_beta, div_index, port_cap_score]],
                    columns=[
                        'Portfolio_Volatility', 'Portfolio_Beta',
                        'Sector_Volatility', 'Sector_Beta',
                        'Diversification_Index', 'Market_Cap_Score'
                    ]
                )
                ai_category = model.predict(input_df)[0]
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_df)[0]
                    prob_dict = {c: round(p * 100) for c, p in zip(model.classes_, probs)}
                else:
                    prob_dict = {}
            else:
                ai_category = "Model Not Found"
                prob_dict = {}

            progress_bar.progress(100)
            elapsed = time.time() - start_time
            status_text.empty()
            progress_bar.empty()


            # ============================================================
            # 📊 DISPLAY RESULTS
            # ============================================================

            st.markdown(f"""
            <div class="success-banner">
                <span style="font-size:1.3rem;">✅</span>
                <span>Analysis Complete — Results generated in <strong>{elapsed:.1f}s</strong></span>
                {'<span class="speed-chip">⚡ Cached</span>' if elapsed < 2 else ''}
            </div>
            """, unsafe_allow_html=True)

            math_class = get_risk_class(score_result['Risk_Category'])
            ai_class   = get_risk_class(ai_category)
            math_emoji = get_risk_emoji(score_result['Risk_Category'])
            ai_emoji   = get_risk_emoji(ai_category)

            col1, col2 = st.columns(2, gap="large")

            with col1:
                risk_score = score_result['Final_Risk_Score']
                st.markdown(f"""
                <div class="result-card math-card">
                    <div class="card-label">Mathematical Model</div>
                    <div class="card-title">🧮 Quantitative Analysis</div>
                    <div class="score-circle {math_class}">
                        <div class="score-value {math_class}">{risk_score}</div>
                        <div class="score-label-small">Risk Score</div>
                    </div>
                    <div style="text-align:center; margin-top:1rem;">
                        <div class="risk-badge {math_class}">
                            {math_emoji} {score_result['Risk_Category']}
                        </div>
                    </div>
                    <div style="
                        margin-top:1.2rem; padding:.8rem;
                        background:rgba(15,23,42,.3); border-radius:10px;
                        font-family:'Inter',sans-serif; color:rgba(200,210,230,.5);
                        font-size:.8rem; text-align:center;
                    ">
                        {get_risk_description(score_result['Risk_Category'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                ai_confidence = prob_dict.get(ai_category, 0)

                bars_html = ""
                if prob_dict:
                    for cat, pct in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
                        bar_class = get_risk_class(cat)
                        sublabel = get_prob_sublabel(cat)
                        bars_html += (
                            '<div class="prob-row">'
                            '<div class="prob-meta">'
                            '<span class="prob-label">'
                            f'<span class="prob-dot {bar_class}"></span>{cat}'
                            '</span>'
                            f'<span class="prob-pct {bar_class}">{pct}%</span>'
                            '</div>'
                            f'<div class="prob-track"><div class="prob-fill {bar_class}" style="width:{pct}%;"></div></div>'
                            f'<div class="prob-sublabel">{sublabel}</div>'
                            '</div>'
                        )
                    prob_block = (
                        '<div style="margin-top:1.2rem;padding:1rem;background:rgba(15,23,42,.3);border-radius:10px;">'
                        '<div class="prob-section-title">Probability Distribution</div>'
                        f'<div class="prob-section">{bars_html}</div>'
                        '</div>'
                    )
                else:
                    prob_block = ""

                st.markdown(
                    f'<div class="result-card ai-card">'
                    f'<div class="card-label">Machine Learning</div>'
                    f'<div class="card-title">🤖 AI Prediction</div>'
                    f'<div class="score-circle {ai_class}">'
                    f'<div class="score-value {ai_class}">{ai_confidence}%</div>'
                    f'<div class="score-label-small">Confidence</div>'
                    f'</div>'
                    f'<div style="text-align:center;margin-top:1rem;">'
                    f'<div class="risk-badge {ai_class}">{ai_emoji} {ai_category}</div>'
                    f'</div>'
                    f'{prob_block}'
                    f'</div>',
                    unsafe_allow_html=True
                )


            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


            # --- Portfolio Metrics ---
            st.markdown('<div class="section-header">📊 Portfolio Metrics</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4, gap="medium")
            metrics_data = [
                ("📉", "Volatility",      f"{vol:.2f}%",            m1),
                ("⚖️", "Beta",            f"{beta:.2f}",            m2),
                ("🔀", "Diversification",  f"{div_index:.2f}",       m3),
                ("🏢", "Market Cap Score", f"{port_cap_score:.2f}",  m4),
            ]
            for icon, name, value, col in metrics_data:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-icon">{icon}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-name">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)


            # --- Portfolio Holdings ---
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📋 Portfolio Holdings</div>', unsafe_allow_html=True)

            holdings_rows = ""
            for t, w in zip(tickers, weights):
                sector = meta_df.loc[t, "Sector"] if (
                    t in meta_df.index and "Sector" in meta_df.columns
                ) else sector_map.get(t, "Unknown")
                cap_score = meta_df.loc[t, "Market_Cap_Score"] if t in meta_df.index else "N/A"
                bar_width = w * 100

                holdings_rows += f"""
                <tr>
                    <td style="font-weight:600; color:#93c5fd;">{t}</td>
                    <td>{sector}</td>
                    <td>
                        <div style="display:flex; align-items:center; gap:.5rem;">
                            <div style="flex:1; height:6px; background:rgba(30,41,59,.8); border-radius:3px; overflow:hidden;">
                                <div style="width:{bar_width}%; height:100%; background:linear-gradient(90deg,#3b82f6,#8b5cf6); border-radius:3px;"></div>
                            </div>
                            <span style="font-weight:600; min-width:45px;">{w*100:.1f}%</span>
                        </div>
                    </td>
                    <td style="text-align:center;">{cap_score}</td>
                </tr>
                """

            st.markdown(f"""
            <div class="holdings-container">
                <table class="sector-table">
                    <thead><tr>
                        <th>Ticker</th><th>Sector</th><th>Weight</th><th>Cap Score</th>
                    </tr></thead>
                    <tbody>{holdings_rows}</tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)


            # --- Sector Exposure ---
            if portfolio_sectors:
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🏭 Sector Exposure</div>', unsafe_allow_html=True)

                sector_cols = st.columns(len(portfolio_sectors), gap="medium")
                colors = ["#3b82f6","#8b5cf6","#06b6d4","#10b981","#f59e0b","#ef4444","#ec4899"]

                for idx, (sec, sec_w) in enumerate(
                    sorted(portfolio_sectors.items(), key=lambda x: x[1], reverse=True)
                ):
                    color = colors[idx % len(colors)]
                    with sector_cols[idx]:
                        st.markdown(f"""
                        <div class="metric-card" style="border-color:{color}30;">
                            <div style="font-family:'Inter',sans-serif; font-size:1.5rem; font-weight:700; color:{color};">
                                {sec_w*100:.1f}%
                            </div>
                            <div class="metric-name" style="font-size:.7rem;">{sec}</div>
                        </div>
                        """, unsafe_allow_html=True)


        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.markdown(f"""
            <div style="
                background:rgba(239,68,68,.1);
                border:1px solid rgba(239,68,68,.3);
                border-radius:12px;
                padding:1.5rem;
                font-family:'Inter',sans-serif;
            ">
                <div style="color:#f87171; font-weight:600; margin-bottom:.5rem;">❌ An Error Occurred</div>
                <div style="color:rgba(200,210,230,.6); font-size:.9rem;">{str(e)}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# 🏠 EMPTY STATE
# ============================================================

else:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; margin-top:2rem;">
        <div style="font-size:4rem; margin-bottom:1rem; opacity:.3;">🔍</div>
        <div style="
            font-family:'Inter',sans-serif;
            color:rgba(200,210,230,.4);
            font-size:1.2rem;
            font-weight:500;
        ">Configure your portfolio in the sidebar and click <strong>Analyze</strong></div>
        <div style="
            font-family:'Inter',sans-serif;
            color:rgba(200,210,230,.25);
            font-size:.9rem;
            margin-top:.5rem;
        ">Add stock tickers with their weights to get started</div>
    </div>

    <div style="display:flex; justify-content:center; gap:2rem; margin-top:3rem; flex-wrap:wrap;">
        <div style="
            background:rgba(30,41,59,.3);
            border:1px solid rgba(100,150,255,.08);
            border-radius:16px;
            padding:1.5rem 2rem;
            text-align:center;
            width:200px;
        ">
            <div style="font-size:2rem; margin-bottom:.5rem;">📊</div>
            <div style="font-family:'Inter',sans-serif; color:#93c5fd; font-size:.9rem; font-weight:600;">Quantitative Analysis</div>
            <div style="font-family:'Inter',sans-serif; color:rgba(200,210,230,.3); font-size:.75rem; margin-top:.3rem;">Volatility, Beta & more</div>
        </div>
        <div style="
            background:rgba(30,41,59,.3);
            border:1px solid rgba(100,150,255,.08);
            border-radius:16px;
            padding:1.5rem 2rem;
            text-align:center;
            width:200px;
        ">
            <div style="font-size:2rem; margin-bottom:.5rem;">🤖</div>
            <div style="font-family:'Inter',sans-serif; color:#a78bfa; font-size:.9rem; font-weight:600;">AI Prediction</div>
            <div style="font-family:'Inter',sans-serif; color:rgba(200,210,230,.3); font-size:.75rem; margin-top:.3rem;">Machine Learning Model</div>
        </div>
        <div style="
            background:rgba(30,41,59,.3);
            border:1px solid rgba(100,150,255,.08);
            border-radius:16px;
            padding:1.5rem 2rem;
            text-align:center;
            width:200px;
        ">
            <div style="font-size:2rem; margin-bottom:.5rem;">🛡️</div>
            <div style="font-family:'Inter',sans-serif; color:#34d399; font-size:.9rem; font-weight:600;">Risk Classification</div>
            <div style="font-family:'Inter',sans-serif; color:rgba(200,210,230,.3); font-size:.75rem; margin-top:.3rem;">Low, Medium, or High</div>
        </div>
    </div>
    """, unsafe_allow_html=True)