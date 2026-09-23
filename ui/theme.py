"""Local styles for the green finance dashboard. No remote assets."""
import streamlit as st


def apply_theme():
    st.markdown("""<style>
    .stApp { background: #f5f7f6; color: #19291f; }
    .block-container { max-width: 1440px; padding: 2rem 2.5rem 3rem; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    h1, h2, h3 { letter-spacing: -.035em; color: #19291f; }
    [data-testid="stVerticalBlock"] { gap: 1rem; }
    .brandbar { display: flex; justify-content: space-between; align-items: center;
        gap: 16px; padding: 8px 0 22px; border-bottom: 1px solid #dfe7e1; }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brandmark { display: grid; place-items: center; background: #16853b; color: white;
        width: 44px; height: 44px; border-radius: 14px; font-size: 28px; font-weight: 600; }
    .brandname { font-size: 20px; font-weight: 750; letter-spacing: -.5px; line-height: 1.2; }
    .brandmeta { color: #6a7d70; font-size: 11px; letter-spacing: 1.4px; margin-top: 4px; }
    .workspace-badge { color: #43634b; border: 1px solid #d7e3da; border-radius: 30px;
        padding: 8px 14px; font-size: 12px; background: #fff; white-space: nowrap; }
    .workspace-badge i { display: inline-block; width: 7px; height: 7px;
        background: #16853b; border-radius: 50%; margin-right: 7px; }
    .hero { position: relative; overflow: hidden; padding: 28px 32px;
        background: #e5f3dc; border: 1px solid #d8ebd0; border-radius: 24px;
        display: flex; align-items: center; justify-content: space-between; gap: 20px; }
    .hero-copy { max-width: 680px; position: relative; z-index: 1; }
    .eyebrow { color: #42764a; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; }
    .hero h1 { font-size: clamp(27px, 3.5vw, 42px); line-height: 1.13; margin: 10px 0;
        padding: 0; font-weight: 650; letter-spacing: -1.6px; }
    .hero p { color: #48614c; font-size: 14px; margin: 0; max-width: 550px; line-height: 1.65; }
    .hero-visual { flex: 0 0 180px; width: 180px; height: 120px; }
    .section-kicker { color: #687b6e; font-size: 11px; font-weight: 650;
        letter-spacing: 1.3px; margin: 3px 0 0; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #e1e8e3;
        padding: 20px; border-radius: 18px; min-height: 112px; }
    [data-testid="stMetricLabel"] { color: #697d70; font-size: 12px; }
    [data-testid="stMetricValue"] { color: #19291f; font-size: clamp(23px, 2.3vw, 33px);
        font-weight: 650; letter-spacing: -1px; }
    .st-key-overview [data-testid="stColumn"]:last-child [data-testid="stMetric"] {
        background: #16853b; border-color: #16853b; }
    .st-key-overview [data-testid="stColumn"]:last-child [data-testid="stMetricLabel"],
    .st-key-overview [data-testid="stColumn"]:last-child [data-testid="stMetricValue"] { color: white; }
    [data-baseweb="tab-list"] { gap: 8px; padding: 6px; background: #eaf0ec;
        border-radius: 15px; flex-wrap: wrap; }
    button[data-baseweb="tab"] { border-radius: 10px; padding: 10px 20px; height: 43px;
        color: #62776a; background: transparent; border: none; }
    button[data-baseweb="tab"][aria-selected="true"] { background: #fff; color: #16853b;
        box-shadow: 0 2px 7px #183d2110; font-weight: 650; }
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }
    [data-baseweb="tab-panel"] { padding-top: 20px; }
    [data-testid="stTextInputRootElement"], [data-baseweb="select"] > div {
        background: #fff; border-color: #dce6df; border-radius: 12px; }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
        border-radius: 11px; border-color: #cfe1d4; color: #176d33; background: #fff;
        min-height: 42px; font-weight: 600; }
    [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
        background: #e6f3e9; color: #125e2a; border-color: #16853b; }
    [data-testid="stAlert"] { border-radius: 12px; font-size: 13px; }
    [data-testid="stAlert"][data-baseweb="notification"] { border: 0; }
    [data-testid="stExpander"] { background: white; border-radius: 14px; }
    [data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
    iframe { border: 1px solid #dce6df; border-radius: 18px; background: #fff; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
    .legend-item { display: inline-flex; align-items: center; gap: 7px; font-size: 11px;
        color: #4c6153; border: 1px solid #e0e8e2; background: #fff;
        padding: 6px 10px; border-radius: 30px; }
    .legend-dot { width: 7px; height: 7px; border-radius: 50%; }
    .footer-note { font-size: 11px; color: #718077; margin-top: 24px;
        padding-top: 18px; border-top: 1px solid #dfe7e1; line-height: 1.7; }
    @media (max-width: 1050px) {
        .st-key-overview [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        .st-key-overview [data-testid="stColumn"] { min-width: calc(50% - 1rem); flex: 1; }
        .hero-visual { flex-basis: 140px; width: 140px; }
    }
    @media (max-width: 640px) {
        .block-container { padding: 1.2rem 1rem 2rem; }
        .hero { padding: 24px; border-radius: 18px; }
        .hero-visual, .workspace-badge { display: none; }
        .hero h1 { letter-spacing: -.8px; }
        button[data-baseweb="tab"] { padding: 8px 10px; font-size: 12px; }
        [data-testid="stMetric"] { padding: 14px; min-height: 95px; }
    }
    </style>""", unsafe_allow_html=True)


def header():
    st.markdown('''<div class="brandbar">
    <div class="brand"><div class="brandmark">↗</div><div>
    <div class="brandname">Ақша графы</div><div class="brandmeta">HACKALEM AI · FINANCIAL INTELLIGENCE</div>
    </div></div><div class="workspace-badge"><i></i>Аналитиктің жұмыс кеңістігі</div></div>''', unsafe_allow_html=True)
    st.markdown('''<div class="hero"><div class="hero-copy">
    <div class="eyebrow">БАЙЛАНЫСТАРДАН — ТҮСІНІККЕ</div>
    <h1>Ақша қозғалысы.<br>Айқын көрініс.</h1>
    <p>Қаржы желісін зерттеңіз, маңызды байланыстарды табыңыз<br>және тексеру басымдығын деректермен негіздеңіз.</p></div>
    <svg class="hero-visual" viewBox="0 0 180 120" fill="none" aria-hidden="true">
    <circle cx="90" cy="60" r="56" stroke="#c1dcb5" stroke-dasharray="4 5"/>
    <path d="M28 28L90 60L155 23M90 60L148 100M90 60L25 98M90 60L95 7" stroke="#94bb85" stroke-width="2"/>
    <circle cx="90" cy="60" r="27" fill="#16853b"/><path d="M77 69L102 47M83 47H102V66" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="28" cy="28" r="12" fill="white" stroke="#9fc895"/><circle cx="155" cy="23" r="15" fill="#9ecf79"/>
    <circle cx="148" cy="100" r="11" fill="white" stroke="#9fc895"/><circle cx="25" cy="98" r="15" fill="#9ecf79"/>
    <circle cx="95" cy="7" r="6" fill="#16853b"/></svg></div>''', unsafe_allow_html=True)
