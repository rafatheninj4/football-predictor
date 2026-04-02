# football_predictor.py (versão deploy)
"""
⚽ Football Predictive Analytics — Value Bet Identifier + Bankroll Manager
Versão Web — Streamlit Cloud Ready
"""

import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd
import math
from datetime import date
import json

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FootballEdge — Value Bet Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# BANKROLL — Usando session_state (funciona na web)
# ──────────────────────────────────────────────
def init_bankroll():
    if "bets_history" not in st.session_state:
        st.session_state.bets_history = []
    if "bankroll" not in st.session_state:
        st.session_state.bankroll = 1000.0

def save_bet(bet_data):
    init_bankroll()
    st.session_state.bets_history.append(bet_data)

def load_bets():
    init_bankroll()
    return pd.DataFrame(st.session_state.bets_history)

def delete_bet(idx):
    init_bankroll()
    if idx < len(st.session_state.bets_history):
        st.session_state.bets_history.pop(idx)

# ──────────────────────────────────────────────
# CUSTOM CSS (seu CSS original)
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0e1a; }
.main .block-container { padding-top: 2rem; max-width: 1400px; }

section[data-testid="stSidebar"] {
    background: #0f1422;
    border-right: 1px solid #1e2540;
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #e8f4fd 0%, #7eb8f7 50%, #4a9eff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #4a5680;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.section-header {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: #c8d8f0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-left: 3px solid #4a9eff;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}
.metric-card {
    background: #111827;
    border: 1px solid #1e2d47;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #4a5680;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    color: #e8f0fe;
}
.metric-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6b7db3;
    margin-top: 0.2rem;
}
.badge-value {
    background: #0d2818; border: 1px solid #1a5c32; color: #4ade80;
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    padding: 2px 10px; border-radius: 20px; font-weight: 500;
}
.badge-novalue {
    background: #1a1422; border: 1px solid #3d1a2e; color: #f87171;
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    padding: 2px 10px; border-radius: 20px;
}
.comp-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.comp-table th {
    font-family: 'DM Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.08em; text-transform: uppercase; color: #4a5680;
    padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #1e2540;
}
.comp-table td { padding: 0.8rem 1rem; color: #c8d8f0; border-bottom: 1px solid #111827; }
.comp-table tr.value-row td { background: #0d1f14; color: #86efac; }
.comp-table tr.value-row td:first-child { border-left: 3px solid #22c55e; }
.kelly-result {
    background: linear-gradient(135deg, #0d1f3c 0%, #0f1a30 100%);
    border: 1px solid #1e3a6e; border-radius: 14px;
    padding: 1.5rem 2rem; text-align: center;
}
.kelly-stake { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 3rem; color: #60a5fa; }
.kelly-label {
    font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #4a5680;
    letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.3rem;
}
.info-alert {
    background: #0d1c35; border: 1px solid #1e3a6e; border-radius: 10px;
    padding: 0.9rem 1.2rem; font-size: 0.83rem; color: #7eb8f7;
    font-family: 'DM Mono', monospace; line-height: 1.6;
}
hr.custom { border: none; border-top: 1px solid #1e2540; margin: 1.5rem 0; }
.poisson-table { font-family: 'DM Mono', monospace; font-size: 0.72rem; border-collapse: separate; border-spacing: 3px; }
.poisson-table td, .poisson-table th { width: 42px; height: 38px; text-align: center; border-radius: 5px; }
.poisson-table th { background: #111827; color: #4a5680; font-weight: 500; }
.stat-card {
    background: #111827; border: 1px solid #1e2d47; border-radius: 12px;
    padding: 1.1rem 1.3rem; text-align: center;
}
.stat-val { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.6rem; }
.stat-lbl { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #4a5680; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CORE FUNCTIONS (seu código original)
# ──────────────────────────────────────────────
LEAGUE_AVG_HOME = 1.55
LEAGUE_AVG_AWAY = 1.10

def calculate_strengths(home_scored, home_conceded, away_scored, away_conceded, n=5):
    home_attack  = (home_scored  / n) / LEAGUE_AVG_HOME
    home_defense = (home_conceded/ n) / LEAGUE_AVG_AWAY
    away_attack  = (away_scored  / n) / LEAGUE_AVG_AWAY
    away_defense = (away_conceded/ n) / LEAGUE_AVG_HOME
    return home_attack, home_defense, away_attack, away_defense

def expected_goals(home_attack, home_defense, away_attack, away_defense):
    home_xg = home_attack * away_defense * LEAGUE_AVG_HOME
    away_xg = away_attack * home_defense * LEAGUE_AVG_AWAY
    return round(home_xg, 3), round(away_xg, 3)

def poisson_matrix(home_xg, away_xg, max_goals=6):
    matrix = np.zeros((max_goals+1, max_goals+1))
    for h in range(max_goals+1):
        for a in range(max_goals+1):
            matrix[h][a] = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
    return matrix

def outcome_probabilities(matrix):
    n = matrix.shape[0]
    home_win = draw = away_win = 0.0
    for h in range(n):
        for a in range(n):
            if h > a:   home_win += matrix[h][a]
            elif h == a: draw    += matrix[h][a]
            else:        away_win += matrix[h][a]
    return home_win, draw, away_win

def over_under_25(matrix):
    n = matrix.shape[0]
    over = under = 0.0
    for h in range(n):
        for a in range(n):
            if h + a > 2: over  += matrix[h][a]
            else:          under += matrix[h][a]
    return over, under

def prob_to_odd(prob):
    if prob <= 0: return float('inf')
    return round(1.0 / prob, 3)

def expected_value(model_prob, bookie_odd):
    return round((model_prob * bookie_odd) - 1.0, 4)

def kelly_fraction(model_prob, bookie_odd, fraction=0.25):
    b = bookie_odd - 1.0
    p = model_prob
    q = 1.0 - p
    if b <= 0:
        return 0.0
    k = (b * p - q) / b
    k = max(0.0, min(k, 0.25))  # Limita a 25% máximo
    return round(k * fraction, 4)

# ──────────────────────────────────────────────
# SIDEBAR — INPUTS
# ──────────────────────────────────────────────
init_bankroll()

with st.sidebar:
    st.markdown('<div class="hero-title">⚽ Edge</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Football Predictive Analytics</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Match Setup</div>', unsafe_allow_html=True)
    home_team = st.text_input("Home Team", value="Real Madrid", key="home_name")
    away_team = st.text_input("Away Team", value="Barcelona",   key="away_name")

    st.markdown('<hr class="custom">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Last 5 Games — Home</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: home_scored   = st.number_input("Goals Scored",   min_value=0, max_value=30, value=9, key="hs")
    with col2: home_conceded = st.number_input("Goals Conceded", min_value=0, max_value=30, value=4, key="hc")

    st.markdown('<div class="section-header">Last 5 Games — Away</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3: away_scored   = st.number_input("Goals Scored",   min_value=0, max_value=30, value=7, key="as_")
    with col4: away_conceded = st.number_input("Goals Conceded", min_value=0, max_value=30, value=6, key="ac")

    st.markdown('<hr class="custom">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Bookmaker Odds</div>', unsafe_allow_html=True)
    bookie_home  = st.number_input("Home Win Odd",  min_value=1.01, max_value=50.0, value=2.10, step=0.05, format="%.2f")
    bookie_draw  = st.number_input("Draw Odd",      min_value=1.01, max_value=50.0, value=3.40, step=0.05, format="%.2f")
    bookie_away  = st.number_input("Away Win Odd",  min_value=1.01, max_value=50.0, value=3.20, step=0.05, format="%.2f")
    bookie_over  = st.number_input("Over 2.5 Odd",  min_value=1.01, max_value=20.0, value=1.85, step=0.05, format="%.2f")
    bookie_under = st.number_input("Under 2.5 Odd", min_value=1.01, max_value=20.0, value=2.05, step=0.05, format="%.2f")

    st.markdown('<hr class="custom">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Bankroll (Kelly)</div>', unsafe_allow_html=True)
    bankroll   = st.number_input("Current Bankroll (R$)", min_value=10.0, max_value=1_000_000.0, value=st.session_state.bankroll, step=50.0, format="%.2f")
    st.session_state.bankroll = bankroll
    kelly_frac = st.select_slider("Kelly Fraction", options=[0.10, 0.25, 0.50, 1.00],
                                   value=0.25, format_func=lambda x: f"{int(x*100)}%")
    st.markdown('<div class="info-alert">⚠ Quarter Kelly (25%) é o padrão para apostadores profissionais.</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# COMPUTE
# ──────────────────────────────────────────────
ha, hd, aa, ad = calculate_strengths(home_scored, home_conceded, away_scored, away_conceded)
home_xg, away_xg = expected_goals(ha, hd, aa, ad)
matrix = poisson_matrix(home_xg, away_xg)
p_home, p_draw, p_away = outcome_probabilities(matrix)
p_over, p_under = over_under_25(matrix)

odd_model_home  = prob_to_odd(p_home)
odd_model_draw  = prob_to_odd(p_draw)
odd_model_away  = prob_to_odd(p_away)
odd_model_over  = prob_to_odd(p_over)
odd_model_under = prob_to_odd(p_under)

ev_home  = expected_value(p_home,  bookie_home)
ev_draw  = expected_value(p_draw,  bookie_draw)
ev_away  = expected_value(p_away,  bookie_away)
ev_over  = expected_value(p_over,  bookie_over)
ev_under = expected_value(p_under, bookie_under)

k_home  = kelly_fraction(p_home,  bookie_home,  kelly_frac)
k_draw  = kelly_fraction(p_draw,  bookie_draw,  kelly_frac)
k_away  = kelly_fraction(p_away,  bookie_away,  kelly_frac)
k_over  = kelly_fraction(p_over,  bookie_over,  kelly_frac)
k_under = kelly_fraction(p_under, bookie_under, kelly_frac)

rows = [
    (f"{home_team} Win",  round(p_home*100,1),  odd_model_home,  bookie_home,  ev_home,  k_home),
    ("Draw",              round(p_draw*100,1),  odd_model_draw,  bookie_draw,  ev_draw,  k_draw),
    (f"{away_team} Win",  round(p_away*100,1),  odd_model_away,  bookie_away,  ev_away,  k_away),
    ("Over 2.5",          round(p_over*100,1),  odd_model_over,  bookie_over,  ev_over,  k_over),
    ("Under 2.5",         round(p_under*100,1), odd_model_under, bookie_under, ev_under, k_under),
]

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
st.markdown('<div class="hero-title">Football Edge Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Poisson Distribution Model · Kelly Criterion · Value Bet Detector</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["⚽  Análise do Jogo", "📊  Gestão de Banca"])

# ═══════════════════════════════════════════════
# TAB 1 — ANÁLISE DO JOGO (seu código original)
# ═══════════════════════════════════════════════
with tab1:
    st.markdown(f"### {home_team}  vs  {away_team}")
    st.markdown('<hr class="custom">', unsafe_allow_html=True)

    # XG cards
    st.markdown('<div class="section-header">Model Output — Expected Goals</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Home xG</div><div class="metric-value" style="color:#60a5fa">{home_xg}</div><div class="metric-sub">{home_team}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Away xG</div><div class="metric-value" style="color:#f87171">{away_xg}</div><div class="metric-sub">{away_team}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Home Attack Str.</div><div class="metric-value" style="color:#a78bfa">{round(ha,3)}</div><div class="metric-sub">vs. league avg 1.0</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Away Attack Str.</div><div class="metric-value" style="color:#fb923c">{round(aa,3)}</div><div class="metric-sub">vs. league avg 1.0</div></div>', unsafe_allow_html=True)

    # Outcome probabilities
    st.markdown('<hr class="custom">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Outcome Probabilities</div>', unsafe_allow_html=True)

    def prob_bar_html(label, prob, color):
        pct = round(prob * 100, 1)
        return f"""<div style="margin-bottom:0.9rem;">
        <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#c8d8f0;margin-bottom:4px;">
            <span>{label}</span><span style="color:{color};font-weight:500;">{pct}%</span>
        </div>
        <div style="background:#0a0e1a;border-radius:6px;height:10px;overflow:hidden;">
            <div style="width:{pct}%;background:{color};height:100%;border-radius:6px;"></div>
        </div></div>"""

    col_prob, col_ou = st.columns([3, 2])
    with col_prob:
        bars = prob_bar_html(f"{home_team} Win", p_home, "#60a5fa") + prob_bar_html("Draw", p_draw, "#a78bfa") + prob_bar_html(f"{away_team} Win", p_away, "#f87171")
        st.markdown(f'<div style="background:#111827;border:1px solid #1e2d47;border-radius:12px;padding:1.4rem 1.6rem;">{bars}</div>', unsafe_allow_html=True)
    with col_ou:
        ou_bars = prob_bar_html("Over 2.5 Goals", p_over, "#4ade80") + prob_bar_html("Under 2.5 Goals", p_under, "#facc15")
        st.markdown(f'<div style="background:#111827;border:1px solid #1e2d47;border-radius:12px;padding:1.4rem 1.6rem;">{ou_bars}</div>', unsafe_allow_html=True)

    # Value table
    st.markdown('<hr class="custom">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Value Bet Comparison Table</div>', unsafe_allow_html=True)

    def ev_badge(ev):
        if ev > 0: return f'<span class="badge-value">+{round(ev*100,1)}% EV ✓</span>'
        return f'<span class="badge-novalue">{round(ev*100,1)}% EV ✗</span>'

    table_html = """<table class="comp-table"><thead><tr><th>Market</th><th>Model Prob.</th><th>Model Odd (Fair)</th><th>Bookie Odd</th><th>Expected Value</th><th>Kelly Stake</th></tr></thead><tbody>"""
    for label, prob, odd_m, odd_b, ev, kel in rows:
        rc = "value-row" if ev > 0 else ""
        stake_val = round(kel * bankroll, 2)
        table_html += f"""<tr class="{rc}"><td><strong>{label}</strong></td><td>{prob}%</td><td style="font-family:'DM Mono',monospace;">{odd_m}</td><td style="font-family:'DM Mono',monospace;">{odd_b}</td><td>{ev_badge(ev)}</td><td style="font-family:'DM Mono',monospace;color:#60a5fa;">R$ {stake_val:,.2f} <span style="color:#4a5680;font-size:0.72rem;">({round(kel*100,1)}%)</span></td></tr>"""
    table_html += "</tbody></table>"
    st.markdown(f'<div style="background:#111827;border:1px solid #1e2d47;border-radius:12px;padding:1.2rem 1.4rem;overflow-x:auto;">{table_html}</div>', unsafe_allow_html=True)

    value_bets = [(l, ev, k, ob, pm) for l, pm, om, ob, ev, k in rows if ev > 0]
    if value_bets:
        best = max(value_bets, key=lambda x: x[1])
        best_label, best_ev, best_k, best_odd, _ = best
        best_stake = round(best_k * bankroll, 2)
        st.success(f"🟢 **Best Value Bet:** {best_label} @ {best_odd} — EV: **+{round(best_ev*100,1)}%** — Stake sugerida: **R$ {best_stake:,.2f}**")
    else:
        st.warning("⚠️ Nenhum EV positivo encontrado neste jogo. Considere pular.")

    # Heatmap
    st.markdown('<hr class="custom">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Scoreline Probability Heatmap</div>', unsafe_allow_html=True)

    max_show = 6
    heatmap_matrix = poisson_matrix(home_xg, away_xg, max_goals=max_show)
    top_scores = [(heatmap_matrix[h][a], h, a) for h in range(max_show+1) for a in range(max_show+1)]
    top_scores.sort(reverse=True)
    top_3 = set((h, a) for _, h, a in top_scores[:3])
    max_p = heatmap_matrix.max()

    def cell_color(prob, max_p):
        intensity = prob / max_p if max_p > 0 else 0
        r = int(10 + intensity * 20)
        g = int(20 + intensity * 80)
        b = int(40 + intensity * 180)
        alpha = 0.25 + intensity * 0.75
        return f"rgba({r},{g},{b},{alpha:.2f})"

    hmap_html = f'<div style="overflow-x:auto;"><table class="poisson-table"><thead><tr><th style="background:transparent;color:#4a5680;">H \\ A</th>'
    for a in range(max_show+1):
        hmap_html += f'<th>{away_team[:3]} {a}</th>'
    hmap_html += "</tr></thead><tbody>"
    for h in range(max_show+1):
        hmap_html += f'<tr><th style="background:#111827;color:#4a5680;text-align:left;padding:4px 8px;">{home_team[:3]} {h}</th>'
        for a in range(max_show+1):
            prob = heatmap_matrix[h][a]
            pct  = round(prob * 100, 1)
            bg   = cell_color(prob, max_p)
            border = "2px solid #4ade80" if (h, a) in top_3 else "none"
            hmap_html += f'<td style="background:{bg};color:#e8f0fe;border:{border};border-radius:5px;">{pct}%</td>'
        hmap_html += "</tr>"
    hmap_html += "</tbody></table></div>"
    st.markdown(f'<div style="background:#111827;border:1px solid #1e2d47;border-radius:12px;padding:1.4rem;">{hmap_html}<p style="font-family:DM Mono,monospace;font-size:0.68rem;color:#4a5680;margin-top:0.8rem;">🟢 Top 3 placares mais prováveis destacados</p></div>', unsafe_allow_html=True)

    # Kelly calculator
    st.markdown('<hr class="custom">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Kelly Criterion Calculator</div>', unsafe_allow_html=True)

    col_k1, col_k2 = st.columns([1, 2])
    with col_k1:
        custom_prob  = st.slider("Probabilidade do seu modelo (%)", 1, 99, 55, 1) / 100.0
        custom_odd   = st.number_input("Odd da Casa", min_value=1.01, max_value=50.0, value=2.00, step=0.05, format="%.2f", key="custom_odd")
        custom_kelly = kelly_fraction(custom_prob, custom_odd, kelly_frac)
        custom_stake = round(custom_kelly * bankroll, 2)
        custom_ev    = expected_value(custom_prob, custom_odd)
        ev_color = "#4ade80" if custom_ev > 0 else "#f87171"
        ev_sign  = "+" if custom_ev > 0 else ""
        st.markdown(f"""<div class="kelly-result"><div class="kelly-label">Stake Sugerida ({int(kelly_frac*100)}% Kelly)</div><div class="kelly-stake">R$ {custom_stake:,.2f}</div><div class="kelly-label" style="color:{ev_color};margin-top:0.5rem;">EV: {ev_sign}{round(custom_ev*100,1)}%</div><div class="kelly-label" style="margin-top:0.3rem;">Fração Kelly: {round(custom_kelly*100,1)}% da banca</div></div>""", unsafe_allow_html=True)
    with col_k2:
        kelly_data = []
        for pct in range(45, 75, 2):
            p  = pct / 100.0
            kf = kelly_fraction(p, custom_odd, kelly_frac)
            ev = expected_value(p, custom_odd)
            kelly_data.append({"Prob Modelo %": pct, "Kelly Stake %": round(kf*100,2), "Stake (R$)": round(kf*bankroll,2), "EV %": round(ev*100,2)})
        df_kelly = pd.DataFrame(kelly_data)
        st.dataframe(df_kelly.style.applymap(lambda v: 'color: #4ade80' if isinstance(v, (int, float)) and v > 0 else 'color: #f87171', subset=["EV %"]), use_container_width=True, height=300)

    st.markdown('<hr class="custom">', unsafe_allow_html=True)
    st.markdown("""<div style="text-align:center;padding:1rem 0 2rem;font-family:'DM Mono',monospace;font-size:0.68rem;color:#2e3a5a;letter-spacing:0.1em;">FOOTBALL EDGE ANALYTICS · POISSON MODEL · APENAS PARA FINS EDUCACIONAIS<br>Resultados passados não garantem resultados futuros. Aposte com responsabilidade.</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 2 — GESTÃO DE BANCA
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Gestão de Banca")
    st.markdown('<hr class="custom">', unsafe_allow_html=True)

    df_bets = load_bets()

    # ── REGISTRAR NOVA BET ──────────────────────
    st.markdown('<div class="section-header">Registrar Nova Aposta</div>', unsafe_allow_html=True)

    with st.form("new_bet_form", clear_on_submit=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            bet_date   = st.date_input("Data", value=date.today())
            bet_home   = st.text_input("Time da Casa", value=home_team)
            bet_away   = st.text_input("Time Visitante", value=away_team)
        with fc2:
            bet_market = st.selectbox("Mercado", [f"{home_team} Win", "Draw", f"{away_team} Win", "Over 2.5", "Under 2.5", "Outro"])
            bet_odd    = st.number_input("Odd apostada", min_value=1.01, max_value=100.0, value=2.00, step=0.05, format="%.2f")
            bet_stake  = st.number_input("Stake apostada (R$)", min_value=1.0, max_value=100000.0, value=50.0, step=5.0, format="%.2f")
        with fc3:
            bet_result = st.selectbox("Resultado", ["Pendente ⏳", "Ganhou ✅", "Perdeu ❌", "Void/Reembolso 🔄"])
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("➕ Registrar Aposta", use_container_width=True)

        if submitted:
            if bet_result == "Ganhou ✅":
                profit = round(bet_stake * (bet_odd - 1), 2)
            elif bet_result == "Perdeu ❌":
                profit = -round(bet_stake, 2)
            elif bet_result == "Void/Reembolso 🔄":
                profit = 0.0
            else:
                profit = None

            save_bet({
                "date":   str(bet_date),
                "home":   bet_home,
                "away":   bet_away,
                "market": bet_market,
                "odd":    bet_odd,
                "stake":  bet_stake,
                "result": bet_result,
                "profit": profit if profit is not None else "",
            })
            st.success("✅ Aposta registrada com sucesso!")
            st.rerun()

    # ── RESUMO GERAL ───────────────────────────
    st.markdown('<hr class="custom">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Resumo Geral</div>', unsafe_allow_html=True)

    if len(df_bets) > 0:
        df_closed = df_bets[df_bets["result"] != "Pendente ⏳"].copy()
        df_closed["profit"] = pd.to_numeric(df_closed["profit"], errors="coerce").fillna(0)
        df_closed["stake"]  = pd.to_numeric(df_closed["stake"],  errors="coerce").fillna(0)

        total_bets   = len(df_bets)
        closed_bets  = len(df_closed)
        total_profit = df_closed["profit"].sum()
        total_staked = df_closed["stake"].sum()
        roi          = (total_profit / total_staked * 100) if total_staked > 0 else 0
        wins         = len(df_closed[df_closed["result"] == "Ganhou ✅"])
        win_rate     = (wins / closed_bets * 100) if closed_bets > 0 else 0
        current_bank = bankroll + total_profit

        profit_color = "#4ade80" if total_profit >= 0 else "#f87171"
        roi_color    = "#4ade80" if roi >= 0 else "#f87171"

        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        with sm1:
            st.markdown(f'<div class="stat-card"><div class="stat-lbl">Total de Bets</div><div class="stat-val" style="color:#60a5fa">{total_bets}</div></div>', unsafe_allow_html=True)
        with sm2:
            st.markdown(f'<div class="stat-card"><div class="stat-lbl">Lucro / Prejuízo</div><div class="stat-val" style="color:{profit_color}">R$ {total_profit:+,.2f}</div></div>', unsafe_allow_html=True)
        with sm3:
            st.markdown(f'<div class="stat-card"><div class="stat-lbl">ROI Geral</div><div class="stat-val" style="color:{roi_color}">{roi:+.1f}%</div></div>', unsafe_allow_html=True)
        with sm4:
            st.markdown(f'<div class="stat-card"><div class="stat-lbl">Taxa de Acerto</div><div class="stat-val" style="color:#a78bfa">{win_rate:.1f}%</div></div>', unsafe_allow_html=True)
        with sm5:
            st.markdown(f'<div class="stat-card"><div class="stat-lbl">Banca Atual</div><div class="stat-val" style="color:#facc15">R$ {current_bank:,.2f}</div></div>', unsafe_allow_html=True)

        # ── GRÁFICO EVOLUÇÃO ───────────────────
        if len(df_closed) > 0:
            st.markdown('<hr class="custom">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Evolução da Banca</div>', unsafe_allow_html=True)

            df_chart = df_closed.sort_values("date").copy()
            df_chart["profit_num"] = pd.to_numeric(df_chart["profit"], errors="coerce").fillna(0)
            df_chart["banca_acum"] = bankroll + df_chart["profit_num"].cumsum()
            df_chart["bet_num"]    = range(1, len(df_chart)+1)

            chart_data = df_chart[["bet_num", "banca_acum"]].rename(columns={"bet_num": "Aposta nº", "banca_acum": "Banca (R$)"})
            st.line_chart(chart_data.set_index("Aposta nº"), color="#60a5fa", use_container_width=True)

        # ── ROI POR MERCADO ───────────────────
        st.markdown('<hr class="custom">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">ROI por Mercado</div>', unsafe_allow_html=True)

        roi_data = []
        for market in df_closed["market"].unique():
            df_m = df_closed[df_closed["market"] == market]
            staked_m = df_m["stake"].sum()
            profit_m = df_m["profit"].sum()
            wins_m   = len(df_m[df_m["result"] == "Ganhou ✅"])
            roi_m    = (profit_m / staked_m * 100) if staked_m > 0 else 0
            roi_data.append({
                "Mercado": market,
                "Bets": len(df_m),
                "Ganhou": wins_m,
                "% Acerto": f"{wins_m/len(df_m)*100:.1f}%" if len(df_m) > 0 else "—",
                "Total Apostado (R$)": f"R$ {staked_m:,.2f}",
                "Lucro (R$)": f"R$ {profit_m:+,.2f}",
                "ROI": f"{roi_m:+.1f}%",
            })

        df_roi = pd.DataFrame(roi_data)
        st.dataframe(df_roi, use_container_width=True, hide_index=True)

    else:
        st.info("Nenhuma aposta registrada ainda. Use o formulário acima para começar!")

    # ── HISTÓRICO COMPLETO ──────────────────────
    st.markdown('<hr class="custom">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Histórico de Apostas</div>', unsafe_allow_html=True)

    if len(df_bets) > 0:
        df_display = df_bets.copy()
        df_display["profit"] = pd.to_numeric(df_display["profit"], errors="coerce")

        def color_result(val):
            if "Ganhou" in str(val): return "color: #4ade80"
            if "Perdeu" in str(val): return "color: #f87171"
            if "Void"   in str(val): return "color: #facc15"
            return "color: #a78bfa"

        def color_profit(val):
            try:
                return "color: #4ade80" if float(val) > 0 else ("color: #f87171" if float(val) < 0 else "color: #facc15")
            except: return ""

        styled = df_display.style\
            .applymap(color_result, subset=["result"])\
            .applymap(color_profit, subset=["profit"])\
            .format({"odd": "{:.2f}", "stake": "R$ {:.2f}", "profit": lambda x: f"R$ {x:+.2f}" if pd.notna(x) and x != "" else "⏳"})

        st.dataframe(styled, use_container_width=True)

        # Delete a bet
        with st.expander("🗑️ Apagar uma aposta"):
            del_idx = st.number_input("Índice da aposta a apagar", min_value=0, max_value=max(0, len(df_bets)-1), step=1, value=0)
            if st.button("Apagar aposta selecionada", type="secondary"):
                delete_bet(del_idx)
                st.warning(f"Aposta #{del_idx} removida.")
                st.rerun()

        # Export JSON (funciona na web)
        if st.button("📥 Exportar dados (JSON)"):
            st.json(st.session_state.bets_history)
    else:
        st.info("Nenhuma aposta registrada ainda.")