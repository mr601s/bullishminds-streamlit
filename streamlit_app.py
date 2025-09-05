# Copyright (c) 2025 Bullish Minds AI
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yaml
import os
import tempfile
import time

from data import (
    build_snapshot, save_snapshot, fetch_sp500_tickers,
    fetch_intraday
)
from scoring import (
    score_value, score_quality, score_momentum, score_risk,
    blend_scores, confidence
)

# Configure page
st.set_page_config(
    page_title="BullishMinds Markets",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'live_ticker' not in st.session_state:
    st.session_state.live_ticker = ""
if 'live_src' not in st.session_state:
    st.session_state.live_src = None

# ---------------------------- config ----------------------------
@st.cache_data
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_config()
APP_TITLE = CFG["app"]["title"]
APP_SUB = CFG["app"]["subtitle"]
M_CFG = CFG["factors"]
RUNTIME = CFG["runtime"]

DISCLAIMER = (
    "Educational tool only. Not investment advice. Data may be delayed or incomplete. "
    "Sources: SEC EDGAR (companyfacts) and Yahoo Finance (EOD / intraday delayed)."
)

# ---------------------------- helpers ----------------------------
def normalize_weights(v, q, m, r):
    vals = [max(0.0, float(v)), max(0.0, float(q)), max(0.0, float(m)), max(0.0, float(r))]
    s = sum(vals) or 1.0
    return [x / s for x in vals]

def plot_prices(prices: list, ma_s: int, ma_l: int):
    if not prices:
        return go.Figure()
    try:
        df = pd.DataFrame(prices).copy()
        df["MA_S"] = pd.to_numeric(df["Close"], errors="coerce").rolling(ma_s, min_periods=max(2, ma_s//2)).mean()
        df["MA_L"] = pd.to_numeric(df["Close"], errors="coerce").rolling(ma_l, min_periods=max(2, ma_l//2)).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", line=dict(color="#0B5FFF")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_S"], name=f"{ma_s}DMA", line=dict(color="#FF6B35")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_L"], name=f"{ma_l}DMA", line=dict(color="#36BA98")))
        fig.update_layout(height=400, margin=dict(l=10,r=10,t=30,b=10), showlegend=True)
        return fig
    except Exception as e:
        st.error(f"Error plotting prices: {e}")
        return go.Figure()

def create_gauge(score: float, title: str):
    try:
        val = score if score is not None else 0
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            number={'suffix': " / 100"},
            gauge={'axis': {'range': [0,100]}, 'bar': {'color': '#0B5FFF'},
                   'steps': [{'range':[0,40],'color':'#ffe5e5'},
                            {'range':[40,70],'color':'#fff6d6'},
                            {'range':[70,100],'color':'#e6ffea'}]},
            title={"text": title, "font": {"size": 16}}
        ))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=10))
        return fig
    except Exception as e:
        st.error(f"Error creating gauge: {e}")
        return go.Figure()

def fmt_price(v): 
    return f"${v:,.2f}" if v is not None else "—"

def fmt_pct(v): 
    return f"{v:.1%}" if v is not None else "—"

def score_from_snapshot(snap, weights):
    try:
        v_score, v_meta = score_value(snap["derived"]["pe"], snap["derived"]["fcf_yield"], M_CFG["value"])
        q_score, q_meta = score_quality(snap["derived"]["roe"], snap["derived"]["gross_margin"], M_CFG["quality"])
        m_score, m_meta = score_momentum(
            snap["price_features"]["rtn_12m"],
            snap["price_features"]["rtn_6m"],
            snap["price_features"]["rtn_1m"],
            snap["price_features"]["above_ma_long"],
            M_CFG["momentum"],
        )
        r_score, r_meta = score_risk(
            snap["derived"]["leverage"],
            snap["price_features"]["vol_90d"],
            snap["price_features"]["max_drawdown_1y"],
            snap["price_features"]["beta_1y"],
            M_CFG["risk"]
        )
        comp = blend_scores({"value": v_score, "quality": q_score, "momentum": m_score, "risk": r_score},
                           {"value": weights[0], "quality": weights[1], "momentum": weights[2], "risk": weights[3]})
        return comp, (v_score, q_score, m_score, r_score), (v_meta, q_meta, m_meta, r_meta)
    except Exception as e:
        st.error(f"Error in scoring: {e}")
        return None, (None, None, None, None), ({"details": []}, {"details": []}, {"details": []}, {"details": []})

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_snapshot(ticker, prices_period, prices_interval, ma_s, ma_l):
    return build_snapshot(
        ticker=ticker,
        prices_period=prices_period,
        prices_interval=prices_interval,
        ma_s=ma_s,
        ma_l=ma_l
    )

def provider_diagnostic():
    poly_ok = bool(os.environ.get("POLYGON_API_KEY"))
    fin_ok = bool(os.environ.get("FINNHUB_API_KEY"))
    sec_ok = bool(os.environ.get("SEC_USER_AGENT"))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Polygon", "✅ Ready" if poly_ok else "❌ No Key")
    with col2:
        st.metric("Finnhub", "✅ Ready" if fin_ok else "❌ No Key")
    with col3:
        st.metric("SEC UA", "✅ Ready" if sec_ok else "❌ No Key")

def parse_ticker_list(s: str):
    if not s:
        return []
    raw = s.replace("\n", ",").replace(";", ",").split(",")
    return sorted({x.strip().upper() for x in raw if x.strip()})

# ---------------------------- Main App ----------------------------
def main():
    # Header
    st.title("⚡ " + APP_TITLE)
    st.markdown(f"*{APP_SUB}*")
    
    # Sidebar for weights
    st.sidebar.header("Factor Weights")
    st.sidebar.markdown("Adjust the importance of each factor (auto-normalized)")
    
    w_v = st.sidebar.slider("Value", 0.0, 1.0, 0.30, 0.05, help="P/E ratio, FCF yield")
    w_q = st.sidebar.slider("Quality", 0.0, 1.0, 0.30, 0.05, help="ROE, gross margin")
    w_m = st.sidebar.slider("Momentum", 0.0, 1.0, 0.30, 0.05, help="Price trends, moving averages")
    w_r = st.sidebar.slider("Risk", 0.0, 1.0, 0.10, 0.05, help="Volatility, leverage, drawdown")
    
    weights = normalize_weights(w_v, w_q, w_m, w_r)
    st.sidebar.markdown(f"**Normalized:** V:{weights[0]:.2f} Q:{weights[1]:.2f} M:{weights[2]:.2f} R:{weights[3]:.2f}")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Single Analysis", "⚖️ Compare", "📊 Watchlist", "🌍 Universe Screen", "📚 Learn"])

    # TAB 1: Single Analysis
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticker = st.text_input("Enter Ticker Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, TSLA").upper()
        
        with col2:
            if st.button("📈 Analyze Stock", type="primary", use_container_width=True):
                if ticker:
                    with st.spinner(f"Analyzing {ticker}..."):
                        try:
                            # Get data
                            snap = get_snapshot(
                                ticker=ticker,
                                prices_period=RUNTIME["price_period"],
                                prices_interval=RUNTIME["price_interval"],
                                ma_s=M_CFG["momentum"]["ma_short"],
                                ma_l=M_CFG["momentum"]["ma_long"]
                            )
                            
                            # Score the stock
                            comp, (v_score, q_score, m_score, r_score), metas = score_from_snapshot(snap, weights)
                            conf = confidence(snap["fundamentals"], snap["price_features"])

                            # Display header info
                            name = snap['info'].get('company') or ticker
                            sector = snap.get("profile", {}).get("sector")
                            industry = snap.get("profile", {}).get("industry")
                            
                            st.subheader(f"📊 {name} ({ticker})")
                            
                            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                            with col_info1:
                                st.metric("Last Close", fmt_price(snap['last_close']))
                            with col_info2:
                                st.metric("Sector", sector or "—")
                            with col_info3:
                                st.metric
