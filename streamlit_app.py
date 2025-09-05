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
                                st.metric("Industry", industry or "—")
                            with col_info4:
                                st.metric("Data Confidence", conf)

                            # Price chart
                            if snap["prices"]:
                                st.subheader("📈 Price & Moving Averages")
                                price_fig = plot_prices(snap["prices"], M_CFG["momentum"]["ma_short"], M_CFG["momentum"]["ma_long"])
                                st.plotly_chart(price_fig, use_container_width=True)

                            # Score gauges
                            st.subheader("🎯 Investment Scores")
                            gauge_col1, gauge_col2, gauge_col3, gauge_col4, gauge_col5 = st.columns(5)
                            
                            with gauge_col1:
                                comp_fig = create_gauge(comp, "Composite")
                                st.plotly_chart(comp_fig, use_container_width=True)
                            with gauge_col2:
                                v_fig = create_gauge(v_score or 0, "Value")
                                st.plotly_chart(v_fig, use_container_width=True)
                            with gauge_col3:
                                q_fig = create_gauge(q_score or 0, "Quality")
                                st.plotly_chart(q_fig, use_container_width=True)
                            with gauge_col4:
                                m_fig = create_gauge(m_score or 0, "Momentum")
                                st.plotly_chart(m_fig, use_container_width=True)
                            with gauge_col5:
                                r_fig = create_gauge(r_score or 0, "Risk")
                                st.plotly_chart(r_fig, use_container_width=True)

                            # Financial metrics table
                            st.subheader("📋 Key Metrics")
                            
                            col_table1, col_table2 = st.columns(2)
                            
                            with col_table1:
                                st.markdown("**Valuation & Profitability**")
                                metrics_df1 = pd.DataFrame([
                                    ["P/E (TTM)", f"{snap['derived']['pe']:.1f}" if snap['derived']['pe'] else "—"],
                                    ["FCF Yield (TTM)", fmt_pct(snap['derived']['fcf_yield'])],
                                    ["ROE (TTM)", fmt_pct(snap['derived']['roe'])],
                                    ["Gross Margin", fmt_pct(snap['derived']['gross_margin'])],
                                    ["Operating Margin", fmt_pct(snap['derived']['operating_margin'])],
                                    ["Net Margin", fmt_pct(snap['derived']['net_margin'])],
                                ], columns=["Metric", "Value"])
                                st.dataframe(metrics_df1, use_container_width=True, hide_index=True)
                            
                            with col_table2:
                                st.markdown("**Risk & Returns**")
                                metrics_df2 = pd.DataFrame([
                                    ["Dividend Yield (TTM)", fmt_pct(snap['derived']['dividend_yield'])],
                                    ["Payout Ratio (TTM)", fmt_pct(snap['derived']['payout_ratio'])],
                                    ["Debt/Assets", fmt_pct(snap['derived']['leverage'])],
                                    ["Volatility 90d (σ)", fmt_pct(snap['price_features']['vol_90d'])],
                                    ["Max Drawdown 1y", fmt_pct(snap['price_features']['max_drawdown_1y'])],
                                    ["Beta 1y vs SPY", f"{snap['price_features']['beta_1y']:.2f}" if snap['price_features']['beta_1y'] is not None else "—"],
                                ], columns=["Metric", "Value"])
                                st.dataframe(metrics_df2, use_container_width=True, hide_index=True)

                            # Explanations
                            st.subheader("💡 Score Explanations")
                            why_text = []
                            for title, meta in zip(["Value","Quality","Momentum","Risk"], metas):
                                if meta and meta.get("details"):
                                    why_text.append(f"**{title}:**")
                                    for detail in meta["details"]:
                                        why_text.append(f"- {detail}")
                                    why_text.append("")
                            
                            if why_text:
                                st.markdown("\n".join(why_text))
                            else:
                                st.info("Insufficient data to generate detailed explanations.")

                            # Live intraday data
                            st.subheader("📊 Live Intraday (1min delayed)")
                            try:
                                df, meta = fetch_intraday(ticker, lookback_minutes=180, interval="1m")
                                if df is not None and meta is not None:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="1m Close"))
                                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    chg = f"{meta['change']*100:.2f}%" if meta.get("change") is not None else "—"
                                    st.info(f"**Live Data** — {ticker}: {fmt_price(meta['last'])} • Session Change: {chg}")
                                else:
                                    st.warning("Live data not available")
                            except Exception as e:
                                st.warning(f"Live data error: {e}")

                            save_snapshot(ticker, snap)

                        except Exception as e:
                            st.error(f"Error analyzing {ticker}: {str(e)}")
                else:
                    st.warning("Please enter a ticker symbol")

    # TAB 2: Compare
    with tab2:
        st.header("⚖️ Compare Stocks")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            t1 = st.text_input("Ticker 1", value="AAPL", key="comp1").upper()
        with col2:
            t2 = st.text_input("Ticker 2", value="MSFT", key="comp2").upper()
        with col3:
            t3 = st.text_input("Ticker 3 (optional)", value="", key="comp3").upper()
            
        if st.button("🔍 Compare Stocks", type="primary"):
            tickers = [t for t in [t1, t2, t3] if t and t.strip()]
            if tickers:
                with st.spinner("Comparing stocks..."):
                    try:
                        rows = []
                        for tk in tickers:
                            snap = get_snapshot(
                                ticker=tk,
                                prices_period=RUNTIME["price_period"],
                                prices_interval=RUNTIME["price_interval"],
                                ma_s=M_CFG["momentum"]["ma_short"],
                                ma_l=M_CFG["momentum"]["ma_long"]
                            )
                            comp, (v, q, m, r), _ = score_from_snapshot(snap, weights)
                            name = snap["info"].get("company") or tk
                            sector = snap.get("profile", {}).get("sector")
                            rows.append({
                                "Ticker": tk,
                                "Company": name,
                                "Sector": sector or "",
                                "Close": fmt_price(snap["last_close"]),
                                "Composite": f"{comp:.1f}" if comp else "—",
                                "Value": f"{v:.1f}" if v else "—",
                                "Quality": f"{q:.1f}" if q else "—",
                                "Momentum": f"{m:.1f}" if m else "—",
                                "Risk": f"{r:.1f}" if r else "—"
                            })
                        
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Highlight winner
                        numeric_df = df.copy()
                        numeric_df["Composite_num"] = pd.to_numeric(df["Composite"].str.replace("—", "0"), errors="coerce")
                        winner = numeric_df.loc[numeric_df["Composite_num"].idxmax(), "Ticker"]
                        st.success(f"🏆 **Top Pick:** {winner}")
                        
                    except Exception as e:
                        st.error(f"Error comparing stocks: {e}")
            else:
                st.warning("Please enter at least one ticker")

    # TAB 3: Watchlist
    with tab3:
        st.header("📊 Watchlist Scoring")
        
        wl_text = st.text_area(
            "Enter tickers (comma, semicolon, or newline separated, max 20):",
            value="AAPL, MSFT, GOOGL, AMZN, TSLA",
            height=100,
            help="Example: AAPL, MSFT; GOOGL\nAMZN, TSLA"
        )
        
        if st.button("📈 Score Watchlist", type="primary"):
            tickers = parse_ticker_list(wl_text)[:20]  # Max 20
            if tickers:
                with st.spinner(f"Scoring {len(tickers)} stocks..."):
                    progress_bar = st.progress(0)
                    rows = []
                    
                    for i, tk in enumerate(tickers):
                        try:
                            snap = get_snapshot(
                                ticker=tk,
                                prices_period=RUNTIME["price_period"],
                                prices_interval=RUNTIME["price_interval"],
                                ma_s=M_CFG["momentum"]["ma_short"],
                                ma_l=M_CFG["momentum"]["ma_long"]
                            )
                            comp, (v, q, m, r), _ = score_from_snapshot(snap, weights)
                            name = snap["info"].get("company") or tk
                            sector = snap.get("profile", {}).get("sector")
                            rows.append({
                                "Ticker": tk,
                                "Company": name,
                                "Sector": sector or "",
                                "Close": snap["last_close"],
                                "Composite": comp or 0,
                                "Value": v or 0,
                                "Quality": q or 0,
                                "Momentum": m or 0,
                                "Risk": r or 0
                            })
                            progress_bar.progress((i + 1) / len(tickers))
                        except Exception as e:
                            st.warning(f"Could not process {tk}: {e}")
                    
                    if rows:
                        df = pd.DataFrame(rows).sort_values("Composite", ascending=False)
                        
                        # Format for display
                        display_df = df.copy()
                        display_df["Close"] = display_df["Close"].apply(fmt_price)
                        display_df["Composite"] = display_df["Composite"].apply(lambda x: f"{x:.1f}")
                        display_df["Value"] = display_df["Value"].apply(lambda x: f"{x:.1f}")
                        display_df["Quality"] = display_df["Quality"].apply(lambda x: f"{x:.1f}")
                        display_df["Momentum"] = display_df["Momentum"].apply(lambda x: f"{x:.1f}")
                        display_df["Risk"] = display_df["Risk"].apply(lambda x: f"{x:.1f}")
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Download link
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"watchlist_scores_{int(time.time())}.csv",
                            mime="text/csv"
                        )
                        
                        st.success(f"🎯 **Top Pick:** {df.iloc[0]['Ticker']} (Score: {df.iloc[0]['Composite']:.1f})")
            else:
                st.warning("Please enter some tickers")

    # TAB 4: Universe Screen
    with tab4:
        st.header("🌍 Universe Screening")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            universe = st.radio("Select Universe:", ["S&P 500", "Custom list"])
            top_n = st.slider("Show top N results:", 5, 100, 25, 5)
        
        with col2:
            if universe == "Custom list":
                custom_list = st.text_area("Custom tickers:", height=100)
            else:
                custom_list = ""
        
        if st.button("🔎 Screen Universe", type="primary"):
            with st.spinner("Screening universe..."):
                try:
                    if universe == "S&P 500":
                        tickers = fetch_sp500_tickers()
                    else:
                        tickers = parse_ticker_list(custom_list)
                    
                    tickers = tickers[:500]  # Limit to prevent timeouts
                    
                    if not tickers:
                        st.warning("No tickers to screen")
                    else:
                        progress_bar = st.progress(0)
                        rows = []
                        
                        for i, tk in enumerate(tickers):
                            try:
                                snap = get_snapshot(
                                    ticker=tk,
                                    prices_period=RUNTIME["price_period"],
                                    prices_interval=RUNTIME["price_interval"],
                                    ma_s=M_CFG["momentum"]["ma_short"],
                                    ma_l=M_CFG["momentum"]["ma_long"]
                                )
                                comp, (v, q, m, r), _ = score_from_snapshot(snap, weights)
                                if comp is not None:  # Only include stocks with valid scores
                                    rows.append({
                                        "Ticker": tk,
                                        "Composite": comp,
                                        "Value": v or 0,
                                        "Quality": q or 0,
                                        "Momentum": m or 0,
                                        "Risk": r or 0
                                    })
                                progress_bar.progress((i + 1) / len(tickers))
                            except:
                                continue  # Skip problematic tickers
                        
                        if rows:
                            df = pd.DataFrame(rows)
                            
                            # Add percentiles
                            for col in ["Composite", "Value", "Quality", "Momentum", "Risk"]:
                                df[f"{col}_pct"] = (df[col].rank(pct=True) * 100).round(1)
                            
                            # Sort and limit
                            df_sorted = df.sort_values("Composite", ascending=False).head(top_n)
                            
                            # Display
                            display_cols = ["Ticker", "Composite", "Composite_pct", "Value", "Quality", "Momentum", "Risk"]
                            display_df = df_sorted[display_cols].copy()
                            display_df.columns = ["Ticker", "Score", "Percentile", "Value", "Quality", "Momentum", "Risk"]
                            
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                            
                            # Download
                            csv = df_sorted.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"universe_screen_{int(time.time())}.csv",
                                mime="text/csv"
                            )
                            
                            st.info(f"📊 Screened {len(df)} stocks from {universe}, showing top {len(df_sorted)}")
                        else:
                            st.warning("No valid results found")
                            
                except Exception as e:
                    st.error(f"Screening error: {e}")

    # TAB 5: Learn
    with tab5:
        st.header("📚 Learning Center")
        
        try:
            with open("tooltips.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())
        except FileNotFoundError:
            try:
                with open("content/tooltips.md", "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            except FileNotFoundError:
                st.markdown("""
                # Investment Factor Analysis

                ## Value Factors
                - **P/E Ratio**: Price-to-Earnings ratio compares stock price to earnings per share
                - **FCF Yield**: Free Cash Flow Yield measures cash generation relative to market cap

                ## Quality Factors  
                - **ROE**: Return on Equity measures how efficiently a company uses shareholder equity
                - **Gross Margin**: Gross profit as percentage of revenue

                ## Momentum Factors
                - **Price Returns**: Historical stock performance over different time periods
                - **Moving Averages**: Trend indicators showing price direction

                ## Risk Factors
                - **Volatility**: Standard deviation of returns measuring price fluctuation
                - **Beta**: Correlation with market (SPY) movements
                - **Max Drawdown**: Largest peak-to-trough decline
                - **Leverage**: Debt-to-assets ratio

                ## Data Sources
                - SEC EDGAR: Official company financials via XBRL
                - Yahoo Finance: Stock prices and market data (delayed)
                - Real-time: Polygon.io and Finnhub (API keys required)

                *Educational purposes only. Not investment advice.*
                """)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("⚡ Built with Streamlit")
    with col2:
        provider_diagnostic()
    with col3:
        st.caption(DISCLAIMER)

if __name__ == "__main__":
    main()
