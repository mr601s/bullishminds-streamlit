# Copyright (c) 2025 Bullish Minds AI
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import os

# Universal secrets loader for Streamlit Cloud compatibility
try:
    import streamlit as st
    for k, v in st.secrets.items():
        os.environ[k] = v
except:
    pass  # No streamlit or secrets available

# Optional: also load .env file if running locally
if os.path.exists('.env'):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed

import json, time
from io import StringIO
from typing import Dict, Any, Optional, Tuple, List

import requests
import pandas as pd
import numpy as np
import yfinance as yf

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

SEC_UA = os.environ.get("SEC_USER_AGENT", "BullishMindsMarkets/1.0 (contact@example.com)")
SEC_HEADERS = {"User-Agent": SEC_UA}

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# ---------------------------- cache helpers ----------------------------
def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)

def _read_cache(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_cache(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def _days_ago(ts: float) -> float:
    return (time.time() - ts) / 86400.0

# ---------------------------- SEC lookups ----------------------------
def get_ticker_cik_map(cache_days=7) -> pd.DataFrame:
    # Try loading local file first (so SEC lookup isn't critical)
    # Place company_tickers.json at top level of your Space (not in a subfolder)
    local_path = "company_tickers.json"
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rows = []
        for _, row in raw.items():
            rows.append({
                "ticker": (row.get("ticker") or "").upper(),
                "cik": f'{int(row.get("cik_str", 0)):010d}',
                "title": row.get("title", "")
            })
        return pd.DataFrame(rows)
    # Otherwise, fallback to the SEC web as a last resort (should only run locally)
    resp = requests.get(SEC_TICKER_MAP_URL, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    rows = []
    for _, row in raw.items():
        rows.append({
            "ticker": (row.get("ticker") or "").upper(),
            "cik": f'{int(row.get("cik_str", 0)):010d}',
            "title": row.get("title", "")
        })
    return pd.DataFrame(rows)

def get_cik_for_ticker(ticker: str) -> Optional[Tuple[str, str]]:
    df = get_ticker_cik_map()
    t = ticker.upper().strip()
    match = df[df["ticker"] == t]
    if match.empty:
        return None
    row = match.iloc[0]
    return row["cik"], row["title"]

# ---------------------------- SEC companyfacts ----------------------------
def fetch_companyfacts(cik: str, cache_days: int = 3) -> Optional[dict]:
    cache_file = _cache_path(f"facts_{cik}.json")
    cached = _read_cache(cache_file)
    if cached and _days_ago(cached.get("_ts", 0)) < cache_days:
        return cached["data"]
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=60)
    if r.status_code != 200:
        return None
    data = r.json()
    _write_cache(cache_file, {"_ts": time.time(), "data": data})
    return data

def _latest_numeric(series: list, units: Optional[str] = None) -> Optional[float]:
    if not series:
        return None
    df = pd.DataFrame(series)
    if "val" not in df.columns:
        return None
    if units and "uom" in df.columns:
        df = df[df["uom"] == units]
    if df.empty:
        return None
    date_col = "end" if "end" in df.columns else ("fy" if "fy" in df.columns else None)
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values("date")
    return float(df["val"].iloc[-1]) if len(df) else None

def parse_facts(facts: dict) -> Dict[str, Optional[float]]:
    """Educational fundamentals from us-gaap tags (TTM when possible)."""
    out = {k: None for k in [
        "revenue_ttm","gross_profit_ttm","operating_income_ttm",
        "net_income_ttm","cfo_ttm","capex_ttm",
        "assets","equity","debt","cash","shares_out"
    ]}
    try:
        us = facts.get("facts", {}).get("us-gaap", {})

        def ttm_sum(tag: str) -> Optional[float]:
            if tag not in us: return None
            series = us[tag].get("units", {}).get("USD", [])
            if not series: return None
            df = pd.DataFrame(series)
            if "end" not in df.columns or "val" not in df.columns:
                return None
            df["end"] = pd.to_datetime(df["end"], errors="coerce")
            df = df.dropna(subset=["end"]).sort_values("end")
            q = df[df.get("fp", "").astype(str).str.upper().isin(["Q1","Q2","Q3","Q4"])]
            if len(q) >= 4:
                return float(q["val"].tail(4).sum())
            return float(df["val"].iloc[-1]) if len(df) else None

        out["revenue_ttm"] = ttm_sum("Revenues") or ttm_sum("SalesRevenueNet")
        out["gross_profit_ttm"] = ttm_sum("GrossProfit")
        out["operating_income_ttm"] = ttm_sum("OperatingIncomeLoss")
        out["net_income_ttm"] = ttm_sum("NetIncomeLoss")
        out["cfo_ttm"] = ttm_sum("NetCashProvidedByUsedInOperatingActivities")
        capex = ttm_sum("PaymentsToAcquirePropertyPlantAndEquipment")
        if capex is not None:
            out["capex_ttm"] = float(capex)

        def latest_usd(tag: str) -> Optional[float]:
            if tag not in us: return None
            return _latest_numeric(us[tag].get("units", {}).get("USD", []), units=None)

        out["assets"] = latest_usd("Assets")
        out["equity"] = latest_usd("StockholdersEquity")
        out["debt"] = latest_usd("LongTermDebt") or latest_usd("LongTermDebtAndCapitalLeaseObligations")
        out["cash"] = latest_usd("CashAndCashEquivalentsAtCarryingValue")

        # Shares outstanding
        shares = None
        if "CommonStockSharesOutstanding" in us:
            shares = _latest_numeric(us["CommonStockSharesOutstanding"].get("units", {}).get("shares", []))
        if shares is None and "WeightedAverageNumberOfDilutedSharesOutstanding" in us:
            shares = _latest_numeric(us["WeightedAverageNumberOfDilutedSharesOutstanding"].get("units", {}).get("shares", []))
        out["shares_out"] = shares
        return out
    except Exception:
        return out

# ---------------------------- Prices, dividends & beta ----------------------------
def _read_df_from_cache(cache_file: str) -> Optional[pd.DataFrame]:
    cached = _read_cache(cache_file)
    if cached:
        return pd.read_json(StringIO(json.dumps(cached["data"])), orient="split")
    return None

def fetch_prices(ticker: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    key = f"px_{ticker}_{period}_{interval}.json"
    cache_file = _cache_path(key)
    cached_meta = _read_cache(cache_file)
    if (cached := _read_df_from_cache(cache_file)) is not None and _days_ago(cached_meta.get("_ts", 0)) < 1:
        return cached
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df.reset_index().rename(columns=str.title)  # Date, Open, High, Low, Close, Volume
        _write_cache(cache_file, {"_ts": time.time(), "data": json.loads(df.to_json(orient="split"))})
        return df
    except Exception:
        return None

def fetch_benchmark_prices(symbol: str = "SPY", period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    key = f"px_bm_{symbol}_{period}_{interval}.json"
    cache_file = _cache_path(key)
    cached_meta = _read_cache(cache_file)
    if (cached := _read_df_from_cache(cache_file)) is not None and _days_ago(cached_meta.get("_ts", 0)) < 7:
        return cached
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df.reset_index().rename(columns=str.title)
        _write_cache(cache_file, {"_ts": time.time(), "data": json.loads(df.to_json(orient="split"))})
        return df
    except Exception:
        return None

def compute_dividend_yield(ticker: str, last_close: Optional[float]) -> Optional[float]:
    if last_close is None or last_close <= 0:
        return None
    try:
        s = yf.Ticker(ticker).dividends
        if s is None or s.empty:
            return None
        cut = pd.Timestamp.today(tz=None) - pd.Timedelta(days=365)
        ttm_total = float(s[s.index >= cut].sum())
        if ttm_total <= 0:
            return None
        return ttm_total / float(last_close)
    except Exception:
        return None

def compute_price_features(df: pd.DataFrame, ma_short: int = 50, ma_long: int = 200,
                           bench_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    out = {"ma_short": None, "ma_long": None, "rtn_12m": None, "rtn_6m": None, "rtn_1m": None,
           "vol_90d": None, "max_drawdown_1y": None, "above_ma_long": None, "beta_1y": None}
    if df is None or df.empty:
        return out
    dfe = df.copy()
    dfe["Close"] = pd.to_numeric(dfe["Close"], errors="coerce")
    dfe["MA_S"] = dfe["Close"].rolling(ma_short, min_periods=max(2, ma_short//2)).mean()
    dfe["MA_L"] = dfe["Close"].rolling(ma_long, min_periods=max(2, ma_long//2)).mean()
    out["ma_short"] = float(dfe["MA_S"].iloc[-1]) if pd.notnull(dfe["MA_S"].iloc[-1]) else None
    out["ma_long"] = float(dfe["MA_L"].iloc[-1]) if pd.notnull(dfe["MA_L"].iloc[-1]) else None

    dfe["Return"] = dfe["Close"].pct_change()

    def cumret(days: int) -> Optional[float]:
        if len(dfe) < days + 1:
            return None
        return float(dfe["Close"].iloc[-1] / dfe["Close"].iloc[-days-1] - 1.0)

    out["rtn_12m"] = cumret(252)
    out["rtn_6m"] = cumret(126)
    out["rtn_1m"] = cumret(21)
    out["vol_90d"] = float(dfe["Return"].tail(90).std()) if len(dfe) >= 90 else None

    if len(dfe) >= 252:
        last_1y = dfe.tail(252).copy()
        roll_max = last_1y["Close"].cummax()
        dd = (last_1y["Close"] / roll_max - 1.0).min()
        out["max_drawdown_1y"] = float(dd)

    out["above_ma_long"] = bool(dfe["Close"].iloc[-1] > dfe["MA_L"].iloc[-1]) if pd.notnull(dfe["MA_L"].iloc[-1]) else None

    # Beta vs benchmark (SPY) over ~1y daily
    try:
        if bench_df is not None and not bench_df.empty and len(dfe) >= 252 and len(bench_df) >= 252:
            a = dfe.set_index("Date")["Close"].pct_change().tail(252)
            b = bench_df.set_index("Date")["Close"].pct_change().tail(252)
            j = pd.concat([a, b], axis=1, join="inner").dropna()
            if len(j) > 30:
                cov = np.cov(j.iloc[:,0], j.iloc[:,1])[0,1]
                var_m = np.var(j.iloc[:,1])
                out["beta_1y"] = float(cov / var_m) if var_m != 0 else None
    except Exception:
        pass

    return out

# ---------------------------- Optional Yahoo profile ----------------------------
def fetch_profile(ticker: str) -> Dict[str, Optional[str]]:
    out = {"sector": None, "industry": None, "name_yf": None}
    try:
        info = yf.Ticker(ticker).get_info() or {}
        out["sector"] = info.get("sector")
        out["industry"] = info.get("industry")
        out["name_yf"] = info.get("shortName") or info.get("longName")
    except Exception:
        pass
    return out

# ---------------------------- Snapshot builder ----------------------------
def safe_pe(price: Optional[float], eps_ttm: Optional[float]) -> Optional[float]:
    if price is None or eps_ttm is None or eps_ttm == 0:
        return None
    if eps_ttm < 0:
        return None
    return float(price / eps_ttm)

def basic_market_cap(price: Optional[float], shares: Optional[float]) -> Optional[float]:
    if price is None or shares is None:
        return None
    return float(price * shares)

def build_snapshot(ticker: str, prices_period: str, prices_interval: str, ma_s: int, ma_l: int) -> Dict[str, Any]:
    t = ticker.upper().strip()
    cik_title = get_cik_for_ticker(t)
    info = {"ticker": t, "company": None, "cik": None}
    if cik_title:
        info["cik"], info["company"] = cik_title[0], cik_title[1]

    facts = fetch_companyfacts(info["cik"]) if info["cik"] else None
    metrics = parse_facts(facts) if facts else {k: None for k in [
        "revenue_ttm","gross_profit_ttm","operating_income_ttm",
        "net_income_ttm","cfo_ttm","capex_ttm","assets","equity","debt","cash","shares_out"
    ]}

    prices = fetch_prices(t, period=prices_period, interval=prices_interval)
    bench = fetch_benchmark_prices("SPY", period=prices_period, interval=prices_interval)
    price_feats = compute_price_features(prices, ma_short=ma_s, ma_long=ma_l, bench_df=bench)
    last_close = float(prices["Close"].iloc[-1]) if prices is not None and not prices.empty else None

    eps_ttm = (metrics["net_income_ttm"] / metrics["shares_out"]) if all([
        metrics.get("net_income_ttm"), metrics.get("shares_out")
    ]) else None
    pe = safe_pe(last_close, eps_ttm)
    mcap = basic_market_cap(last_close, metrics.get("shares_out"))
    fcf = None
    if metrics.get("cfo_ttm") is not None and metrics.get("capex_ttm") is not None:
        fcf = float(metrics["cfo_ttm"] - abs(metrics["capex_ttm"]))
    fcf_yield = (fcf / mcap) if (fcf is not None and mcap not in (None, 0)) else None

    gross_margin = (metrics["gross_profit_ttm"] / metrics["revenue_ttm"]) if (metrics.get("gross_profit_ttm") and metrics.get("revenue_ttm")) else None
    op_margin = (metrics["operating_income_ttm"] / metrics["revenue_ttm"]) if (metrics.get("operating_income_ttm") and metrics.get("revenue_ttm")) else None
    net_margin = (metrics["net_income_ttm"] / metrics["revenue_ttm"]) if (metrics.get("net_income_ttm") and metrics.get("revenue_ttm")) else None
    roe = (metrics["net_income_ttm"] / metrics["equity"]) if (metrics.get("net_income_ttm") and metrics.get("equity")) else None
    leverage = (metrics["debt"] / metrics["assets"]) if (metrics.get("debt") and metrics.get("assets")) else None

    div_yield = compute_dividend_yield(t, last_close)
    payout_ratio = None
    try:
        if metrics.get("net_income_ttm") and metrics.get("shares_out"):
            div_ps = div_yield * last_close if (div_yield is not None and last_close) else None
            eps_ps = eps_ttm
            if div_ps is not None and eps_ps and eps_ps > 0:
                payout_ratio = float(div_ps / eps_ps)
    except Exception:
        pass

    profile = fetch_profile(t)
    if not info["company"] and profile.get("name_yf"):
        info["company"] = profile["name_yf"]

    snapshot = {
        "info": info,
        "profile": profile,
        "sec_url": f"https://www.sec.gov/edgar/browse/?CIK={info['cik']}&owner=exclude" if info.get("cik") else None,
        "last_close": last_close,
        "prices": prices.to_dict(orient="records") if prices is not None else None,
        "price_features": price_feats,
        "fundamentals": metrics,
        "derived": {
            "eps_ttm": eps_ttm,
            "pe": pe,
            "market_cap": mcap,
            "fcf_ttm": fcf,
            "fcf_yield": fcf_yield,
            "gross_margin": gross_margin,
            "operating_margin": op_margin,
            "net_margin": net_margin,
            "roe": roe,
            "leverage": leverage,
            "dividend_yield": div_yield,
            "payout_ratio": payout_ratio,
        },
        "_ts": time.time()
    }
    return snapshot

def save_snapshot(ticker: str, snap: Dict[str, Any]) -> None:
    path = _cache_path(f"snap_{ticker.upper()}.json")
    _write_cache(path, {"_ts": time.time(), "data": snap})

def load_snapshot(ticker: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(f"snap_{ticker.upper()}.json")
    cached = _read_cache(path)  # ✅ FIXED: was cache_file, now path
    return cached["data"] if cached else None

# ---------------------------- Universe helpers ----------------------------
def fetch_sp500_tickers(cache_days: int = 7) -> List[str]:
    """Best effort S&P 500 tickers via Wikipedia; cached. Fallback list if it fails."""
    cache_file = _cache_path("sp500_list.json")
    cached = _read_cache(cache_file)
    if cached and _days_ago(cached.get("_ts", 0)) < cache_days:
        return cached.get("tickers", [])
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers = list(pd.Series(tables[0]["Symbol"]).astype(str).str.replace(".", "-").str.upper().unique())
        _write_cache(cache_file, {"_ts": time.time(), "tickers": tickers})
        return tickers
    except Exception:
        fallback = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","BRK-B","TSLA","JPM","JNJ"]
        _write_cache(cache_file, {"_ts": time.time(), "tickers": fallback})
        return fallback

# ---------------------------- Intraday (Yahoo) ----------------------------
def fetch_intraday(ticker: str, lookback_minutes: int = 180, interval: str = "1m"):
    """
    Intraday 1-min bars via Yahoo Finance (educational use; typically delayed).
    Returns (df, meta). df has Date, Close. meta has 'last' and 'change' (session %).
    """
    try:
        df = yf.download(tickers=ticker, period="1d", interval=interval,
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None, None
        df = df.reset_index().rename(columns=str.title)  # Date, Close, ...
        if lookback_minutes and len(df) > lookback_minutes:
            df = df.tail(lookback_minutes)
        last = float(df["Close"].iloc[-1])
        first = float(df["Close"].iloc[0])
        chg = (last/first - 1.0) if first else None
        return df[["Date", "Close"]], {"last": last, "change": chg}
    except Exception:
        return None, None
