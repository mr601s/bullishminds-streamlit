import os
import json
import time
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

# Streamlit secrets for API keys/environment
try:
    import streamlit as st
    for k, v in st.secrets.items():
        os.environ[k] = v
except Exception:
    pass

if os.path.exists('.env'):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _read_cache(fname):
    try:
        with open(fname, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_cache(fname, d):
    try:
        with open(fname, "w") as f:
            json.dump(d, f)
    except Exception:
        pass

def _read_df_from_cache(fname):
    d = _read_cache(fname)
    if "data" in d:
        try:
            return pd.read_json(json.dumps(d["data"]), orient="split")
        except Exception:
            return None
    return None

def _days_ago(ts):
    return (time.time() - ts) / (24*3600)

# ------------- SEC/S&P/Universe Helpers ----------------
def get_sec_tickers():
    fname = os.path.join(CACHE_DIR,'sec_companies.json')
    if not os.path.exists(fname):
        url = "https://www.sec.gov/files/company_tickers.json"
        df = pd.read_json(url).transpose()
        df.to_json(fname)
    else:
        df = pd.read_json(fname)
    return df

def get_cik_for_ticker(ticker: str) -> Optional[Tuple[str, str]]:
    df = get_sec_tickers()
    match = df[df['ticker'].str.upper() == ticker.upper()]
    if not match.empty:
        cik = str(match.iloc[0]['cik_str']).zfill(10)
        company = match.iloc[0]['title']
        return cik, company
    return None

def fetch_companyfacts(cik: str) -> Optional[dict]:
    fname = os.path.join(CACHE_DIR, f"facts_{cik}.json")
    if not os.path.exists(fname) or _days_ago(_read_cache(fname).get('_ts', 0)) > 4:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            j = pd.read_json(url)
            j['_ts'] = time.time()
            with open(fname, "w") as f:
                json.dump(j, f)
        except Exception:
            return None
    else:
        with open(fname) as f:
            j = json.load(f)
    return j

def parse_facts(facts: dict) -> dict:
    # You can expand this with richer metrics as desired.
    out = {
        "revenue_ttm": None, "gross_profit_ttm": None, "operating_income_ttm": None, 
        "net_income_ttm": None, "cfo_ttm": None, "capex_ttm": None, "assets": None, 
        "equity": None, "debt": None, "cash": None, "shares_out": None
    }
    # This is a placeholder. You should adapt with your favorite SEC XBRL extract logic.
    return out

# ------------ Price Fetching: Robust Close Handling -----------

def _get_close_any(row):
    """Return close price from any common field."""
    for field in ["Close", "close", "c", "adjclose", "Adj Close", "AdjClose"]:
        if field in row and pd.notnull(row[field]):
            return row[field]
    return None

def fetch_prices(ticker: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    key = f"px_{ticker}_{period}_{interval}.json"
    cache_file = os.path.join(CACHE_DIR, key)
    cached_meta = _read_cache(cache_file)
    if (cached := _read_df_from_cache(cache_file)) is not None and _days_ago(cached_meta.get("_ts", 0)) < 1:
        return cached
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df.reset_index().rename(columns=str.title)
        # Robust close-extraction
        if "Close" not in df.columns:
            if "Adj Close" in df.columns:
                df['Close'] = df['Adj Close']
            elif "close" in df.columns:
                df['Close'] = df['close']
            elif "adjclose" in df.columns:
                df['Close'] = df['adjclose']
            else:
                df['Close'] = df.apply(_get_close_any, axis=1)
        _write_cache(cache_file, {"_ts": time.time(), "data": json.loads(df.to_json(orient="split"))})
        return df
    except Exception as e:
        print(f"fetch_prices ERROR for {ticker}: {e}")
        return None

def fetch_intraday(ticker: str, lookback_minutes: int = 180, interval: str = "1m"):
    try:
        import yfinance as yf
        df = yf.download(tickers=ticker, period="1d", interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None, None
        df = df.reset_index().rename(columns=str.title)
        if "Close" not in df.columns:
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            elif "close" in df.columns:
                df["Close"] = df["close"]
            else:
                df["Close"] = df.apply(_get_close_any, axis=1)
        if lookback_minutes and len(df) > lookback_minutes:
            df = df.tail(lookback_minutes)
        last = float(df["Close"].iloc[-1])
        first = float(df["Close"].iloc[0])
        chg = (last/first - 1.0) if first else None
        # Date logic
        if "Datetime" in df.columns:
            df["Date"] = df["Datetime"]
        elif "Date" not in df.columns:
            df["Date"] = df.index
        return df[["Date", "Close"]], {"last": last, "change": chg}
    except Exception as e:
        print(f"fetch_intraday ERROR for {ticker}: {e}")
        return None, None

# ---------- Profile/Universe ---------

def fetch_profile(ticker: str) -> dict:
    # Wire up yfinance or Finnhub as needed for richer company info.
    return {}

# -------- Feature Computation Examples -------
def compute_price_features(df, ma_short=20, ma_long=50, bench_df=None):
    feats = {}
    if df is None or df.empty or "Close" not in df.columns:
        return feats
    df["ma_short"] = df["Close"].rolling(ma_short).mean()
    df["ma_long"] = df["Close"].rolling(ma_long).mean()
    feats["session_return"] = float(df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1.0 if len(df) > 1 else None
    return feats

def basic_market_cap(close, shares_out):
    if close is not None and shares_out:
        return float(close) * float(shares_out)
    return None

def safe_pe(close, eps):
    try:
        if close and eps and eps != 0:
            return float(close) / eps
    except Exception:
        return None
    return None

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
    price_feats = compute_price_features(prices, ma_short=ma_s, ma_long=ma_l)
    last_close = None
    if prices is not None and not prices.empty:
        for close_field in ["Close", "close", "c", "Adj Close", "adjclose"]:
            if close_field in prices.columns:
                try:
                    last_close = float(prices[close_field].iloc[-1])
                    break
                except Exception:
                    continue
        if last_close is None:
            try:
                last_close = float(prices.apply(_get_close_any, axis=1).dropna().iloc[-1])
            except Exception:
                pass
    eps_ttm = (metrics.get("net_income_ttm") / metrics.get("shares_out")) if (metrics.get("net_income_ttm") and metrics.get("shares_out")) else None
    pe = safe_pe(last_close, eps_ttm)
    mcap = basic_market_cap(last_close, metrics.get("shares_out"))
    gross_margin = (metrics.get("gross_profit_ttm") / metrics.get("revenue_ttm")) if (metrics.get("gross_profit_ttm") and metrics.get("revenue_ttm")) else None
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
            "gross_margin": gross_margin,
        },
        "_ts": time.time()
    }
    if last_close is None and prices is not None:
        print(f"[WARN] Could not determine last_close for {ticker}. Price columns: {prices.columns.tolist()}")
    return snapshot

# If you need more helpers from your original file (for tickers list parsing etc.), copy them above here.
