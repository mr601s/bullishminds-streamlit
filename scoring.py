# Copyright (c) 2025 Bullish Minds AI
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from typing import Dict, Any, Tuple

def scale_between(x, good, bad, invert=False) -> float:
    """Scale to 0–100; if invert=True, higher x => worse."""
    if x is None:
        return None
    a, b = (bad, good) if invert else (good, bad)
    lo, hi = min(a, b), max(a, b)
    if x <= lo: return 100.0 if invert else 0.0
    if x >= hi: return 0.0 if invert else 100.0
    frac = (x - lo) / (hi - lo)
    return float((1 - frac) * 100.0) if invert else float(frac * 100.0)

def score_value(pe, fcf_yield, cfg) -> Tuple[float, Dict[str, Any]]:
    parts, expl = [], []
    if fcf_yield is not None:
        vy = scale_between(fcf_yield, cfg["fcf_yield_good"], cfg["fcf_yield_bad"], invert=False)
        parts.append(vy); expl.append(f"FCF Yield contribution: {vy:.0f} (FCF/MarketCap = {fcf_yield:.2%})")
    if pe is not None:
        pe_s = scale_between(pe, cfg["pe_good"], cfg["pe_bad"], invert=True)
        parts.append(pe_s); expl.append(f"P/E contribution: {pe_s:.0f} (P/E = {pe:.1f})")
    score = sum(parts)/len(parts) if parts else None
    return score, {"details": expl}

def score_quality(roe, gross_margin, cfg) -> Tuple[float, Dict[str, Any]]:
    parts, expl = [], []
    if roe is not None:
        s = scale_between(roe, cfg["roe_good"], cfg["roe_bad"], invert=False)
        parts.append(s); expl.append(f"ROE contribution: {s:.0f} (ROE = {roe:.1%})")
    if gross_margin is not None:
        s = scale_between(gross_margin, cfg["gross_margin_good"], cfg["gross_margin_bad"], invert=False)
        parts.append(s); expl.append(f"Gross Margin contribution: {s:.0f} (GM = {gross_margin:.1%})")
    score = sum(parts)/len(parts) if parts else None
    return score, {"details": expl}

def score_momentum(r12, r6, r1, above_ma_long, cfg) -> Tuple[float, Dict[str, Any]]:
    parts, expl = [], []
    if r12 is not None:
        s = scale_between(r12, 0.25, -0.15, invert=False)
        parts.append(s); expl.append(f"12m return: {s:.0f} ({r12:.1%})")
    if r6 is not None:
        s = scale_between(r6, 0.20, -0.10, invert=False)
        parts.append(s); expl.append(f"6m return: {s:.0f} ({r6:.1%})")
    if r1 is not None:
        s = scale_between(r1, 0.05, -0.05, invert=False)
        parts.append(s); expl.append(f"1m return: {s:.0f} ({r1:.1%})")
    if above_ma_long is not None:
        s = 100.0 if above_ma_long else 20.0
        parts.append(s); expl.append(f"200DMA trend: {s:.0f} ({'above' if above_ma_long else 'below'})")
    score = sum(parts)/len(parts) if parts else None
    return score, {"details": expl}

def score_risk(leverage, vol_90d, drawdown_1y, beta_1y, cfg) -> Tuple[float, Dict[str, Any]]:
    parts, expl = [], []
    if leverage is not None:
        s = scale_between(leverage, cfg["leverage_low"], cfg["leverage_high"], invert=True)
        parts.append(s); expl.append(f"Leverage: {s:.0f} (Debt/Assets = {leverage:.1%})")
    if vol_90d is not None:
        s = scale_between(vol_90d, cfg["vol_low"], cfg["vol_high"], invert=True)
        parts.append(s); expl.append(f"Volatility (90d σ): {s:.0f} ({vol_90d:.2%})")
    if drawdown_1y is not None:
        s = scale_between(drawdown_1y, -0.05, -0.60, invert=False)
        parts.append(s); expl.append(f"Max drawdown 1y: {s:.0f} ({drawdown_1y:.1%})")
    if beta_1y is not None:
        s = scale_between(beta_1y, 0.8, 1.6, invert=True)
        parts.append(s); expl.append(f"Beta (1y vs SPY): {s:.0f} (β = {beta_1y:.2f})")
    score = sum(parts)/len(parts) if parts else None
    return score, {"details": expl}

def blend_scores(parts: Dict[str, float], weights: Dict[str, float]) -> float:
    weighted, total_w = [], 0.0
    for k, v in parts.items():
        if v is None:
            continue
        w = float(weights.get(k, 0.0))
        weighted.append(v * w)
        total_w += w
    if total_w == 0:
        return None
    return float(sum(weighted) / total_w)

def confidence(fund: Dict[str, Any], price_feats: Dict[str, Any]) -> str:
    has_core = all(fund.get(k) is not None for k in ["net_income_ttm","cfo_ttm","capex_ttm","equity","shares_out"])
    has_px = price_feats.get("ma_long") is not None
    if has_core and has_px:
        return "High"
    if has_px or has_core:
        return "Medium"
    return "Low"
