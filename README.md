---
title: BullishMinds Markets — Investor Education
emoji: 📈
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: true
---

# BullishMinds Markets (Standalone)

An education-first stock evaluator with **live/delayed intraday**, **explainable factor scores**, comparisons, watchlists, and a simple **universe screen**.

> **Not investment advice.** Data may be delayed or incomplete. For learning purposes only.

## Features
- **Live intraday strip** (Yahoo 1m delayed; **optional** Polygon/Finnhub real-time)
- **Single**: prices + gauges (Value, Quality, Momentum, Risk) with plain-English explanations
- **Compare**: up to 3 tickers
- **Watchlist**: score many at once + CSV export
- **Universe**: S&P 500 or custom list; percentiles by factor; CSV export
- **Learn**: glossary/tooltips for each concept

## Setup (Hugging Face)
1. **Add Variables & Secrets** (Settings → Variables and secrets)
   - **Variable**:  
     `SEC_USER_AGENT = BullishMindsMarkets/1.0 (you@example.com)`
   - **Secrets** *(optional for real-time)*:  
     `POLYGON_API_KEY = <your polygon key>`  
     `FINNHUB_API_KEY = <your finnhub key>`
2. **Restart the Space**.
3. Open the app → **Single** tab → choose **Live data source**:
   - *Yahoo (1m, delayed)* → no keys required  
   - *Polygon (real-time)* → requires `POLYGON_API_KEY`  
   - *Finnhub (real-time)* → requires `FINNHUB_API_KEY`

## License
- **Code**: Apache-2.0 (see `LICENSE`)
- **Educational content in /content**: CC BY-NC 4.0
- **Logos/brand**: All Rights Reserved
