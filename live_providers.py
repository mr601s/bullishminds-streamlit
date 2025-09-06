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

import json
import threading
import time
from typing import Optional
import pandas as pd

try:
    import websocket
except ImportError:
    websocket = None

class PolygonStream:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.api_key = os.environ.get("POLYGON_API_KEY", "")
        self.ws = None
        self.data = []
        self.running = False
        
    def start(self):
        if not websocket or not self.api_key:
            return
        self.running = True
        self.thread = threading.Thread(target=self._connect)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if hasattr(self, 'ws') and self.ws:
            self.ws.close()
            
    def _connect(self):
        try:
            url = f"wss://socket.polygon.io/stocks"
            self.ws = websocket.WebSocketApp(url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error)
            self.ws.run_forever()
        except Exception:
            pass
            
    def _on_open(self, ws):
        auth = {"action": "auth", "params": self.api_key}
        subscribe = {"action": "subscribe", "params": f"T.{self.ticker}"}
        ws.send(json.dumps(auth))
        ws.send(json.dumps(subscribe))
        
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if isinstance(data, list):
                for item in data:
                    if item.get("ev") == "T":
                        self.data.append({
                            "Date": pd.to_datetime(item["t"], unit="ms"),
                            "Close": float(item["p"])
                        })
        except Exception:
            pass
            
    def _on_error(self, ws, error):
        pass
        
    def get_df(self) -> Optional[pd.DataFrame]:
        if not self.data:
            return None
        return pd.DataFrame(self.data[-100:])  # Keep last 100 points

class FinnhubStream:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.api_key = os.environ.get("FINNHUB_API_KEY", "")
        self.ws = None
        self.data = []
        self.running = False
        
    def start(self):
        if not websocket or not self.api_key:
            return
        self.running = True
        self.thread = threading.Thread(target=self._connect)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if hasattr(self, 'ws') and self.ws:
            self.ws.close()
            
    def _connect(self):
        try:
            url = f"wss://ws.finnhub.io?token={self.api_key}"
            self.ws = websocket.WebSocketApp(url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error)
            self.ws.run_forever()
        except Exception:
            pass
            
    def _on_open(self, ws):
        subscribe = {"type": "subscribe", "symbol": self.ticker}
        ws.send(json.dumps(subscribe))
        
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get("type") == "trade":
                for trade in data.get("data", []):
                    self.data.append({
                        "Date": pd.to_datetime(trade["t"], unit="ms"),
                        "Close": float(trade["p"])
                    })
        except Exception:
            pass
            
    def _on_error(self, ws, error):
        pass
        
    def get_df(self) -> Optional[pd.DataFrame]:
        if not self.data:
            return None
        return pd.DataFrame(self.data[-100:])  # Keep last 100 points
