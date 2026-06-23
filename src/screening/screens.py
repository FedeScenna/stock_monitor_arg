"""Screening logic shared by ``scripts/weekly_screen.py`` and ``app.py``.

All functions operate on the underlying US-listed equity's USD OHLCV data
(yfinance) — *not* the BYMA peso CEDEAR price.

Screens (independent, non-exclusive — a ticker may hit several):

    A  Tangible Value  P/TBV <= 1 AND FCF>0 AND D/E<1 AND current_ratio>=1 AND TBV growing YoY
    B  Below SMA200    last Close < 200-day SMA
    C  Below 20-wk MA  last weekly close < 20-period MA of weekly (W-FRI) closes
    D  EMA9/21 + MACD  EMA9 x EMA21 cross AND MACD x signal cross, same direction, last N bars
    E  RSI buy / sell  RSI14 <= 30 (buy) ; RSI14 >= 70 (sell)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    P_TBV_MAX, RSI_BUY, RSI_SELL, WMA_WEEKS,
    EMA_FAST, EMA_SLOW, CROSS_LOOKBACK,
)


# ---------------------------------------------------------------------------
# Indicators (single source of truth — also imported by app.py for charts)
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA / EMA / Bollinger / RSI / MACD columns to an OHLCV frame.

    Expects columns ``Date, Open, High, Low, Close, Volume`` sorted ascending.
    """
    df = df.copy()
    close = df["Close"]

    # SMAs
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    # EMAs ("dynamic means")
    df["EMA9"] = close.ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA21"] = close.ewm(span=EMA_SLOW, adjust=False).mean()

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_mid"] = bb_mid
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std

    # RSI (14) — Wilder EWM
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crossed(fast: pd.Series, slow: pd.Series, n: int, direction: str) -> bool:
    """True if ``fast`` crossed ``slow`` in ``direction`` within the last ``n`` bars."""
    f = fast.to_numpy(dtype="float64")
    s = slow.to_numpy(dtype="float64")
    if len(f) < 2:
        return False
    start = max(1, len(f) - n)
    for i in range(start, len(f)):
        if np.isnan(f[i]) or np.isnan(s[i]) or np.isnan(f[i - 1]) or np.isnan(s[i - 1]):
            continue
        if direction == "up" and f[i] > s[i] and f[i - 1] <= s[i - 1]:
            return True
        if direction == "down" and f[i] < s[i] and f[i - 1] >= s[i - 1]:
            return True
    return False


def _weekly_ma(df: pd.DataFrame, weeks: int) -> tuple[float, float]:
    """Return (last weekly close, last 20-week MA) from W-FRI resampled closes."""
    weekly = df.set_index("Date")["Close"].resample("W-FRI").last().dropna()
    if len(weekly) == 0:
        return np.nan, np.nan
    last_weekly = float(weekly.iloc[-1])
    wma = float(weekly.rolling(weeks).mean().iloc[-1]) if len(weekly) >= weeks else np.nan
    return last_weekly, wma


# ---------------------------------------------------------------------------
# Technical screens (B, C, D, E)
# ---------------------------------------------------------------------------

def technical_screens(
    df: pd.DataFrame,
    rsi_buy: float = RSI_BUY,
    rsi_sell: float = RSI_SELL,
    cross_lookback: int = CROSS_LOOKBACK,
    wma_weeks: int = WMA_WEEKS,
) -> dict:
    """Compute the price-based screens for a single ticker's OHLCV frame."""
    ind = compute_indicators(df)
    last = ind.iloc[-1]
    last_close = float(last["Close"])

    # B — below 200-day SMA
    sma200 = float(last["SMA200"]) if pd.notna(last["SMA200"]) else np.nan
    below_sma200 = bool(last_close < sma200) if not np.isnan(sma200) else False

    # C — below 20-week MA
    last_weekly, wma20 = _weekly_ma(df, wma_weeks)
    below_wma20 = bool(last_weekly < wma20) if not np.isnan(wma20) else False

    # D — EMA9/21 cross confirmed by MACD cross (same direction, last N bars)
    ema_up = _crossed(ind["EMA9"], ind["EMA21"], cross_lookback, "up")
    ema_dn = _crossed(ind["EMA9"], ind["EMA21"], cross_lookback, "down")
    macd_up = _crossed(ind["MACD"], ind["MACD_signal"], cross_lookback, "up")
    macd_dn = _crossed(ind["MACD"], ind["MACD_signal"], cross_lookback, "down")
    cross_up = bool(ema_up and macd_up)
    cross_down = bool(ema_dn and macd_dn)

    # E — RSI buy / sell zones
    rsi = float(last["RSI"]) if pd.notna(last["RSI"]) else np.nan
    rsi_buy_flag = bool(rsi <= rsi_buy) if not np.isnan(rsi) else False
    rsi_sell_flag = bool(rsi >= rsi_sell) if not np.isnan(rsi) else False

    # Liquidity — underlying US stock's 20-day avg dollar volume
    dollar_vol_20d = float((ind["Close"] * ind["Volume"]).tail(20).mean())

    return {
        "last_close": round(last_close, 4),
        "dollar_vol_20d": round(dollar_vol_20d, 0),
        "sma200": round(sma200, 4) if not np.isnan(sma200) else np.nan,
        "wma20": round(wma20, 4) if not np.isnan(wma20) else np.nan,
        "ema9": round(float(last["EMA9"]), 4) if pd.notna(last["EMA9"]) else np.nan,
        "ema21": round(float(last["EMA21"]), 4) if pd.notna(last["EMA21"]) else np.nan,
        "macd": round(float(last["MACD"]), 4) if pd.notna(last["MACD"]) else np.nan,
        "macd_signal": round(float(last["MACD_signal"]), 4) if pd.notna(last["MACD_signal"]) else np.nan,
        "rsi": round(rsi, 2) if not np.isnan(rsi) else np.nan,
        "below_sma200": below_sma200,
        "below_wma20": below_wma20,
        "cross_up": cross_up,
        "cross_down": cross_down,
        "rsi_buy": rsi_buy_flag,
        "rsi_sell": rsi_sell_flag,
    }


# ---------------------------------------------------------------------------
# Tangible value screen (A) — Michael Burry "tangible value"
# ---------------------------------------------------------------------------

def tangible_value_screen(fund: dict | None, p_tbv_max: float = P_TBV_MAX) -> dict:
    """Evaluate the Burry tangible-value stack from a fundamentals row.

    ``fund`` is one row from FundamentalsFetcher (may be None / partly NaN).
    debt_to_equity is expected as a *ratio* (yfinance reports a percentage,
    converted upstream by FundamentalsFetcher).
    """
    fund = fund or {}
    p_tbv = fund.get("p_tbv", np.nan)
    fcf = fund.get("fcf", np.nan)
    dte = fund.get("debt_to_equity", np.nan)
    cr = fund.get("current_ratio", np.nan)
    tbv_yoy = fund.get("tbv_yoy", np.nan)

    q_ptbv = bool(pd.notna(p_tbv) and 0 < p_tbv <= p_tbv_max)
    q_fcf = bool(pd.notna(fcf) and fcf > 0)
    q_dte = bool(pd.notna(dte) and dte < 1.0)
    q_cr = bool(pd.notna(cr) and cr >= 1.0)
    q_tbv_growth = bool(pd.notna(tbv_yoy) and tbv_yoy > 0)

    value_hit = all([q_ptbv, q_fcf, q_dte, q_cr, q_tbv_growth])

    return {
        "p_tbv": round(float(p_tbv), 4) if pd.notna(p_tbv) else np.nan,
        "tbvps": fund.get("tbvps", np.nan),
        "fcf": fund.get("fcf", np.nan),
        "debt_to_equity": round(float(dte), 4) if pd.notna(dte) else np.nan,
        "current_ratio": round(float(cr), 4) if pd.notna(cr) else np.nan,
        "tbv_yoy": round(float(tbv_yoy), 4) if pd.notna(tbv_yoy) else np.nan,
        "q_ptbv": q_ptbv,
        "q_fcf": q_fcf,
        "q_dte": q_dte,
        "q_cr": q_cr,
        "q_tbv_growth": q_tbv_growth,
        "value_hit": value_hit,
    }


# ---------------------------------------------------------------------------
# Confluence — how many *buy-oriented* screens a ticker hits
# ---------------------------------------------------------------------------

BUY_SCREENS = ("value_hit", "below_sma200", "below_wma20", "cross_up", "rsi_buy")


def confluence(flags: dict) -> int:
    """Count of buy-oriented screens that are True for this ticker."""
    return int(sum(bool(flags.get(k)) for k in BUY_SCREENS))
