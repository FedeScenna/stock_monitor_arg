"""Murphy technical-analysis strategies.

Implements the classic indicator toolkit and trading signals from
John J. Murphy, *Technical Analysis of the Financial Markets*, and combines
them into a net "weight of evidence" score (Murphy's own framing: no single
indicator is decisive — you weigh trend tools and oscillators together).

The book groups its tools into two families, and so does this module:

  Trend-following (work best in trending markets)
    * Moving-average alignment & the 50/200 Golden/Death cross   (ch. "Moving Averages")
    * MACD                                                        (ch. "Moving Averages")
    * Directional movement: +DI / -DI / ADX  (Wilder)            (ch. "The Directional System")
    * On-Balance Volume (OBV)                                    (ch. "Volume and Open Interest")
    * Donchian channel / 52-week breakout                        (ch. "Basic Concepts of Trend")

  Oscillators (work best in trading ranges; flag overbought/oversold reversals)
    * RSI (Wilder)                                               (ch. "Oscillators and Contrary Opinion")
    * Stochastic %K / %D (slow)                                  (ch. "Oscillators and Contrary Opinion")
    * Williams %R                                                (ch. "Oscillators and Contrary Opinion")
    * Rate of Change / momentum                                  (ch. "Oscillators and Contrary Opinion")

ATR (Wilder) is also computed — Murphy uses it for volatility-scaled stops.

All functions operate on the underlying US-listed equity's USD OHLCV frame
(``Date, Open, High, Low, Close, Volume`` sorted ascending), the same
convention as :mod:`src.screening.screens`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    ADX_LEN, ADX_TREND_MIN, STOCH_K, STOCH_D, STOCH_SMOOTH,
    STOCH_OVERSOLD, STOCH_OVERBOUGHT, WILLIAMS_LEN,
    WILLIAMS_OVERSOLD, WILLIAMS_OVERBOUGHT, ROC_LEN,
    OBV_SLOPE_LEN, DONCHIAN_LEN, ATR_LEN,
    RSI_BUY, RSI_SELL, CROSS_LOOKBACK,
)
from src.screening.screens import compute_indicators, _crossed


# ---------------------------------------------------------------------------
# Wilder smoothing (RMA) — the moving average Wilder used for ATR / DI / ADX.
# ---------------------------------------------------------------------------

def _wilder(series: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (a.k.a. RMA): EWM with alpha = 1/n."""
    return series.ewm(alpha=1.0 / n, adjust=False).mean()


# ---------------------------------------------------------------------------
# Individual Murphy indicators
# ---------------------------------------------------------------------------

def average_true_range(df: pd.DataFrame, n: int = ATR_LEN) -> pd.Series:
    """Wilder's Average True Range."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return _wilder(tr, n)


def directional_movement(df: pd.DataFrame, n: int = ADX_LEN) -> pd.DataFrame:
    """Wilder's Directional Movement system: +DI, -DI and ADX.

    +DI/-DI measure the share of range moving up vs down; ADX measures the
    *strength* of the trend (regardless of direction). Murphy reads ADX >= 25
    as a trending market where trend-following tools are reliable.
    """
    high, low = df["High"], df["Low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    atr = average_true_range(df, n).replace(0, np.nan)
    plus_di = 100 * _wilder(plus_dm, n) / atr
    minus_di = 100 * _wilder(minus_dm, n) / atr

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = _wilder(dx, n)

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})


def stochastic(df: pd.DataFrame, k: int = STOCH_K, d: int = STOCH_D,
               smooth: int = STOCH_SMOOTH) -> pd.DataFrame:
    """Slow stochastic oscillator (%K smoothed, %D = SMA of %K)."""
    low_k = df["Low"].rolling(k).min()
    high_k = df["High"].rolling(k).max()
    rng = (high_k - low_k).replace(0, np.nan)
    fast_k = 100 * (df["Close"] - low_k) / rng
    slow_k = fast_k.rolling(smooth).mean()          # slow %K
    slow_d = slow_k.rolling(d).mean()               # %D
    return pd.DataFrame({"stoch_k": slow_k, "stoch_d": slow_d})


def williams_r(df: pd.DataFrame, n: int = WILLIAMS_LEN) -> pd.Series:
    """Williams %R (0 to -100; -20 overbought, -80 oversold)."""
    high_n = df["High"].rolling(n).max()
    low_n = df["Low"].rolling(n).min()
    rng = (high_n - low_n).replace(0, np.nan)
    return -100 * (high_n - df["Close"]) / rng


def rate_of_change(close: pd.Series, n: int = ROC_LEN) -> pd.Series:
    """Rate of Change / momentum, in percent."""
    return 100 * (close / close.shift(n) - 1)


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """Granville's On-Balance Volume — running volume signed by daily direction."""
    direction = np.sign(df["Close"].diff().fillna(0.0))
    return (direction * df["Volume"]).cumsum()


# ---------------------------------------------------------------------------
# Compute-all: base indicators (from screens.py) + Murphy toolkit
# ---------------------------------------------------------------------------

def murphy_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the screens.py indicators + the Murphy toolkit added."""
    out = compute_indicators(df)           # SMA50/200, EMA9/21, Bollinger, RSI, MACD
    out["ATR"] = average_true_range(out, ATR_LEN)
    dm = directional_movement(out, ADX_LEN)
    out["plus_di"], out["minus_di"], out["ADX"] = dm["plus_di"], dm["minus_di"], dm["adx"]
    st = stochastic(out, STOCH_K, STOCH_D, STOCH_SMOOTH)
    out["stoch_k"], out["stoch_d"] = st["stoch_k"], st["stoch_d"]
    out["williams_r"] = williams_r(out, WILLIAMS_LEN)
    out["ROC"] = rate_of_change(out["Close"], ROC_LEN)
    out["OBV"] = on_balance_volume(out)
    out["donchian_high"] = out["High"].rolling(DONCHIAN_LEN).max()
    out["donchian_low"] = out["Low"].rolling(DONCHIAN_LEN).min()
    return out


# ---------------------------------------------------------------------------
# Signal engine — each rule votes +1 (bullish) / -1 (bearish) / 0 (neutral)
# ---------------------------------------------------------------------------

def _sign(cond_bull: bool, cond_bear: bool) -> int:
    if cond_bull and not cond_bear:
        return 1
    if cond_bear and not cond_bull:
        return -1
    return 0


# Trend-following votes vs oscillator (mean-reversion) votes — Murphy keeps
# these two families conceptually separate, so we score them separately too.
TREND_SIGNALS = ("ma_cross", "ma_align", "adx_di", "macd", "obv", "donchian")
OSC_SIGNALS = ("rsi", "stoch", "williams", "roc")


def murphy_signals(
    df: pd.DataFrame,
    cross_lookback: int = CROSS_LOOKBACK,
    adx_trend_min: float = ADX_TREND_MIN,
    rsi_buy: float = RSI_BUY,
    rsi_sell: float = RSI_SELL,
) -> dict:
    """Compute Murphy's indicator votes and a combined weight-of-evidence score.

    Returns a flat dict of indicator values, per-rule votes (+1/-1/0), the
    ``trend_score`` / ``osc_score`` sub-totals, an overall ``murphy_score`` and
    a human ``rating`` (STRONG BUY / BUY / HOLD / SELL / STRONG SELL).
    """
    ind = murphy_indicators(df)
    last = ind.iloc[-1]
    close = float(last["Close"])

    def val(col):
        return float(last[col]) if pd.notna(last[col]) else np.nan

    sma50, sma200 = val("SMA50"), val("SMA200")
    ema9, ema21 = val("EMA9"), val("EMA21")
    adx, plus_di, minus_di = val("ADX"), val("plus_di"), val("minus_di")
    macd_v, macd_sig = val("MACD"), val("MACD_signal")
    rsi = val("RSI")
    stoch_k, stoch_d = val("stoch_k"), val("stoch_d")
    wr = val("williams_r")
    roc = val("ROC")
    don_hi, don_lo = val("donchian_high"), val("donchian_low")

    votes: dict[str, int] = {}

    # --- Trend-following family --------------------------------------------
    # 1. Golden / Death cross (SMA50 vs SMA200) + recent-cross confirmation.
    if not np.isnan(sma50) and not np.isnan(sma200):
        gc = _crossed(ind["SMA50"], ind["SMA200"], cross_lookback, "up")
        dc = _crossed(ind["SMA50"], ind["SMA200"], cross_lookback, "down")
        votes["ma_cross"] = _sign(sma50 > sma200 or gc, sma50 < sma200 or dc)
    else:
        votes["ma_cross"] = 0

    # 2. Moving-average alignment ("stacked" MAs = strong, clean trend).
    if not any(np.isnan(x) for x in (ema9, ema21, sma50)):
        votes["ma_align"] = _sign(
            close > ema9 > ema21 > sma50,
            close < ema9 < ema21 < sma50,
        )
    else:
        votes["ma_align"] = 0

    # 3. Directional system: DI cross, but only when ADX confirms a trend.
    if not any(np.isnan(x) for x in (adx, plus_di, minus_di)) and adx >= adx_trend_min:
        votes["adx_di"] = _sign(plus_di > minus_di, minus_di > plus_di)
    else:
        votes["adx_di"] = 0

    # 4. MACD vs signal line (+ recent cross).
    if not np.isnan(macd_v) and not np.isnan(macd_sig):
        mu = _crossed(ind["MACD"], ind["MACD_signal"], cross_lookback, "up")
        md = _crossed(ind["MACD"], ind["MACD_signal"], cross_lookback, "down")
        votes["macd"] = _sign(macd_v > macd_sig or mu, macd_v < macd_sig or md)
    else:
        votes["macd"] = 0

    # 5. On-Balance Volume slope confirms/denies the price move.
    obv = ind["OBV"].dropna()
    if len(obv) > OBV_SLOPE_LEN:
        obv_chg = obv.iloc[-1] - obv.iloc[-1 - OBV_SLOPE_LEN]
        votes["obv"] = _sign(obv_chg > 0, obv_chg < 0)
    else:
        votes["obv"] = 0

    # 6. Donchian / 52-week breakout (new high = bullish breakout).
    if not np.isnan(don_hi) and not np.isnan(don_lo):
        # Compare to the channel *excluding* today to detect a fresh breakout.
        prior_hi = float(ind["High"].iloc[:-1].rolling(DONCHIAN_LEN).max().iloc[-1]) \
            if len(ind) > DONCHIAN_LEN else don_hi
        prior_lo = float(ind["Low"].iloc[:-1].rolling(DONCHIAN_LEN).min().iloc[-1]) \
            if len(ind) > DONCHIAN_LEN else don_lo
        votes["donchian"] = _sign(close >= prior_hi, close <= prior_lo)
    else:
        votes["donchian"] = 0

    # --- Oscillator family (contrarian: extremes flag reversals) -----------
    # 7. RSI overbought / oversold.
    votes["rsi"] = _sign(
        (not np.isnan(rsi)) and rsi <= rsi_buy,
        (not np.isnan(rsi)) and rsi >= rsi_sell,
    )
    # 8. Stochastic: %K/%D cross inside oversold / overbought zones.
    if not np.isnan(stoch_k) and not np.isnan(stoch_d):
        votes["stoch"] = _sign(
            stoch_k <= STOCH_OVERSOLD and stoch_k >= stoch_d,
            stoch_k >= STOCH_OVERBOUGHT and stoch_k <= stoch_d,
        )
    else:
        votes["stoch"] = 0
    # 9. Williams %R extremes.
    votes["williams"] = _sign(
        (not np.isnan(wr)) and wr <= WILLIAMS_OVERSOLD,
        (not np.isnan(wr)) and wr >= WILLIAMS_OVERBOUGHT,
    )
    # 10. Rate of change / momentum sign.
    votes["roc"] = _sign(
        (not np.isnan(roc)) and roc > 0,
        (not np.isnan(roc)) and roc < 0,
    )

    trend_score = sum(votes[k] for k in TREND_SIGNALS)
    osc_score = sum(votes[k] for k in OSC_SIGNALS)
    total = trend_score + osc_score

    return {
        "last_close": round(close, 4),
        # indicator readings
        "sma50": round(sma50, 4) if not np.isnan(sma50) else np.nan,
        "sma200": round(sma200, 4) if not np.isnan(sma200) else np.nan,
        "adx": round(adx, 2) if not np.isnan(adx) else np.nan,
        "plus_di": round(plus_di, 2) if not np.isnan(plus_di) else np.nan,
        "minus_di": round(minus_di, 2) if not np.isnan(minus_di) else np.nan,
        "macd": round(macd_v, 4) if not np.isnan(macd_v) else np.nan,
        "rsi": round(rsi, 2) if not np.isnan(rsi) else np.nan,
        "stoch_k": round(stoch_k, 2) if not np.isnan(stoch_k) else np.nan,
        "stoch_d": round(stoch_d, 2) if not np.isnan(stoch_d) else np.nan,
        "williams_r": round(wr, 2) if not np.isnan(wr) else np.nan,
        "roc": round(roc, 2) if not np.isnan(roc) else np.nan,
        "atr": round(val("ATR"), 4) if not np.isnan(val("ATR")) else np.nan,
        # per-rule votes
        **{f"sig_{k}": v for k, v in votes.items()},
        # aggregates
        "trend_score": int(trend_score),
        "osc_score": int(osc_score),
        "murphy_score": int(total),
        "rating": rating_from_score(total),
    }


def rating_from_score(score: int) -> str:
    """Map the net signal score to a human rating."""
    if score >= 4:
        return "STRONG BUY"
    if score >= 2:
        return "BUY"
    if score <= -4:
        return "STRONG SELL"
    if score <= -2:
        return "SELL"
    return "HOLD"


# Column order shared by the script CSV and the app table.
SIGNAL_COLS = [f"sig_{k}" for k in (*TREND_SIGNALS, *OSC_SIGNALS)]
