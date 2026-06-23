#!/usr/bin/env python
"""
CEDEAR Full Screen — Kronos + Momentum Buy Recommender
======================================================
Downloads USD price data for all BYMA CEDEARs NOT in the current portfolio,
runs Kronos-small 21-day price forecasts, combines with momentum/RSI signals,
and outputs ranked buy recommendations.

Usage:
    /c/Users/feder/anaconda3/python.exe scripts/cedear_full_screen.py
    /c/Users/feder/anaconda3/python.exe scripts/cedear_full_screen.py --top 30
    /c/Users/feder/anaconda3/python.exe scripts/cedear_full_screen.py --no-fetch
"""
import argparse
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import Kronos, KronosTokenizer, KronosPredictor
from config.settings import (
    CEDEAR_TICKERS, CEDEAR_SKIP, PORTFOLIO_ASSETS,
    DATA_DIR, PORTFOLIO_DIR, START_DATE,
)

# Portfolio holdings to EXCLUDE from recommendations
PORTFOLIO_HELD = set(PORTFOLIO_ASSETS.keys())

# Screen the full official CEDEAR universe, excluding held positions and skip list
SCREEN_UNIVERSE = {
    k: v for k, v in CEDEAR_TICKERS.items()
    if k not in PORTFOLIO_HELD and k not in CEDEAR_SKIP
}
MIN_ROWS      = 100   # minimum history rows required for Kronos


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{ticker}.csv"
    today    = date.today().isoformat()

    existing = None
    if csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=["Date"])
        existing["Date"] = pd.to_datetime(existing["Date"]).dt.date
        max_date = str(existing["Date"].max())
        if max_date >= today:
            return _prep(existing)
        fetch_start = str(existing["Date"].max() + timedelta(days=1))
    else:
        fetch_start = START_DATE

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(ticker, start=fetch_start, end=today,
                          auto_adjust=True, progress=False)

    if raw.empty:
        return _prep(existing) if existing is not None else pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.date

    if existing is not None:
        raw = pd.concat([existing, raw], ignore_index=True)
    raw = (raw.drop_duplicates(subset=["Date"])
              .sort_values("Date").reset_index(drop=True))
    raw.to_csv(csv_path, index=False)
    return _prep(raw)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df["date"]  = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ── Momentum signals ───────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss  = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    if loss == 0:
        return 100.0
    return round(100 - 100 / (1 + gain / loss), 2)


def momentum_signals(df: pd.DataFrame) -> dict:
    close = df["close"].astype(float)
    last  = float(close.iloc[-1])

    def ret(n):
        return (last / close.iloc[-n-1] - 1) * 100 if len(close) > n else np.nan

    ma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else np.nan
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
    vol   = float(close.pct_change().dropna().std() * np.sqrt(252) * 100)
    ret1y = ret(252)
    sharpe = (ret1y / vol) if (not np.isnan(ret1y) and vol > 0) else np.nan
    high52 = float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max())

    return {
        "last_close":   round(last, 4),
        "ret_1m":       round(ret(21), 2)  if not np.isnan(ret(21))  else np.nan,
        "ret_3m":       round(ret(63), 2)  if not np.isnan(ret(63))  else np.nan,
        "ret_6m":       round(ret(126), 2) if not np.isnan(ret(126)) else np.nan,
        "ret_1y":       round(ret1y, 2)    if not np.isnan(ret1y)    else np.nan,
        "rsi":          compute_rsi(close),
        "above_ma50":   last > ma50  if not np.isnan(ma50)  else False,
        "above_ma200":  last > ma200 if not np.isnan(ma200) else False,
        "golden_cross": (ma50 > ma200) if (not np.isnan(ma50) and not np.isnan(ma200)) else False,
        "vol_ann":      round(vol, 2),
        "sharpe_1y":    round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "dd_52w":       round((last / high52 - 1) * 100, 2),
    }


# ── Kronos forecast ────────────────────────────────────────────────────────────

def kronos_predict(df: pd.DataFrame, predictor: KronosPredictor,
                   pred_days: int = 21, samples: int = 10) -> dict | None:
    df = df.tail(predictor.max_context).reset_index(drop=True)
    x_df = df[["open", "high", "low", "close"]].reset_index(drop=True)
    x_ts = pd.Series(df["date"].values)

    last_date = df["date"].iloc[-1].date()
    future_dates = []
    d = last_date + timedelta(days=1)
    while len(future_dates) < pred_days:
        if d.weekday() < 5:
            future_dates.append(pd.Timestamp(d))
        d += timedelta(days=1)
    y_ts = pd.Series(future_dates)

    try:
        pred = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_days, T=1.0, top_p=0.9,
            sample_count=samples, verbose=False,
        )
    except Exception as exc:
        return None

    closes = pred["close"].values
    last   = float(df["close"].iloc[-1])
    return {
        "pred_7d":     round(float(closes[min(4,  pred_days-1)]), 4),
        "pred_14d":    round(float(closes[min(9,  pred_days-1)]), 4),
        "pred_21d":    round(float(closes[min(20, pred_days-1)]), 4),
        "upside_21d":  round((float(closes[min(20, pred_days-1)]) / last - 1) * 100, 2),
        "pred_high":   round(float(pred["high"].max()), 4),
        "pred_low":    round(float(pred["low"].min()),  4),
    }


# ── Composite scoring ──────────────────────────────────────────────────────────

def composite_score(row: dict) -> float:
    """
    Weighted score (higher = stronger buy signal):
      40% Kronos 21-day upside
      25% 3-month momentum (rank-based)
      20% 6-month momentum (rank-based)
      15% RSI quality (40-65 = ideal zone)
    """
    # Kronos upside: map [-30%, +30%] → [0, 1]
    k_up   = row.get("upside_21d", 0) or 0
    k_score = np.clip((k_up + 30) / 60, 0, 1)

    rsi = row.get("rsi", 50) or 50
    if 40 <= rsi <= 65:
        rsi_s = 1.0
    elif rsi < 30:
        rsi_s = 0.3
    elif rsi <= 75:
        rsi_s = 0.65
    else:
        rsi_s = 0.2

    # MA structure bonus
    ma_bonus = (
        (0.4 if row.get("above_ma50")  else 0) +
        (0.4 if row.get("above_ma200") else 0) +
        (0.2 if row.get("golden_cross") else 0)
    )

    return round(k_score * 0.40 + rsi_s * 0.15 + ma_bonus * 0.10, 4)


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_top(df: pd.DataFrame, top_n: int):
    df = df.sort_values("upside_21d", ascending=False).head(top_n)
    print("\n" + "=" * 100)
    print(f"  TOP {top_n} CEDEAR BUY CANDIDATES (not in portfolio)  —  {date.today()}")
    print("=" * 100)
    print(
        f"{'#':>3}  {'Ticker':<7}  {'Name':<28}  {'Price':>8}  "
        f"{'1M%':>6}  {'3M%':>6}  {'6M%':>6}  "
        f"{'RSI':>5}  {'Pred21d':>9}  {'Up21d%':>7}  {'Score':>6}"
    )
    print("-" * 100)
    for i, (_, r) in enumerate(df.iterrows(), 1):
        flag = " GC" if r.get("golden_cross") else (" ^" if r.get("above_ma200") else "")
        print(
            f"{i:>3}  {r['ticker']:<7}  {str(r['name'])[:28]:<28}  "
            f"{r['last_close']:>8.2f}  "
            f"{r['ret_1m']:>+6.1f}  {r['ret_3m']:>+6.1f}  {r['ret_6m']:>+6.1f}  "
            f"{r['rsi']:>5.1f}  "
            f"{r['pred_21d']:>9.2f}  {r['upside_21d']:>+6.1f}%  "
            f"{r['score']:>6.3f}{flag}"
        )
    print("\nGC = Golden Cross (MA50>MA200)   ^ = above MA200")
    print("Up21d% = Kronos predicted price change over next 21 trading days")


def print_avoid(df: pd.DataFrame, n: int = 10):
    worst = df.sort_values("upside_21d").head(n)
    print("\n" + "=" * 80)
    print("  AVOID (worst Kronos signal)")
    print("=" * 80)
    print(f"{'Ticker':<7}  {'Name':<28}  {'3M%':>6}  {'RSI':>5}  {'Up21d%':>8}")
    print("-" * 60)
    for _, r in worst.iterrows():
        print(f"{r['ticker']:<7}  {str(r['name'])[:28]:<28}  "
              f"{r['ret_3m']:>+6.1f}  {r['rsi']:>5.1f}  {r['upside_21d']:>+7.1f}%")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",      type=int,  default=25)
    parser.add_argument("--pred-days",type=int,  default=21)
    parser.add_argument("--samples",  type=int,  default=10)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--model",    default="Kronos-small",
                        choices=["Kronos-mini", "Kronos-small", "Kronos-base"])
    args = parser.parse_args()

    TOKENIZER = {"Kronos-mini": "NeoQuasar/Kronos-Tokenizer-2k",
                 "Kronos-small": "NeoQuasar/Kronos-Tokenizer-base",
                 "Kronos-base":  "NeoQuasar/Kronos-Tokenizer-base"}
    MAX_CTX   = {"Kronos-mini": 2048, "Kronos-small": 512, "Kronos-base": 512}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n_universe = len(SCREEN_UNIVERSE)

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    if not args.no_fetch:
        print(f"\nStep 1 — Downloading {n_universe} tickers (USD prices from US exchanges) ...")
    else:
        print(f"\nStep 1 — Loading cached data for {n_universe} tickers ...")

    datasets: dict[str, pd.DataFrame] = {}
    failed = []
    for i, (ticker, name) in enumerate(SCREEN_UNIVERSE.items(), 1):
        prefix = f"  [{i:>3}/{n_universe}] {ticker:<7}"
        if args.no_fetch:
            p = DATA_DIR / f"{ticker}.csv"
            if p.exists():
                df = pd.read_csv(p, parse_dates=["Date"])
                df["Date"] = pd.to_datetime(df["Date"]).dt.date
                datasets[ticker] = _prep(df)
                print(f"{prefix} loaded ({len(datasets[ticker])} rows)")
            else:
                print(f"{prefix} no cache, fetching...")
                df = fetch_ohlcv(ticker)
                if df.empty:
                    failed.append(ticker)
                else:
                    datasets[ticker] = df
                    print(f"{prefix} {len(df)} rows")
        else:
            df = fetch_ohlcv(ticker)
            if df.empty:
                print(f"{prefix} NO DATA — skipping")
                failed.append(ticker)
            else:
                datasets[ticker] = df
                print(f"{prefix} {len(df)} rows  {df['date'].min().date()} to {df['date'].max().date()}")

    if failed:
        print(f"\n  No data: {', '.join(failed)}")

    # ── 2. Load Kronos ────────────────────────────────────────────────────────
    print(f"\nStep 2 — Loading {args.model} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER[args.model])
    model     = Kronos.from_pretrained(f"NeoQuasar/{args.model}")
    model     = model.to(device)
    predictor = KronosPredictor(model, tokenizer, device=device,
                                max_context=MAX_CTX[args.model])
    print(f"  Ready on {device}")

    # ── 3. Compute signals + Kronos forecast ──────────────────────────────────
    print(f"\nStep 3 — Computing momentum + Kronos forecast for {len(datasets)} assets ...")
    rows = []
    for i, (ticker, df) in enumerate(datasets.items(), 1):
        name = SCREEN_UNIVERSE[ticker]
        print(f"  [{i:>3}/{len(datasets)}] {ticker:<7} ({len(df)} rows) ...", end=" ")

        if len(df) < MIN_ROWS:
            print("skip (insufficient history)")
            continue

        # Momentum
        mom = momentum_signals(df)

        # Kronos
        kr = kronos_predict(df, predictor, pred_days=args.pred_days,
                            samples=args.samples)
        if kr is None:
            print("Kronos error")
            continue

        row = {"ticker": ticker, "name": name, **mom, **kr}
        row["score"] = composite_score(row)
        rows.append(row)
        print(f"up21d={kr['upside_21d']:+.1f}%  rsi={mom['rsi']:.0f}  score={row['score']:.3f}")

    if not rows:
        print("No results. Aborting.")
        sys.exit(1)

    results = pd.DataFrame(rows)

    # ── 4. Save full results ──────────────────────────────────────────────────
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PORTFOLIO_DIR / f"cedear_screen_{date.today()}.csv"
    cols = ["ticker", "name", "last_close", "ret_1m", "ret_3m", "ret_6m", "ret_1y",
            "rsi", "above_ma50", "above_ma200", "golden_cross",
            "vol_ann", "sharpe_1y", "dd_52w",
            "pred_7d", "pred_14d", "pred_21d", "upside_21d",
            "pred_high", "pred_low", "score"]
    results[cols].sort_values("upside_21d", ascending=False).to_csv(out_path, index=False)
    print(f"\nFull results saved -> {out_path}")

    # ── 5. Print report ───────────────────────────────────────────────────────
    print_top(results, top_n=args.top)
    print_avoid(results, n=10)

    # ── 6. Sector summary ─────────────────────────────────────────────────────
    bullish = results[results["upside_21d"] > 5]
    bearish = results[results["upside_21d"] < -5]
    neutral = results[(results["upside_21d"] >= -5) & (results["upside_21d"] <= 5)]
    print(f"\nKronos signal summary across {len(results)} CEDEARs:")
    print(f"  Bullish (>+5%):  {len(bullish):>3} tickers")
    print(f"  Neutral (-5/+5): {len(neutral):>3} tickers")
    print(f"  Bearish (<-5%):  {len(bearish):>3} tickers")
    print(f"\nMedian predicted upside: {results['upside_21d'].median():+.1f}%")
    print(f"Mean predicted upside:   {results['upside_21d'].mean():+.1f}%")


if __name__ == "__main__":
    main()
