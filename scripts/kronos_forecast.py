#!/usr/bin/env python
"""
Kronos OHLCV Foundation Model — Portfolio Price Forecasting
===========================================================
Downloads fresh OHLCV data for all portfolio assets (CEDEARs + Argentine
stocks as USD ADRs), then runs the Kronos-small model to forecast the next
21 trading days (~1 month) for each position.

Usage:
    /c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py
    /c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py --pred-days 10
    /c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py --no-fetch
    /c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py --model Kronos-base

Model options: Kronos-mini | Kronos-small (default) | Kronos-base
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

from config.settings import DATA_DIR, PORTFOLIO_DIR, PORTFOLIO_ASSETS
from model import Kronos, KronosTokenizer, KronosPredictor

# Tokenizer paired with each model variant
TOKENIZER_MAP = {
    "Kronos-mini":  "NeoQuasar/Kronos-Tokenizer-2k",
    "Kronos-small": "NeoQuasar/Kronos-Tokenizer-base",
    "Kronos-base":  "NeoQuasar/Kronos-Tokenizer-base",
}
MAX_CONTEXT_MAP = {
    "Kronos-mini":  2048,
    "Kronos-small": 512,
    "Kronos-base":  512,
}


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, lookback_days: int = 600) -> pd.DataFrame:
    """Download or update OHLCV CSV, return DataFrame with lowercase columns."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"{ticker}.csv"
    today = date.today()

    existing = None
    if csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=["Date"])
        existing["Date"] = pd.to_datetime(existing["Date"]).dt.date
        max_date = existing["Date"].max()
        fetch_start = (max_date + timedelta(days=1)).isoformat()
        if fetch_start >= today.isoformat():
            print(f"  [{ticker}] up-to-date ({max_date})")
            return _prepare(existing)
    else:
        fetch_start = (today - timedelta(days=lookback_days)).isoformat()

    print(f"  [{ticker}] fetching {fetch_start} to {today} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(ticker, start=fetch_start, end=today.isoformat(),
                          auto_adjust=True, progress=False)

    if raw.empty:
        print(f"  [{ticker}] WARNING: no data returned")
        return _prepare(existing) if existing is not None else pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.date

    if existing is not None:
        raw = pd.concat([existing, raw], ignore_index=True)

    raw = raw.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    raw.to_csv(csv_path, index=False)
    print(f"  [{ticker}] {len(raw)} rows saved")
    return _prepare(raw)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to lowercase and return sorted DataFrame."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Future trading-day timestamps ─────────────────────────────────────────────

def next_trading_days(n: int, from_date: date = None) -> pd.DatetimeIndex:
    """Generate n future trading days (Mon-Fri, skipping US holidays roughly)."""
    if from_date is None:
        from_date = date.today()
    days = []
    cur = from_date + timedelta(days=1)
    while len(days) < n:
        if cur.weekday() < 5:  # Mon-Fri
            days.append(pd.Timestamp(cur))
        cur += timedelta(days=1)
    return pd.DatetimeIndex(days)


# ── Kronos inference ───────────────────────────────────────────────────────────

def run_kronos(
    ticker: str,
    df: pd.DataFrame,
    predictor: KronosPredictor,
    pred_days: int = 7,
    sample_count: int = 20,
) -> dict:
    """
    Run Kronos forecast for one asset.
    Returns dict with ticker, last_price, predicted prices, and upside.
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        print(f"  [{ticker}] missing OHLC columns — skipping")
        return None

    # Use the last max_context rows as history
    max_ctx = predictor.max_context
    hist = df.tail(max_ctx).reset_index(drop=True)

    x_df = hist[["open", "high", "low", "close"]]
    # Kronos expects pd.Series (with .dt accessor), not DatetimeIndex
    x_timestamp = pd.Series(pd.to_datetime(hist["date"]).values)

    last_date = hist["date"].iloc[-1]
    if hasattr(last_date, "date"):
        last_date = last_date.date()
    else:
        last_date = date.fromisoformat(str(last_date))

    y_timestamp = pd.Series(next_trading_days(pred_days, from_date=last_date))

    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_days,
            T=1.0,
            top_p=0.9,
            sample_count=sample_count,
            verbose=False,
        )
    except Exception as exc:
        print(f"  [{ticker}] Kronos error: {exc}")
        return None

    last_close = float(hist["close"].iloc[-1])
    pred_close = pred_df["close"].values

    return {
        "ticker":        ticker,
        "name":          PORTFOLIO_ASSETS.get(ticker, ""),
        "last_close":    round(last_close, 4),
        "pred_close_7d":  round(float(pred_close[min(4,  pred_days-1)]), 4),
        "pred_close_14d": round(float(pred_close[min(9,  pred_days-1)]), 4),
        "pred_close_21d": round(float(pred_close[min(20, pred_days-1)]), 4),
        "pred_series":   pred_close.tolist(),
        "pred_dates":    [str(d.date()) for d in y_timestamp],
        "pred_high_max": round(float(pred_df["high"].max()), 4),
        "pred_low_min":  round(float(pred_df["low"].min()),  4),
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_report(results: list, pred_days: int):
    print("\n" + "=" * 80)
    print(f"  KRONOS FORECAST — {date.today()} — next {pred_days} trading days")
    print("=" * 80)
    print(f"{'Ticker':<6}  {'Name':<28}  {'Last':>8}  {'7d':>8}  {'14d':>8}  {'21d':>8}  {'Upside%':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: (x["pred_close_21d"] / x["last_close"]), reverse=True):
        upside = (r["pred_close_21d"] / r["last_close"] - 1) * 100
        flag = " *" if upside > 5 else ("  " if upside >= 0 else " !")
        print(
            f"{r['ticker']:<6}  {r['name'][:28]:<28}  "
            f"{r['last_close']:>8.2f}  "
            f"{r['pred_close_7d']:>8.2f}  "
            f"{r['pred_close_14d']:>8.2f}  "
            f"{r['pred_close_21d']:>8.2f}  "
            f"{upside:>+7.1f}%{flag}"
        )
    print("\n* = predicted upside >5%   ! = predicted downside")
    print("Note: Kronos forecasts OHLCV as a foundation model — treat as probabilistic")
    print("      signal, not a precise price target. Always combine with fundamental analysis.")


def save_results(results: list):
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        for i, (d, c) in enumerate(zip(r["pred_dates"], r["pred_series"])):
            rows.append({
                "ticker": r["ticker"],
                "name": r["name"],
                "forecast_date": d,
                "pred_day": i + 1,
                "pred_close": round(c, 4),
                "last_close": r["last_close"],
                "upside_pct": round((c / r["last_close"] - 1) * 100, 2),
            })
    df = pd.DataFrame(rows)
    out = PORTFOLIO_DIR / f"kronos_forecast_{date.today()}.csv"
    df.to_csv(out, index=False)
    print(f"\nFull forecast saved -> {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kronos portfolio price forecast")
    parser.add_argument("--model",     default="Kronos-small",
                        choices=["Kronos-mini", "Kronos-small", "Kronos-base"])
    parser.add_argument("--pred-days", type=int, default=21,
                        help="Trading days to forecast (default 21 ~ 1 month)")
    parser.add_argument("--samples",   type=int, default=20,
                        help="Monte Carlo sample paths (default 20, higher = smoother)")
    parser.add_argument("--no-fetch",  action="store_true",
                        help="Skip downloading, use cached CSVs only")
    parser.add_argument("--lookback",  type=int, default=600,
                        help="Days of history to download (default 600)")
    args = parser.parse_args()

    # 1. Fetch / update data
    print(f"\nStep 1 — Fetch OHLCV data ({'cached' if args.no_fetch else 'updating'})")
    datasets: dict[str, pd.DataFrame] = {}
    for ticker in PORTFOLIO_ASSETS:
        if args.no_fetch:
            csv_path = DATA_DIR / f"{ticker}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=["Date"])
                df["Date"] = pd.to_datetime(df["Date"]).dt.date
                datasets[ticker] = _prepare(df)
            else:
                print(f"  [{ticker}] no cached file, fetching...")
                datasets[ticker] = fetch_ohlcv(ticker, args.lookback)
        else:
            datasets[ticker] = fetch_ohlcv(ticker, args.lookback)

    # 2. Load Kronos
    print(f"\nStep 2 — Loading {args.model} from HuggingFace Hub ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    tokenizer_id = TOKENIZER_MAP[args.model]
    model_id = f"NeoQuasar/{args.model}"

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id)
    model = model.to(device)

    max_ctx = MAX_CONTEXT_MAP[args.model]
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_ctx)
    print(f"  {args.model} loaded ({max_ctx} context)")

    # 3. Run forecasts
    print(f"\nStep 3 — Forecasting {args.pred_days} trading days for {len(PORTFOLIO_ASSETS)} assets ...")
    results = []
    for ticker, df in datasets.items():
        if df.empty:
            print(f"  [{ticker}] no data, skipping")
            continue
        print(f"  [{ticker}] running Kronos ({len(df)} rows history) ...")
        res = run_kronos(ticker, df, predictor, pred_days=args.pred_days, sample_count=args.samples)
        if res:
            results.append(res)

    if not results:
        print("No results — aborting.")
        sys.exit(1)

    # 4. Report
    print_report(results, args.pred_days)
    save_results(results)


if __name__ == "__main__":
    main()
