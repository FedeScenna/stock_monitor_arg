#!/usr/bin/env python
"""
Kronos OHLCV Foundation Model — CEDEAR Price Forecasting
========================================================
Runs the Kronos model to forecast the next 21 trading days (~1 month) for every
stock that has a CEDEAR (the full universe), regardless of portfolio membership.
Pass --portfolio to restrict to current holdings instead.

Usage:
    /c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py              # all CEDEARs (default)
    /c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py --portfolio  # holdings only
    /c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py --pred-days 10
    /c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py --no-fetch
    /c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py --model Kronos-small

Model options: Kronos-mini | Kronos-small | Kronos-base (default)
GPU (CUDA) is used automatically when available.
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

from config.settings import (
    DATA_DIR, PORTFOLIO_DIR, PORTFOLIO_ASSETS, CEDEAR_TICKERS, CEDEAR_SKIP,
)
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
    name: str = "",
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
        # Per-sample paths: (sample_count, pred_len, 6) cols = open,high,low,close,vol,amount
        samples = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_days,
            T=1.0,
            top_p=0.9,
            sample_count=sample_count,
            verbose=False,
            return_samples=True,
        )
    except Exception as exc:
        print(f"  [{ticker}] Kronos error: {exc}")
        return None

    last_close = float(hist["close"].iloc[-1])

    # Mean path across samples = the point forecast (matches the old averaged output)
    close_samples = samples[:, :, 3]                 # (sample_count, pred_len)
    pred_close = close_samples.mean(axis=0)          # (pred_len,)
    high_mean = samples[:, :, 1].mean(axis=0)
    low_mean = samples[:, :, 2].mean(axis=0)

    # Directional confidence: fraction of sample paths whose 21-day move agrees
    # with the predicted (mean) direction. 1.0 = unanimous, ~0.5 = coin flip.
    final_returns = close_samples[:, -1] / last_close - 1.0   # per-sample 21d return
    mean_return = float(pred_close[-1] / last_close - 1.0)
    pred_dir = 1.0 if mean_return >= 0 else -1.0
    confidence = float(np.mean(np.sign(final_returns) == pred_dir))

    return {
        "ticker":        ticker,
        "name":          name,
        "last_close":    round(last_close, 4),
        "pred_close_7d":  round(float(pred_close[min(4,  pred_days-1)]), 4),
        "pred_close_14d": round(float(pred_close[min(9,  pred_days-1)]), 4),
        "pred_close_21d": round(float(pred_close[min(20, pred_days-1)]), 4),
        "pred_series":   pred_close.tolist(),
        "pred_dates":    [str(d.date()) for d in y_timestamp],
        "pred_high_max": round(float(high_mean.max()), 4),
        "pred_low_min":  round(float(low_mean.min()),  4),
        "confidence":    round(confidence, 3),
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_report(results: list, pred_days: int, min_confidence: float = 0.55):
    print("\n" + "=" * 92)
    print(f"  KRONOS FORECAST — {date.today()} — next {pred_days} trading days")
    print("=" * 92)
    print(f"{'Ticker':<6}  {'Name':<28}  {'Last':>8}  {'7d':>8}  {'14d':>8}  {'21d':>8}  {'Upside%':>8}  {'Conf':>5}")
    print("-" * 92)
    for r in sorted(results, key=lambda x: (x["pred_close_21d"] / x["last_close"]), reverse=True):
        upside = (r["pred_close_21d"] / r["last_close"] - 1) * 100
        conf = r.get("confidence", float("nan")) * 100
        flag = " *" if upside > 5 else ("  " if upside >= 0 else " !")
        # mark predictions the model is not confident enough about
        conf_flag = "" if conf >= min_confidence * 100 else "  (low)"
        print(
            f"{r['ticker']:<6}  {r['name'][:28]:<28}  "
            f"{r['last_close']:>8.2f}  "
            f"{r['pred_close_7d']:>8.2f}  "
            f"{r['pred_close_14d']:>8.2f}  "
            f"{r['pred_close_21d']:>8.2f}  "
            f"{upside:>+7.1f}%{flag}  {conf:>4.0f}%{conf_flag}"
        )
    n_conf = sum(1 for r in results if r.get("confidence", 0) >= min_confidence)
    print(f"\n* = predicted upside >5%   ! = predicted downside")
    print(f"Conf = share of sample paths agreeing on the 21d direction. "
          f"{n_conf}/{len(results)} meet the >={min_confidence*100:.0f}% bar; "
          f"'(low)' = below it (treat as no signal).")
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
                "confidence": r.get("confidence", float("nan")),
            })
    df = pd.DataFrame(rows)
    out = PORTFOLIO_DIR / f"kronos_forecast_{date.today()}.csv"
    df.to_csv(out, index=False)
    print(f"\nFull forecast saved -> {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kronos portfolio price forecast")
    parser.add_argument("--model",     default="Kronos-base",
                        choices=["Kronos-mini", "Kronos-small", "Kronos-base"])
    parser.add_argument("--pred-days", type=int, default=21,
                        help="Trading days to forecast (default 21 ~ 1 month)")
    parser.add_argument("--samples",   type=int, default=20,
                        help="Monte Carlo sample paths (default 20, higher = smoother)")
    parser.add_argument("--no-fetch",  action="store_true",
                        help="Skip downloading, use cached CSVs only")
    parser.add_argument("--lookback",  type=int, default=600,
                        help="Days of history to download (default 600)")
    parser.add_argument("--portfolio", action="store_true",
                        help="Forecast only the portfolio holdings instead of the full CEDEAR universe")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Only forecast the first N assets (testing / partial runs)")
    args = parser.parse_args()

    # Default: forecast every CEDEAR (the full universe), regardless of portfolio
    # membership. Pass --portfolio to restrict to current holdings.
    if args.portfolio:
        assets = dict(PORTFOLIO_ASSETS)
    else:
        assets = {t: n for t, n in CEDEAR_TICKERS.items() if t not in CEDEAR_SKIP}
    if args.limit:
        assets = dict(list(assets.items())[: args.limit])

    # 1. Fetch / update data
    print(f"\nStep 1 — Fetch OHLCV data ({'cached' if args.no_fetch else 'updating'})"
          f" for {len(assets)} assets")
    datasets: dict[str, pd.DataFrame] = {}
    for ticker in assets:
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
    print(f"\nStep 3 — Forecasting {args.pred_days} trading days for {len(assets)} assets ...")
    results = []
    for ticker, df in datasets.items():
        if df.empty:
            print(f"  [{ticker}] no data, skipping")
            continue
        print(f"  [{ticker}] running Kronos ({len(df)} rows history) ...")
        res = run_kronos(ticker, df, predictor, pred_days=args.pred_days,
                         sample_count=args.samples, name=assets.get(ticker, ""))
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
