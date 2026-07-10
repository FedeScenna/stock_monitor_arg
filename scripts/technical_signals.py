#!/usr/bin/env python
"""
Murphy Technical-Analysis Signals
=================================
Scans the CEDEAR universe (underlying US-listed equities, USD) and scores each
name with the classic indicator toolkit from John J. Murphy, *Technical
Analysis of the Financial Markets*, combined into a net weight-of-evidence
rating (STRONG BUY / BUY / HOLD / SELL / STRONG SELL).

Ten rules vote +1 / -1 / 0, split into two families (Murphy's own grouping):

    Trend-following : MA cross (50/200), MA alignment, ADX/DI, MACD, OBV, Donchian
    Oscillators     : RSI, Stochastic, Williams %R, Rate of Change

Output: data/portfolio/technical_signals_YYYY-MM-DD.csv

Usage:
    /c/Users/feder/anaconda3/python.exe scripts/technical_signals.py
    /c/Users/feder/anaconda3/python.exe scripts/technical_signals.py --no-fetch
    /c/Users/feder/anaconda3/python.exe scripts/technical_signals.py --no-fetch --portfolio
    /c/Users/feder/anaconda3/python.exe scripts/technical_signals.py --no-fetch --limit 20
"""
import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import (
    CEDEAR_TICKERS, CEDEAR_SKIP, PORTFOLIO_ASSETS, DATA_DIR, PORTFOLIO_DIR,
    SCREEN_MIN_ROWS,
)
from src.data.fetcher import DataFetcher
from src.screening.murphy import murphy_signals, SIGNAL_COLS

UNIVERSE = {k: v for k, v in CEDEAR_TICKERS.items() if k not in CEDEAR_SKIP}

OUT_COLS = [
    "ticker", "name", "last_close", "rating", "murphy_score",
    "trend_score", "osc_score",
    "adx", "plus_di", "minus_di", "rsi", "stoch_k", "stoch_d",
    "williams_r", "roc", "macd", "sma50", "sma200", "atr",
    *SIGNAL_COLS,
]


def load_ohlcv(ticker: str) -> pd.DataFrame:
    """Read a cached OHLCV CSV and normalise to Date-sorted columns."""
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Murphy technical-analysis signal screen")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Use cached OHLCV (skip downloads)")
    parser.add_argument("--portfolio", action="store_true",
                        help="Only score the portfolio assets (default: full universe)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N tickers (testing)")
    args = parser.parse_args()

    universe = PORTFOLIO_ASSETS if args.portfolio else UNIVERSE
    if args.limit:
        universe = dict(list(universe.items())[: args.limit])
    tickers = list(universe.keys())

    # 1. OHLCV --------------------------------------------------------------
    if not args.no_fetch:
        print(f"Step 1 - Updating OHLCV for {len(tickers)} tickers ...")
        DataFetcher().update_all(tickers)
    else:
        print(f"Step 1 - Using cached OHLCV for {len(tickers)} tickers")

    # 2. Score --------------------------------------------------------------
    print(f"\nStep 2 - Scoring {len(tickers)} tickers (Murphy signals) ...")
    rows, skipped = [], []
    for ticker, name in universe.items():
        df = load_ohlcv(ticker)
        if len(df) < SCREEN_MIN_ROWS:
            skipped.append(ticker)
            continue
        sig = murphy_signals(df)
        rows.append({"ticker": ticker, "name": name, **sig})

    if skipped:
        print(f"  Skipped (insufficient history): {len(skipped)} tickers")
    if not rows:
        print("No tickers scored - aborting.")
        sys.exit(1)

    results = pd.DataFrame(rows)
    for col in OUT_COLS:
        if col not in results.columns:
            results[col] = np.nan
    results = results[OUT_COLS].sort_values(
        ["murphy_score", "trend_score"], ascending=[False, False]
    ).reset_index(drop=True)

    # 3. Save ---------------------------------------------------------------
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PORTFOLIO_DIR / f"technical_signals_{date.today()}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # 4. Console summary ----------------------------------------------------
    n = len(results)
    counts = results["rating"].value_counts()
    print("\n" + "=" * 70)
    print(f"  MURPHY TECHNICAL SIGNALS  -  {date.today()}  ({n} tickers)")
    print("=" * 70)
    for r in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]:
        print(f"  {r:<12}: {int(counts.get(r, 0)):>3}")

    def _show(title, sub):
        if sub.empty:
            return
        print(f"\n  {title}")
        print(f"  {'Ticker':<7} {'Score':>5} {'Trend':>5} {'Osc':>4} "
              f"{'ADX':>5} {'RSI':>5}  Rating")
        print("  " + "-" * 58)
        for _, r in sub.iterrows():
            adx = f"{r['adx']:.0f}" if pd.notna(r["adx"]) else " -"
            rsi = f"{r['rsi']:.0f}" if pd.notna(r["rsi"]) else " -"
            print(f"  {r['ticker']:<7} {int(r['murphy_score']):>5} "
                  f"{int(r['trend_score']):>5} {int(r['osc_score']):>4} "
                  f"{adx:>5} {rsi:>5}  {r['rating']}")

    _show("Top bullish (highest score):", results.head(15))
    _show("Top bearish (lowest score):",
          results.sort_values("murphy_score").head(10))


if __name__ == "__main__":
    main()
