#!/usr/bin/env python
"""
Weekly CEDEAR Value + Technical Screener
========================================
Scans the full BYMA CEDEAR universe (underlying US-listed equities, USD) and
flags each name against several independent, non-exclusive screens:

    A  Tangible Value (Burry)  P/TBV<=1 AND FCF>0 AND D/E<1 AND CurrRatio>=1 AND TBV growing
    B  Below SMA200            last Close < 200-day SMA
    C  Below 20-week MA        last weekly close < 20-week MA
    D  EMA9/21 + MACD cross    EMA9xEMA21 and MACDxsignal cross, same dir, last 5 bars
    E  RSI buy / sell          RSI14 <= 30 (buy) ; RSI14 >= 70 (sell)

Output: data/portfolio/weekly_screen_YYYY-MM-DD.csv

Usage:
    /c/Users/feder/anaconda3/python.exe scripts/weekly_screen.py
    /c/Users/feder/anaconda3/python.exe scripts/weekly_screen.py --no-fetch
    /c/Users/feder/anaconda3/python.exe scripts/weekly_screen.py --no-fetch --skip-fundamentals
    /c/Users/feder/anaconda3/python.exe scripts/weekly_screen.py --no-fetch --limit 15
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
    CEDEAR_TICKERS, CEDEAR_SKIP, DATA_DIR, PORTFOLIO_DIR,
    RSI_BUY, RSI_SELL, CROSS_LOOKBACK, WMA_WEEKS, MIN_DOLLAR_VOL,
    SCREEN_MIN_ROWS,
)
from src.data.fetcher import DataFetcher
from src.data.fundamentals import FundamentalsFetcher
from src.screening.screens import (
    technical_screens, tangible_value_screen, confluence,
)

UNIVERSE = {k: v for k, v in CEDEAR_TICKERS.items() if k not in CEDEAR_SKIP}

# Final column order for the output CSV.
OUT_COLS = [
    "ticker", "name", "last_close", "dollar_vol_20d", "rsi",
    "sma200", "wma20", "ema9", "ema21", "macd", "macd_signal",
    "below_sma200", "below_wma20", "cross_up", "cross_down",
    "rsi_buy", "rsi_sell",
    "p_tbv", "tbvps", "fcf", "debt_to_equity", "current_ratio", "tbv_yoy",
    "q_ptbv", "q_fcf", "q_dte", "q_cr", "q_tbv_growth", "value_hit",
    "confluence",
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
    parser = argparse.ArgumentParser(description="Weekly CEDEAR value + technical screen")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Use cached OHLCV / fundamentals (skip downloads)")
    parser.add_argument("--skip-fundamentals", action="store_true",
                        help="Technical screens only (no tangible-value screen)")
    parser.add_argument("--rsi-buy", type=float, default=RSI_BUY)
    parser.add_argument("--rsi-sell", type=float, default=RSI_SELL)
    parser.add_argument("--cross-lookback", type=int, default=CROSS_LOOKBACK)
    parser.add_argument("--min-dollar-vol", type=float, default=MIN_DOLLAR_VOL,
                        help="Drop names below this 20-day avg $ volume")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only screen the first N tickers (testing)")
    args = parser.parse_args()

    universe = dict(list(UNIVERSE.items())[: args.limit]) if args.limit else UNIVERSE
    tickers = list(universe.keys())

    # 1. OHLCV --------------------------------------------------------------
    if not args.no_fetch:
        print(f"Step 1 - Updating OHLCV for {len(tickers)} tickers ...")
        DataFetcher().update_all(tickers)
    else:
        print(f"Step 1 - Using cached OHLCV for {len(tickers)} tickers")

    # 2. Fundamentals -------------------------------------------------------
    fund_map: dict[str, dict] = {}
    if not args.skip_fundamentals:
        ff = FundamentalsFetcher()
        if args.no_fetch:
            fdf = ff.load_latest()
            if fdf is None:
                print("  No cached fundamentals found - run without --no-fetch first, "
                      "or pass --skip-fundamentals.")
        else:
            print("\nStep 2 - Fetching fundamentals ...")
            fdf = ff.update_all(tickers)
        if fdf is not None:
            fund_map = {r["ticker"]: r for r in fdf.to_dict("records")}
    else:
        print("Step 2 - Skipping fundamentals (technical-only run)")

    # 3. Screen -------------------------------------------------------------
    print(f"\nStep 3 - Screening {len(tickers)} tickers ...")
    rows = []
    skipped = []
    for ticker, name in universe.items():
        df = load_ohlcv(ticker)
        if len(df) < SCREEN_MIN_ROWS:
            skipped.append(ticker)
            continue

        tech = technical_screens(
            df, rsi_buy=args.rsi_buy, rsi_sell=args.rsi_sell,
            cross_lookback=args.cross_lookback, wma_weeks=WMA_WEEKS,
        )
        val = tangible_value_screen(fund_map.get(ticker))

        row = {"ticker": ticker, "name": name, **tech, **val}
        row["confluence"] = confluence(row)
        rows.append(row)

    if skipped:
        print(f"  Skipped (insufficient history): {len(skipped)} tickers")
    if not rows:
        print("No tickers screened - aborting.")
        sys.exit(1)

    results = pd.DataFrame(rows)

    # Liquidity floor
    if args.min_dollar_vol > 0:
        before = len(results)
        results = results[results["dollar_vol_20d"] >= args.min_dollar_vol]
        print(f"  Liquidity filter: kept {len(results)}/{before} "
              f"(>= ${args.min_dollar_vol:,.0f} avg $vol)")

    # Order columns (tolerate any missing) and sort by confluence
    for col in OUT_COLS:
        if col not in results.columns:
            results[col] = np.nan
    results = results[OUT_COLS].sort_values(
        ["confluence", "p_tbv"], ascending=[False, True]
    ).reset_index(drop=True)

    # 4. Save ---------------------------------------------------------------
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PORTFOLIO_DIR / f"weekly_screen_{date.today()}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # 5. Console summary ----------------------------------------------------
    n = len(results)
    print("\n" + "=" * 70)
    print(f"  WEEKLY CEDEAR SCREEN  -  {date.today()}  ({n} tickers)")
    print("=" * 70)
    print(f"  A  Tangible value (Burry) : {int(results['value_hit'].sum()):>3}")
    print(f"  B  Below SMA200           : {int(results['below_sma200'].sum()):>3}")
    print(f"  C  Below 20-week MA       : {int(results['below_wma20'].sum()):>3}")
    print(f"  D  EMA9/21 + MACD cross up : {int(results['cross_up'].sum()):>3}"
          f"   (down: {int(results['cross_down'].sum())})")
    print(f"  E  RSI buy (<= {args.rsi_buy:.0f})        : {int(results['rsi_buy'].sum()):>3}"
          f"   (sell >= {args.rsi_sell:.0f}: {int(results['rsi_sell'].sum())})")

    top = results[results["confluence"] > 0].head(20)
    if not top.empty:
        print("\n  Top confluence (most buy screens hit):")
        print(f"  {'Ticker':<7} {'Conf':>4}  {'P/TBV':>7}  {'RSI':>5}  Screens")
        print("  " + "-" * 60)
        for _, r in top.iterrows():
            hits = []
            if r["value_hit"]:    hits.append("VALUE")
            if r["below_sma200"]: hits.append("<SMA200")
            if r["below_wma20"]:  hits.append("<20wMA")
            if r["cross_up"]:     hits.append("CROSS^")
            if r["rsi_buy"]:      hits.append("RSI-buy")
            if r["rsi_sell"]:     hits.append("RSI-sell")
            ptbv = f"{r['p_tbv']:.2f}" if pd.notna(r["p_tbv"]) else "  -"
            rsi = f"{r['rsi']:.0f}" if pd.notna(r["rsi"]) else " -"
            print(f"  {r['ticker']:<7} {int(r['confluence']):>4}  {ptbv:>7}  {rsi:>5}  {', '.join(hits)}")


if __name__ == "__main__":
    main()
