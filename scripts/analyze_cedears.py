#!/usr/bin/env python
"""
Fetch OHLCV data for all CEDEARs traded on BYMA and rank them
using momentum, RSI, and risk-adjusted signals.

Usage:
    /c/Users/feder/anaconda3/python.exe scripts/analyze_cedears.py
    /c/Users/feder/anaconda3/python.exe scripts/analyze_cedears.py --top 20
    /c/Users/feder/anaconda3/python.exe scripts/analyze_cedears.py --no-fetch
"""
import argparse
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import CEDEAR_TICKERS, CEDEAR_SKIP, DATA_DIR as CEDEAR_DATA_DIR

BYMA_CEDEARS = {k: v for k, v in CEDEAR_TICKERS.items() if k not in CEDEAR_SKIP}


# -- Data fetching --------------------------------------------------------------

def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV for one ticker via yfinance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.date
    return raw


def fetch_all(tickers: dict, data_dir: Path, lookback_days: int = 400) -> dict:
    """Fetch or load cached OHLCV for all tickers."""
    data_dir.mkdir(parents=True, exist_ok=True)
    end = date.today()
    start = (end - timedelta(days=lookback_days)).isoformat()
    end_str = end.isoformat()

    results = {}
    failed = []
    print(f"\nFetching {len(tickers)} CEDEARs ({start} to {end_str})...\n")

    for i, (ticker, name) in enumerate(tickers.items(), 1):
        csv_path = data_dir / f"{ticker}.csv"
        prefix = f"[{i:>3}/{len(tickers)}] {ticker:<6}"

        # Load existing
        existing = None
        if csv_path.exists():
            try:
                existing = pd.read_csv(csv_path, parse_dates=["Date"])
                existing["Date"] = pd.to_datetime(existing["Date"]).dt.date
                max_date = existing["Date"].max()
                fetch_start = (max_date + timedelta(days=1)).isoformat()
                if fetch_start >= end_str:
                    print(f"{prefix} up-to-date ({max_date})")
                    results[ticker] = existing
                    continue
            except Exception:
                existing = None
                fetch_start = start
        else:
            fetch_start = start

        try:
            df = fetch_ticker(ticker, fetch_start, end_str)
        except Exception as exc:
            print(f"{prefix} ERROR: {exc}")
            failed.append(ticker)
            if existing is not None:
                results[ticker] = existing
            continue

        if df.empty:
            print(f"{prefix} no data returned")
            failed.append(ticker)
            if existing is not None:
                results[ticker] = existing
            continue

        if existing is not None:
            df = pd.concat([existing, df], ignore_index=True)

        df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        print(f"{prefix} {len(df):>4} rows  {df['Date'].min()} -> {df['Date'].max()}")
        results[ticker] = df

    if failed:
        print(f"\nFailed / no data: {', '.join(failed)}")
    return results


# -- Signal computation ---------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)


def compute_signals(ticker: str, df: pd.DataFrame) -> dict | None:
    """Compute momentum + RSI signals. Returns None if insufficient data."""
    min_rows = 200
    if len(df) < min_rows:
        return None

    close = df.set_index("Date")["Close"].sort_index().astype(float)
    last = close.iloc[-1]

    def ret(days: int) -> float:
        if len(close) < days + 1:
            return np.nan
        return (last / close.iloc[-(days + 1)] - 1) * 100

    # Momentum returns
    ret_5d   = ret(5)
    ret_21d  = ret(21)
    ret_63d  = ret(63)
    ret_126d = ret(126)
    ret_252d = ret(252) if len(close) >= 253 else np.nan

    # RSI
    rsi = compute_rsi(close)

    # Moving averages
    ma50  = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    above_ma50  = last > ma50
    above_ma200 = last > ma200
    golden_cross = ma50 > ma200  # bullish when 50 > 200

    # Volatility (annualised)
    daily_returns = close.pct_change().dropna()
    vol_ann = daily_returns.std() * np.sqrt(252) * 100  # in %

    # Sharpe (excess return over vol, using 1-year return)
    sharpe = (ret_252d / vol_ann) if (not np.isnan(ret_252d) and vol_ann > 0) else np.nan

    # Drawdown from 52-week high
    high_52w = close.iloc[-252:].max() if len(close) >= 252 else close.max()
    drawdown = (last / high_52w - 1) * 100

    return {
        "ticker":       ticker,
        "name":         BYMA_CEDEARS.get(ticker, ""),
        "price":        round(last, 2),
        "ret_5d":       round(ret_5d, 2),
        "ret_21d":      round(ret_21d, 2),
        "ret_63d":      round(ret_63d, 2),
        "ret_126d":     round(ret_126d, 2),
        "ret_252d":     round(ret_252d, 2) if not np.isnan(ret_252d) else np.nan,
        "rsi_14":       rsi,
        "ma50":         round(ma50, 2),
        "ma200":        round(ma200, 2),
        "above_ma50":   above_ma50,
        "above_ma200":  above_ma200,
        "golden_cross": golden_cross,
        "vol_ann_pct":  round(vol_ann, 2),
        "sharpe_1y":    round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "drawdown_52w": round(drawdown, 2),
    }


# -- Scoring & ranking ----------------------------------------------------------

def score_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite score (0-100) weighting:
      - 3-month momentum  (25%)
      - 6-month momentum  (20%)
      - RSI position      (20%) -- penalise overbought >75, reward 40-65 range
      - Sharpe 1Y         (20%)
      - MA structure      (15%) -- above_ma50, above_ma200, golden_cross
    """
    df = df.copy()

    def rank_pct(col: str) -> pd.Series:
        return df[col].rank(pct=True, na_option="bottom")

    # RSI score: reward 40-65 range, penalise <30 or >75
    def rsi_score(rsi: float) -> float:
        if pd.isna(rsi):
            return 0.5
        if rsi < 30:
            return 0.2   # oversold -- might bounce but risky
        if 40 <= rsi <= 65:
            return 1.0   # ideal zone
        if rsi <= 75:
            return 0.7   # slight overbought
        return 0.3       # very overbought

    df["score_mom3m"]  = rank_pct("ret_63d")
    df["score_mom6m"]  = rank_pct("ret_126d")
    df["score_sharpe"] = rank_pct("sharpe_1y")
    df["score_rsi"]    = df["rsi_14"].apply(rsi_score)
    df["score_ma"]     = (
        df["above_ma50"].astype(int) * 0.4
        + df["above_ma200"].astype(int) * 0.4
        + df["golden_cross"].astype(int) * 0.2
    )

    df["composite_score"] = (
        df["score_mom3m"]  * 0.25
        + df["score_mom6m"]  * 0.20
        + df["score_rsi"]    * 0.20
        + df["score_sharpe"] * 0.20
        + df["score_ma"]     * 0.15
    ) * 100

    return df.sort_values("composite_score", ascending=False)


# -- Pretty printing ------------------------------------------------------------

def print_report(ranked: pd.DataFrame, top_n: int = 20):
    buy = ranked.head(top_n)
    print("\n" + "=" * 90)
    print(f"  TOP {top_n} CEDEAR BUY CANDIDATES  --  {date.today()}")
    print("=" * 90)
    print(
        f"{'#':>3}  {'Ticker':<6}  {'Name':<28}  {'Price':>7}  "
        f"{'1M%':>6}  {'3M%':>6}  {'6M%':>6}  {'RSI':>5}  "
        f"{'Shr':>5}  {'DD52W':>6}  {'Score':>6}"
    )
    print("-" * 90)
    for rank, (_, row) in enumerate(buy.iterrows(), 1):
        ma_flag = ""
        if row["golden_cross"]:
            ma_flag = " GC"
        elif row["above_ma200"]:
            ma_flag = " ^"
        print(
            f"{rank:>3}  {row['ticker']:<6}  {str(row['name'])[:28]:<28}  "
            f"{row['price']:>7.2f}  "
            f"{row['ret_21d']:>+6.1f}  {row['ret_63d']:>+6.1f}  {row['ret_126d']:>+6.1f}  "
            f"{row['rsi_14']:>5.1f}  "
            f"{row['sharpe_1y']:>5.2f}  {row['drawdown_52w']:>+6.1f}  "
            f"{row['composite_score']:>6.1f}{ma_flag}"
        )

    print("\nLegend: GC = Golden Cross (MA50>MA200)  ^ = above MA200")
    print(
        "\nColumn guide:\n"
        "  1M%  = 21-day return   3M%  = 63-day return   6M%  = 126-day return\n"
        "  RSI  = 14-day RSI      Shr  = 1-year Sharpe   DD52W = drawdown from 52w high\n"
        "  Score = composite score (0-100, higher = stronger buy signal)"
    )

    print("\n" + "=" * 90)
    print("  AVOID / WATCH (lowest scores -- momentum deterioration)")
    print("=" * 90)
    avoid = ranked.tail(10)
    print(
        f"{'Ticker':<6}  {'Name':<28}  {'1M%':>6}  {'3M%':>6}  {'RSI':>5}  {'Score':>6}"
    )
    print("-" * 60)
    for _, row in avoid.sort_values("composite_score").iterrows():
        print(
            f"{row['ticker']:<6}  {str(row['name'])[:28]:<28}  "
            f"{row['ret_21d']:>+6.1f}  {row['ret_63d']:>+6.1f}  "
            f"{row['rsi_14']:>5.1f}  {row['composite_score']:>6.1f}"
        )


def save_report(ranked: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cedear_ranking_{date.today()}.csv"
    cols = [
        "ticker", "name", "price",
        "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
        "rsi_14", "above_ma50", "above_ma200", "golden_cross",
        "vol_ann_pct", "sharpe_1y", "drawdown_52w", "composite_score",
    ]
    ranked[cols].to_csv(path, index=False)
    print(f"\nFull ranking saved -> {path}")


# -- Main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch & rank all BYMA CEDEARs")
    parser.add_argument("--top",      type=int, default=20, help="Number of top picks to show")
    parser.add_argument("--no-fetch", action="store_true",  help="Skip downloading, use cached CSVs only")
    parser.add_argument("--lookback", type=int, default=400, help="Days of history to fetch (default 400)")
    args = parser.parse_args()

    # 1. Fetch data
    if args.no_fetch:
        print("Loading cached CEDEAR data...")
        data: dict = {}
        for ticker in BYMA_CEDEARS:
            p = CEDEAR_DATA_DIR / f"{ticker}.csv"
            if p.exists():
                df = pd.read_csv(p, parse_dates=["Date"])
                df["Date"] = pd.to_datetime(df["Date"]).dt.date
                data[ticker] = df
    else:
        data = fetch_all(BYMA_CEDEARS, CEDEAR_DATA_DIR, lookback_days=args.lookback)

    # 2. Compute signals
    rows = []
    skipped = []
    for ticker, df in data.items():
        sig = compute_signals(ticker, df)
        if sig:
            rows.append(sig)
        else:
            skipped.append(ticker)

    if skipped:
        print(f"\nSkipped (insufficient history): {', '.join(skipped)}")

    if not rows:
        print("No signals computed -- aborting.")
        sys.exit(1)

    signals_df = pd.DataFrame(rows)

    # 3. Score & rank
    ranked = score_signals(signals_df)

    # 4. Print report
    print_report(ranked, top_n=args.top)

    # 5. Save full ranking CSV
    from config.settings import PORTFOLIO_DIR
    save_report(ranked, PORTFOLIO_DIR)


if __name__ == "__main__":
    main()
