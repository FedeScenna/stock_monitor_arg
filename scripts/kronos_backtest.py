#!/usr/bin/env python
"""
Kronos Walk-Forward Backtesting
================================
Evaluates Kronos forecast accuracy on each portfolio asset using a
walk-forward strategy over the most recent 2 years of data, then
reports RMSE, MAE, MAPE and directional accuracy per ticker.

Strategy
--------
  - Context (lookback) : 400 trading days  (within Kronos-small 512 cap)
  - Forecast horizon   : 21 trading days per step (~1 month)
  - Stride             : 21 days (non-overlapping windows)
  - Test window        : last 504 trading days (~2 years)  ← configurable
  - Samples            : 10 Monte Carlo paths per window

With context=400 + test=504 we need at least 904 rows — all assets except
NU (1086 rows) and VIST (1685 rows) have 4000-6600 rows, so they'll use
the most recent portion.

Usage:
    /c/Users/feder/anaconda3/python.exe scripts/kronos_backtest.py
    /c/Users/feder/anaconda3/python.exe scripts/kronos_backtest.py --test-days 252 --stride 10
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR, PORTFOLIO_DIR, PORTFOLIO_ASSETS
from model import Kronos, KronosTokenizer, KronosPredictor


# ── Data loading ───────────────────────────────────────────────────────────────

def load_asset(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Single forecast window ─────────────────────────────────────────────────────

def forecast_window(
    predictor: KronosPredictor,
    hist: pd.DataFrame,
    future: pd.DataFrame,
    pred_len: int,
    samples: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    available = min(pred_len, len(future))
    if available == 0:
        return None

    x_df = hist[["open", "high", "low", "close"]].reset_index(drop=True)
    x_ts  = pd.Series(hist["date"].values)
    y_ts  = pd.Series(future["date"].values[:available])

    try:
        pred = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=available,
            T=1.0,
            top_p=0.9,
            sample_count=samples,
            verbose=False,
        )
    except Exception as exc:
        print(f"      Kronos error: {exc}")
        return None

    return pred["close"].values[:available], future["close"].values[:available]


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    err  = pred - actual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / actual)) * 100)
    # Directional accuracy (day-over-day sign match)
    if len(pred) > 1:
        da = float(np.mean(np.sign(np.diff(pred)) == np.sign(np.diff(actual))) * 100)
    else:
        da = float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape, "dir_acc": da}


# ── Walk-forward per asset ─────────────────────────────────────────────────────

def backtest_asset(
    ticker: str,
    df: pd.DataFrame,
    predictor: KronosPredictor,
    context: int,
    pred_len: int,
    stride: int,
    test_days: int,
    samples: int,
) -> dict | None:

    n = len(df)
    min_rows = context + pred_len
    if n < min_rows:
        print(f"  [{ticker}] only {n} rows, need {min_rows} — skipping")
        return None

    # Restrict test set to last `test_days` rows (use preceding rows as context pool)
    test_start_idx = max(context, n - test_days)
    all_pred, all_actual = [], []
    windows = 0
    t = test_start_idx  # first evaluation point (end of first context window)

    while t + pred_len <= n:
        hist   = df.iloc[t - context : t]
        future = df.iloc[t : t + pred_len]

        res = forecast_window(predictor, hist, future, pred_len, samples)
        if res is not None:
            pred_c, act_c = res
            all_pred.extend(pred_c.tolist())
            all_actual.extend(act_c.tolist())
            windows += 1
            upside = (pred_c[-1] / act_c[-1] - 1) * 100
            print(f"    w{windows:>2}  [{df['date'].iloc[t].date()} + {pred_len}d]  "
                  f"actual={act_c[-1]:.2f}  pred={pred_c[-1]:.2f}  "
                  f"err={pred_c[-1]-act_c[-1]:+.2f} ({upside:+.1f}%)")
        t += stride

    if not all_pred:
        return None

    preds   = np.array(all_pred)
    actuals = np.array(all_actual)
    m = compute_metrics(preds, actuals)
    m.update({
        "ticker":      ticker,
        "name":        PORTFOLIO_ASSETS.get(ticker, ""),
        "n_windows":   windows,
        "n_points":    len(preds),
        "last_price":  float(df["close"].iloc[-1]),
        "total_rows":  n,
    })
    return m


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_report(results: list, args):
    print("\n" + "=" * 86)
    print(f"  KRONOS BACKTEST  |  context={args.context}d  pred={args.pred_len}d  "
          f"stride={args.stride}d  test_window={args.test_days}d  samples={args.samples}")
    print("=" * 86)
    header = (f"{'#':>2}  {'Ticker':<6}  {'Name':<28}  {'RMSE':>8}  {'MAE':>8}  "
              f"{'MAPE%':>7}  {'DirAcc%':>8}  {'Rows':>6}  {'Win':>4}")
    print(header)
    print("-" * 86)

    ranked = sorted(results, key=lambda x: x["mape"])
    for i, r in enumerate(ranked, 1):
        print(
            f"{i:>2}  {r['ticker']:<6}  {r['name'][:28]:<28}  "
            f"{r['rmse']:>8.3f}  {r['mae']:>8.3f}  {r['mape']:>6.2f}%  "
            f"{r['dir_acc']:>7.1f}%  {r['total_rows']:>6}  {r['n_windows']:>4}"
        )

    vals = lambda k: [r[k] for r in results]
    print("-" * 86)
    print(
        f"{'':>2}  {'MEAN':<6}  {'':28}  "
        f"{np.mean(vals('rmse')):>8.3f}  {np.mean(vals('mae')):>8.3f}  "
        f"{np.mean(vals('mape')):>6.2f}%  {np.mean(vals('dir_acc')):>7.1f}%"
    )
    print("\nColumn guide:")
    print("  RMSE    = Root Mean Squared Error on close price (USD) — penalises big misses")
    print("  MAE     = Mean Absolute Error on close price (USD)")
    print("  MAPE%   = Mean Absolute % Error — scale-independent (lower = better)")
    print("  DirAcc% = % of days where predicted direction (up/down) matched actual")
    print("  Rows    = total history rows available for this asset")
    print("  Win     = number of non-overlapping 21-day forecast windows evaluated")


def save_report(results: list, args):
    from datetime import date as date_cls
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(results)
    cols = ["ticker", "name", "rmse", "mae", "mape", "dir_acc",
            "n_windows", "n_points", "last_price", "total_rows"]
    path = PORTFOLIO_DIR / f"kronos_backtest_{date_cls.today()}.csv"
    df_out[cols].to_csv(path, index=False)
    print(f"\nResults saved -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kronos walk-forward backtest")
    parser.add_argument("--context",   type=int, default=400,
                        help="Context window in trading days (default 400, max 512 for Kronos-small)")
    parser.add_argument("--pred-len",  type=int, default=21,
                        help="Forecast horizon per step in trading days (default 21)")
    parser.add_argument("--stride",    type=int, default=21,
                        help="Days between steps (default 21 = non-overlapping)")
    parser.add_argument("--test-days", type=int, default=504,
                        help="Size of the test window in trading days (default 504 ~ 2 years)")
    parser.add_argument("--samples",   type=int, default=10,
                        help="Monte Carlo sample paths per window (default 10)")
    parser.add_argument("--model",     default="Kronos-small",
                        choices=["Kronos-mini", "Kronos-small", "Kronos-base"])
    args = parser.parse_args()

    TOKENIZER_MAP = {
        "Kronos-mini":  "NeoQuasar/Kronos-Tokenizer-2k",
        "Kronos-small": "NeoQuasar/Kronos-Tokenizer-base",
        "Kronos-base":  "NeoQuasar/Kronos-Tokenizer-base",
    }
    MAX_CTX_MAP = {"Kronos-mini": 2048, "Kronos-small": 512, "Kronos-base": 512}

    if args.context > MAX_CTX_MAP[args.model]:
        print(f"Warning: context {args.context} > model max {MAX_CTX_MAP[args.model]}, clamping.")
        args.context = MAX_CTX_MAP[args.model]

    # 1. Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {args.model} on {device} ...")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_MAP[args.model])
    model     = Kronos.from_pretrained(f"NeoQuasar/{args.model}")
    model     = model.to(device)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=args.context)

    # 2. Backtest each asset
    results = []
    t0_total = time.time()

    for ticker in PORTFOLIO_ASSETS:
        print(f"\n[{ticker}] {PORTFOLIO_ASSETS[ticker]}")
        df = load_asset(ticker)
        if df.empty:
            print(f"  no data file found")
            continue

        t0 = time.time()
        res = backtest_asset(
            ticker, df, predictor,
            context=args.context,
            pred_len=args.pred_len,
            stride=args.stride,
            test_days=args.test_days,
            samples=args.samples,
        )
        elapsed = time.time() - t0

        if res:
            res["elapsed_s"] = round(elapsed, 1)
            results.append(res)
            print(f"  -> RMSE={res['rmse']:.3f}  MAE={res['mae']:.3f}  "
                  f"MAPE={res['mape']:.2f}%  DirAcc={res['dir_acc']:.1f}%  "
                  f"({windows_str(res)}  {elapsed:.0f}s)")

    if not results:
        print("No results. Aborting.")
        sys.exit(1)

    print_report(results, args)
    save_report(results, args)
    print(f"\nTotal time: {time.time() - t0_total:.0f}s")


def windows_str(r):
    return f"{r['n_windows']} windows x {r['n_points']//max(r['n_windows'],1)} pts"


if __name__ == "__main__":
    main()
