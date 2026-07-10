#!/usr/bin/env python
"""
Multi-Model Walk-Forward Benchmark
==================================
Compares the Kronos foundation model against the Nixtla deep time-series models
(N-HiTS, PatchTST, TFT) on a strictly out-of-sample walk-forward, then scores an
equal-weight ensemble of all of them. Reports RMSE / MAE / MAPE / directional
accuracy per ticker per model, and saves a tidy CSV the ensemble forecaster reads
back to weight models by their historical accuracy.

Each window: history[t-context : t] -> forecast `pred_len` days -> compare to the
actual closes. The neural models are *re-trained on each window's history only*
(no look-ahead); Kronos is a foundation model and just runs inference.

Usage:
    /c/Users/feder/anaconda3/python.exe scripts/forecast_benchmark.py
    /c/Users/feder/anaconda3/python.exe scripts/forecast_benchmark.py --tickers AAPL NVDA MELI
    /c/Users/feder/anaconda3/python.exe scripts/forecast_benchmark.py --test-days 252 --stride 42
    /c/Users/feder/anaconda3/python.exe scripts/forecast_benchmark.py --no-kronos   # neural only
"""
import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_DIR, PORTFOLIO_DIR, PORTFOLIO_ASSETS,
    FORECAST_HORIZON, FORECAST_INPUT, NEURAL_MODELS, FORECAST_QUANTILES,
)


# ── Data ────────────────────────────────────────────────────────────────────────

def load_asset(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Metrics ──────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    err = pred - actual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / actual)) * 100)
    if len(pred) > 1:
        da = float(np.mean(np.sign(np.diff(pred)) == np.sign(np.diff(actual))) * 100)
    else:
        da = float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape, "dir_acc": da}


# ── Per-window forecasts from every model ────────────────────────────────────────

def window_forecasts(ticker, hist, horizon, neural, kronos, max_steps) -> dict[str, np.ndarray]:
    """Return {model_name: point_close_array} for one history window."""
    out: dict[str, np.ndarray] = {}
    if neural is not None:
        neural.max_steps = max_steps
        for alias, res in neural.forecast_all(ticker, hist, horizon).items():
            out[alias] = np.asarray(res.point, dtype=float)
    if kronos is not None:
        res = kronos.forecast(ticker, hist, horizon)
        if res is not None:
            out[kronos.name] = np.asarray(res.point, dtype=float)
    return out


def backtest_ticker(ticker, df, neural, kronos, context, horizon, stride,
                    test_days, max_steps) -> list[dict]:
    n = len(df)
    if n < context + horizon:
        print(f"  [{ticker}] only {n} rows, need {context + horizon} — skipping")
        return []

    test_start = max(context, n - test_days)
    # accumulate per-model and ensemble preds/actuals across windows
    acc: dict[str, dict[str, list]] = {}
    windows = 0
    t = test_start
    while t + horizon <= n:
        hist = df.iloc[t - context: t]
        actual = df["close"].to_numpy(dtype=float)[t: t + horizon]
        fc = window_forecasts(ticker, hist, horizon, neural, kronos, max_steps)
        if fc:
            # equal-weight ensemble of whatever models produced a forecast
            stack = np.vstack([v[:horizon] for v in fc.values()])
            fc["Ensemble"] = stack.mean(axis=0)
            for m, pred in fc.items():
                a = acc.setdefault(m, {"p": [], "a": []})
                a["p"].extend(pred[:horizon].tolist())
                a["a"].extend(actual[:len(pred)].tolist())
            windows += 1
            print(f"    w{windows:>2} [{df['date'].iloc[t].date()}] "
                  + "  ".join(f"{m}:{v[-1]:.1f}" for m, v in fc.items()))
        t += stride

    rows = []
    for m, a in acc.items():
        if not a["p"]:
            continue
        met = compute_metrics(np.array(a["p"]), np.array(a["a"]))
        met.update({"ticker": ticker, "name": PORTFOLIO_ASSETS.get(ticker, ""),
                    "model": m, "n_windows": windows, "n_points": len(a["p"]),
                    "total_rows": n})
        rows.append(met)
    return rows


# ── Reporting ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict]):
    if not results:
        print("No results.")
        return
    df = pd.DataFrame(results)
    print("\n" + "=" * 78)
    print("  MULTI-MODEL BENCHMARK — mean across tickers (lower RMSE/MAPE better)")
    print("=" * 78)
    agg = (df.groupby("model")[["rmse", "mae", "mape", "dir_acc"]]
             .mean().sort_values("mape"))
    print(f"{'Model':<12}  {'RMSE':>8}  {'MAE':>8}  {'MAPE%':>7}  {'DirAcc%':>8}")
    print("-" * 78)
    for m, r in agg.iterrows():
        print(f"{m:<12}  {r['rmse']:>8.3f}  {r['mae']:>8.3f}  "
              f"{r['mape']:>6.2f}%  {r['dir_acc']:>7.1f}%")
    print("-" * 78)
    best = agg["mape"].idxmin()
    print(f"Best by MAPE: {best}   |   Best by DirAcc: {agg['dir_acc'].idxmax()}")


def save_results(results: list[dict]):
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["ticker", "name", "model", "rmse", "mae", "mape", "dir_acc",
            "n_windows", "n_points", "total_rows"]
    out = PORTFOLIO_DIR / f"forecast_benchmark_{date.today()}.csv"
    pd.DataFrame(results)[cols].to_csv(out, index=False)
    print(f"\nBenchmark saved -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Kronos vs Nixtla deep models — walk-forward benchmark")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Tickers to benchmark (default: portfolio holdings)")
    ap.add_argument("--context", type=int, default=max(FORECAST_INPUT, 400))
    ap.add_argument("--pred-len", type=int, default=FORECAST_HORIZON)
    ap.add_argument("--stride", type=int, default=42, help="Days between windows (default 42)")
    ap.add_argument("--test-days", type=int, default=252, help="Test window size (default 252 ~1y)")
    ap.add_argument("--max-steps", type=int, default=100, help="Neural training steps per window")
    ap.add_argument("--samples", type=int, default=10, help="Kronos Monte-Carlo paths per window")
    ap.add_argument("--kronos-model", default="Kronos-base",
                    choices=["Kronos-mini", "Kronos-small", "Kronos-base"])
    ap.add_argument("--no-kronos", action="store_true", help="Benchmark neural models only")
    ap.add_argument("--no-neural", action="store_true", help="Benchmark Kronos only")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tickers = args.tickers or list(PORTFOLIO_ASSETS.keys())
    if args.limit:
        tickers = tickers[: args.limit]

    # Build forecasters once
    neural = kronos = None
    if not args.no_neural:
        from src.forecasting.neural import NeuralForecaster
        neural = NeuralForecaster(models=NEURAL_MODELS, input_size=FORECAST_INPUT,
                                  max_steps=args.max_steps, quantiles=FORECAST_QUANTILES)
        print(f"Neural models: {NEURAL_MODELS} on {neural.accelerator}")
    if not args.no_kronos:
        from src.forecasting.kronos_model import KronosForecaster
        print(f"Loading {args.kronos_model} ...")
        kronos = KronosForecaster(args.kronos_model, samples=args.samples,
                                  quantiles=FORECAST_QUANTILES)
        print(f"Kronos on {kronos.device}")

    print(f"\nWalk-forward: context={args.context} pred={args.pred_len} "
          f"stride={args.stride} test={args.test_days} | {len(tickers)} tickers\n")

    results: list[dict] = []
    t0 = time.time()
    for tk in tickers:
        df = load_asset(tk)
        if df.empty:
            print(f"[{tk}] no data file — skipping")
            continue
        print(f"[{tk}] {PORTFOLIO_ASSETS.get(tk, '')}")
        ts = time.time()
        rows = backtest_ticker(tk, df, neural, kronos, args.context, args.pred_len,
                               args.stride, args.test_days, args.max_steps)
        results.extend(rows)
        if rows:
            best = min(rows, key=lambda r: r["mape"])
            print(f"  -> best {best['model']} MAPE={best['mape']:.2f}%  ({time.time()-ts:.0f}s)")

    if not results:
        print("No results — aborting.")
        sys.exit(1)

    print_report(results)
    save_results(results)
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
