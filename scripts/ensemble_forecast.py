#!/usr/bin/env python
"""
Ensemble 21-Day Forecast (Kronos + Nixtla N-HiTS / PatchTST / TFT)
==================================================================
Runs every model on each ticker and blends them into one forecast. If a recent
`forecast_benchmark_*.csv` exists, models are weighted by inverse error (per
ticker when available, else globally) so historically more accurate models count
more; otherwise an equal-weight blend is used.

Outputs `data/portfolio/ensemble_forecast_{date}.csv` with one row per
(ticker, model, day) — including each member model and the `Ensemble` row — so
the app can show the blend and its disagreement band.

Usage:
    /c/Users/feder/anaconda3/python.exe scripts/ensemble_forecast.py            # portfolio
    /c/Users/feder/anaconda3/python.exe scripts/ensemble_forecast.py --universe # all CEDEARs
    /c/Users/feder/anaconda3/python.exe scripts/ensemble_forecast.py --tickers AAPL NVDA --no-kronos
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
    DATA_DIR, PORTFOLIO_DIR, PORTFOLIO_ASSETS, CEDEAR_TICKERS, CEDEAR_SKIP,
    FORECAST_HORIZON, FORECAST_INPUT, FORECAST_MAX_STEPS, NEURAL_MODELS,
    FORECAST_QUANTILES, ENSEMBLE_MIN_ROWS,
)
from src.forecasting import ensemble, weights_from_backtest


def load_asset(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_values("date").reset_index(drop=True)


def latest_benchmark() -> pd.DataFrame | None:
    files = sorted(PORTFOLIO_DIR.glob("forecast_benchmark_*.csv"))
    if not files:
        return None
    print(f"Weighting models by backtest: {files[-1].name}")
    return pd.read_csv(files[-1])


def main():
    ap = argparse.ArgumentParser(description="Ensemble multi-model price forecast")
    ap.add_argument("--universe", action="store_true",
                    help="Forecast the full CEDEAR universe instead of the portfolio")
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--pred-days", type=int, default=FORECAST_HORIZON)
    ap.add_argument("--max-steps", type=int, default=FORECAST_MAX_STEPS)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--kronos-model", default="Kronos-base",
                    choices=["Kronos-mini", "Kronos-small", "Kronos-base"])
    ap.add_argument("--no-kronos", action="store_true")
    ap.add_argument("--no-neural", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.tickers:
        names = {t: PORTFOLIO_ASSETS.get(t, CEDEAR_TICKERS.get(t, "")) for t in args.tickers}
    elif args.universe:
        names = {t: n for t, n in CEDEAR_TICKERS.items() if t not in CEDEAR_SKIP}
    else:
        names = dict(PORTFOLIO_ASSETS)
    if args.limit:
        names = dict(list(names.items())[: args.limit])

    bench = latest_benchmark()

    neural = kronos = None
    if not args.no_neural:
        from src.forecasting.neural import NeuralForecaster
        neural = NeuralForecaster(models=NEURAL_MODELS, input_size=FORECAST_INPUT,
                                  max_steps=args.max_steps, quantiles=FORECAST_QUANTILES)
        print(f"Neural models {NEURAL_MODELS} on {neural.accelerator}")
    if not args.no_kronos:
        from src.forecasting.kronos_model import KronosForecaster
        print(f"Loading {args.kronos_model} ...")
        kronos = KronosForecaster(args.kronos_model, samples=args.samples,
                                  quantiles=FORECAST_QUANTILES)
        print(f"Kronos on {kronos.device}")

    print(f"\nForecasting {args.pred_days} days for {len(names)} assets\n")
    rows: list[dict] = []
    summary: list[dict] = []
    t0 = time.time()

    for tk, nm in names.items():
        df = load_asset(tk)
        if df.empty or len(df) < ENSEMBLE_MIN_ROWS:
            print(f"[{tk}] insufficient history ({len(df)} rows) — skipping")
            continue
        members = []
        if neural is not None:
            members.extend(neural.forecast_all(tk, df, args.pred_days, nm).values())
        if kronos is not None:
            r = kronos.forecast(tk, df, args.pred_days, nm)
            if r is not None:
                members.append(r)
        if not members:
            print(f"[{tk}] no model produced a forecast — skipping")
            continue

        w = weights_from_backtest(bench, ticker=tk) if bench is not None else None
        ens = ensemble(members, weights=w)
        for r in members + [ens]:
            rows.extend(r.to_rows())
        summary.append({"ticker": tk, "name": nm, "last_close": ens.last_close,
                        "upside_pct": ens.upside_pct(),
                        "weights": ens.extra.get("members", {})})
        print(f"[{tk}] ensemble 21d upside {ens.upside_pct():+.1f}%  "
              f"({', '.join(f'{m}:{v:.2f}' for m, v in ens.extra.get('members', {}).items())})")

    if not rows:
        print("No forecasts produced — aborting.")
        sys.exit(1)

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    out = PORTFOLIO_DIR / f"ensemble_forecast_{date.today()}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    print("\n" + "=" * 64)
    print(f"  ENSEMBLE FORECAST — {date.today()} — next {args.pred_days} trading days")
    print("=" * 64)
    print(f"{'Ticker':<7}  {'Name':<26}  {'Last':>9}  {'21d up%':>8}")
    print("-" * 64)
    for s in sorted(summary, key=lambda x: x["upside_pct"], reverse=True):
        print(f"{s['ticker']:<7}  {s['name'][:26]:<26}  {s['last_close']:>9.2f}  "
              f"{s['upside_pct']:>+7.1f}%")
    print(f"\nSaved -> {out}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
