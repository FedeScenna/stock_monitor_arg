#!/usr/bin/env python
"""
One-shot pipeline - refresh data, then run the Kronos forecast and the weekly screen.

Runs everything end to end in the right order, refreshing each data source exactly
once and then running the two analyses offline against that fresh data:

  1. OHLCV refresh     - full CEDEAR universe + portfolio assets   (DataFetcher)
  2. Fundamentals      - tangible book / FCF / debt for the value screen (FundamentalsFetcher)
  3. Kronos forecast   - 21-day Kronos forecast      -> data/portfolio/kronos_forecast_DATE.csv
  4. Weekly screen     - value + technical screen     -> data/portfolio/weekly_screen_DATE.csv
  5. Technical signals - Murphy TA weight-of-evidence -> data/portfolio/technical_signals_DATE.csv
  6. Ensemble forecast - Kronos + Nixtla deep models  -> data/portfolio/ensemble_forecast_DATE.csv
     (opt-in via --with-ensemble; trains N-HiTS/PatchTST/TFT, so it is the slow step)

Steps 3-6 run with --no-fetch since steps 1-2 already refreshed everything.

Usage:
    /c/Users/feder/anaconda3/python.exe scripts/refresh_all.py
    /c/Users/feder/anaconda3/python.exe scripts/refresh_all.py --no-fetch            # caches only
    /c/Users/feder/anaconda3/python.exe scripts/refresh_all.py --skip-fundamentals   # faster, no value screen
    /c/Users/feder/anaconda3/python.exe scripts/refresh_all.py --skip-kronos
    /c/Users/feder/anaconda3/python.exe scripts/refresh_all.py --with-ensemble       # add the multi-model ensemble
    /c/Users/feder/anaconda3/python.exe scripts/refresh_all.py --kronos-model Kronos-base --kronos-samples 50
"""
import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    CEDEAR_TICKERS, CEDEAR_SKIP, PORTFOLIO_ASSETS, PORTFOLIO_STOCKS, PORTFOLIO_DIR,
)
from src.data.fetcher import DataFetcher
from src.data.fundamentals import FundamentalsFetcher

PYTHON = sys.executable
SCRIPTS = ROOT / "scripts"

# Windows consoles default to cp1252; keep our own output robust to non-ASCII.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _banner(label: str):
    print("\n" + "=" * 74)
    print(f"  {label}")
    print("=" * 74)


def _run_script(label: str, script: str, *script_args) -> tuple[str, bool, float]:
    """Run another script in this repo as a subprocess, streaming its output."""
    _banner(label)
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, "-u", str(SCRIPTS / script), *script_args],
        cwd=str(ROOT),
    )
    dt = time.time() - t0
    ok = proc.returncode == 0
    print(f"\n  [{'ok' if ok else 'FAILED'}] {label} - {dt:.0f}s"
          + ("" if ok else f" (exit {proc.returncode})"))
    return label, ok, dt


def main():
    p = argparse.ArgumentParser(description="Refresh data + run Kronos forecast + weekly screen")
    p.add_argument("--no-fetch", action="store_true",
                   help="Skip OHLCV + fundamentals refresh; run analyses on cached data")
    p.add_argument("--skip-fundamentals", action="store_true",
                   help="Skip fundamentals refresh and run the technical-only screen")
    p.add_argument("--skip-kronos", action="store_true", help="Skip the Kronos forecast")
    p.add_argument("--skip-screen", action="store_true", help="Skip the weekly screen")
    p.add_argument("--skip-signals", action="store_true", help="Skip the Murphy technical signals")
    p.add_argument("--kronos-model", default="Kronos-base",
                   choices=["Kronos-mini", "Kronos-small", "Kronos-base"])
    p.add_argument("--kronos-samples", type=int, default=20)
    p.add_argument("--pred-days", type=int, default=21)
    p.add_argument("--kronos-portfolio", action="store_true",
                   help="Forecast only the portfolio (default forecasts every CEDEAR)")
    p.add_argument("--with-ensemble", action="store_true",
                   help="Also run the multi-model ensemble forecast (Kronos + Nixtla deep models)")
    p.add_argument("--ensemble-universe", action="store_true",
                   help="Run the ensemble over the full CEDEAR universe (default: portfolio only)")
    args = p.parse_args()

    started = time.time()
    results: list[tuple[str, bool, float]] = []

    # Tickers whose OHLCV the downstream steps need (screen universe + every portfolio asset).
    ohlcv_tickers = sorted(
        (set(CEDEAR_TICKERS) | set(PORTFOLIO_ASSETS) | set(PORTFOLIO_STOCKS)) - set(CEDEAR_SKIP)
    )
    cedear_universe = [t for t in CEDEAR_TICKERS if t not in CEDEAR_SKIP]

    # ── 1. OHLCV refresh ──────────────────────────────────────────────────────
    if not args.no_fetch:
        _banner(f"Step 1/6 - Refresh OHLCV ({len(ohlcv_tickers)} tickers)")
        t0 = time.time()
        DataFetcher().update_all(ohlcv_tickers)
        results.append(("OHLCV refresh", True, time.time() - t0))
    else:
        print("Step 1/6 - OHLCV refresh skipped (--no-fetch)")

    # ── 2. Fundamentals refresh ───────────────────────────────────────────────
    if not args.no_fetch and not args.skip_fundamentals:
        _banner(f"Step 2/6 - Refresh fundamentals ({len(cedear_universe)} tickers)")
        t0 = time.time()
        FundamentalsFetcher().update_all(cedear_universe, force=True)
        results.append(("Fundamentals refresh", True, time.time() - t0))
    else:
        print("Step 2/6 - Fundamentals refresh skipped"
              f" ({'--no-fetch' if args.no_fetch else '--skip-fundamentals'})")

    # ── 3. Kronos forecast (offline; data already fresh) ──────────────────────
    if not args.skip_kronos:
        kronos_args = [
            "--no-fetch", "--model", args.kronos_model,
            "--samples", str(args.kronos_samples), "--pred-days", str(args.pred_days),
        ]
        if args.kronos_portfolio:
            kronos_args.append("--portfolio")
        results.append(_run_script("Step 3/6 - Kronos forecast", "kronos_forecast.py", *kronos_args))
    else:
        print("\nStep 3/6 - Kronos forecast skipped (--skip-kronos)")

    # ── 4. Weekly screen (offline; uses fresh OHLCV + fundamentals caches) ─────
    if not args.skip_screen:
        screen_args = ["--no-fetch"]
        if args.skip_fundamentals:
            screen_args.append("--skip-fundamentals")
        results.append(_run_script("Step 4/6 - Weekly screen", "weekly_screen.py", *screen_args))
    else:
        print("\nStep 4/6 - Weekly screen skipped (--skip-screen)")

    # ── 5. Technical signals (offline; Murphy TA weight-of-evidence) ──────────
    if not args.skip_signals:
        results.append(_run_script("Step 5/6 - Technical signals", "technical_signals.py", "--no-fetch"))
    else:
        print("\nStep 5/6 - Technical signals skipped (--skip-signals)")

    # ── 6. Ensemble forecast (opt-in; Kronos + Nixtla deep models) ────────────
    if args.with_ensemble:
        ens_args = ["--pred-days", str(args.pred_days), "--kronos-model", args.kronos_model]
        if args.ensemble_universe:
            ens_args.append("--universe")
        results.append(_run_script("Step 6/6 - Ensemble forecast", "ensemble_forecast.py", *ens_args))
    else:
        print("\nStep 6/6 - Ensemble forecast skipped (pass --with-ensemble to enable)")

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner(f"PIPELINE COMPLETE - {date.today()}  ({time.time() - started:.0f}s total)")
    for label, ok, dt in results:
        print(f"  [{'ok' if ok else 'FAIL'}]  {label:<28} {dt:>6.0f}s")
    today = date.today().isoformat()
    print("\n  Outputs:")
    outputs = [f"kronos_forecast_{today}.csv", f"weekly_screen_{today}.csv"]
    if not args.skip_signals:
        outputs.append(f"technical_signals_{today}.csv")
    if args.with_ensemble:
        outputs.append(f"ensemble_forecast_{today}.csv")
    for pat in outputs:
        f = PORTFOLIO_DIR / pat
        print(f"    {'+' if f.exists() else '-'} data/portfolio/{pat}"
              + ("" if f.exists() else "  (not written)"))

    if any(not ok for _, ok, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
