"""
Blend several models' forecasts into a single ensemble forecast.

Weights default to equal, but `weights_from_backtest` derives inverse-error
weights from a benchmark CSV so that historically more accurate models on a given
ticker count more — the standard "trust what backtested well" rule from the
quant-analyst playbook (risk-adjusted, strictly out-of-sample).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.base import ForecastResult


def ensemble(results: list[ForecastResult],
             weights: dict[str, float] | None = None,
             label: str = "Ensemble") -> ForecastResult | None:
    """Weighted-average the point forecasts (and bands) of several models.

    All inputs must be for the same ticker; horizons are aligned to the shortest.
    """
    results = [r for r in results if r is not None and r.horizon > 0]
    if not results:
        return None

    h = min(r.horizon for r in results)
    if weights is None:
        weights = {r.model: 1.0 for r in results}

    # A model present in the forecast but absent from `weights` (e.g. it wasn't
    # in the benchmark run) should still participate — give it the mean of the
    # known weights rather than silently zeroing it out.
    known = [v for v in weights.values() if v > 0]
    fallback = float(np.mean(known)) if known else 1.0
    w = np.array([max(weights.get(r.model, fallback), 0.0) for r in results], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(results))
    w = w / w.sum()

    point = np.zeros(h)
    lower = np.zeros(h)
    upper = np.zeros(h)
    have_band = all(r.lower is not None and r.upper is not None for r in results)
    for wi, r in zip(w, results):
        point += wi * r.point[:h]
        if have_band:
            lower += wi * r.lower[:h]
            upper += wi * r.upper[:h]

    base = results[0]
    return ForecastResult(
        ticker=base.ticker, model=label, name=base.name,
        last_close=base.last_close, dates=base.dates[:h],
        point=point,
        lower=lower if have_band else None,
        upper=upper if have_band else None,
        extra={"members": {r.model: float(wi) for wi, r in zip(w, results)}},
    )


def weights_from_backtest(benchmark: pd.DataFrame, ticker: str | None = None,
                          metric: str = "rmse", floor: float = 1e-6) -> dict[str, float]:
    """Inverse-error weights per model from a benchmark frame.

    Expects columns: ticker, model, <metric>. If `ticker` is given, uses that
    ticker's rows; otherwise averages each model's metric across all tickers.
    Lower error -> higher weight.
    """
    if benchmark is None or benchmark.empty or metric not in benchmark.columns:
        return {}
    df = benchmark
    if ticker is not None and (df["ticker"] == ticker).any():
        df = df[df["ticker"] == ticker]
    agg = df.groupby("model")[metric].mean().dropna()
    if agg.empty:
        return {}
    inv = 1.0 / np.maximum(agg.to_numpy(dtype=float), floor)
    inv = inv / inv.sum()
    return {m: float(v) for m, v in zip(agg.index, inv)}
