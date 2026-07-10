"""
Nixtla `neuralforecast` deep time-series models (N-HiTS, PatchTST, TFT).

Each model is trained per-ticker on the close-price series and produces a
probabilistic forecast (median + quantile band) for the next `horizon` steps.
The series is mapped onto a synthetic gap-free business-day axis so US market
holidays never create holes that confuse the fixed-frequency models.
"""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd

from src.forecasting.base import Forecaster, ForecastResult, next_trading_days

# Silence the very chatty pytorch-lightning / neuralforecast logs.
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")
for _name in ("lightning.pytorch", "pytorch_lightning", "lightning_fabric"):
    logging.getLogger(_name).setLevel(logging.ERROR)


def _build_one(name: str, horizon: int, input_size: int, max_steps: int,
               quantiles, accelerator: str, batch_size: int, windows_batch_size: int):
    """Instantiate a single Nixtla model with a shared quantile loss.

    Models are built (and trained) one at a time so a 6 GB laptop GPU never has
    to hold all of them at once — TFT in particular is memory-hungry.
    """
    from neuralforecast.models import NHITS, PatchTST, TFT
    from neuralforecast.losses.pytorch import MQLoss

    loss = MQLoss(quantiles=list(quantiles))
    common = dict(
        h=horizon,
        input_size=input_size,
        max_steps=max_steps,
        loss=loss,
        scaler_type="robust",
        val_check_steps=max_steps,            # no intermediate validation
        batch_size=batch_size,
        windows_batch_size=windows_batch_size,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    registry = {
        "NHITS":    lambda: NHITS(**common),
        "PATCHTST": lambda: PatchTST(**common),
        "TFT":      lambda: TFT(**common),
    }
    key = name.upper()
    return registry[key]() if key in registry else None


class NeuralForecaster(Forecaster):
    """Trains the Nixtla deep models fresh for each ticker (no cross-series leak)."""

    name = "neural"

    def __init__(self, models=None, input_size: int = 252, max_steps: int = 200,
                 quantiles=(0.1, 0.5, 0.9), accelerator: str | None = None,
                 batch_size: int = 32, windows_batch_size: int = 128):
        self.model_names = list(models) if models else ["NHITS", "PatchTST", "TFT"]
        self.input_size = input_size
        self.max_steps = max_steps
        self.quantiles = list(quantiles)
        self.batch_size = batch_size
        self.windows_batch_size = windows_batch_size
        if accelerator is None:
            try:
                import torch
                accelerator = "gpu" if torch.cuda.is_available() else "cpu"
            except Exception:
                accelerator = "cpu"
        self.accelerator = accelerator

    @staticmethod
    def _free_gpu():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Returns {model_alias: ForecastResult}
    def forecast_all(self, ticker: str, df: pd.DataFrame, horizon: int,
                     name: str = "") -> dict[str, ForecastResult]:
        from neuralforecast import NeuralForecast

        df = self._prepare(df)
        if "close" not in df.columns:
            return {}

        n = len(df)
        input_size = int(min(self.input_size, n - horizon - 1))
        if input_size < horizon:
            return {}   # not enough history to train

        # Synthetic gap-free business-day index → no holiday holes.
        ds = pd.bdate_range(end=pd.Timestamp("2024-01-01"), periods=n)
        train = pd.DataFrame({
            "unique_id": ticker,
            "ds": ds,
            "y": df["close"].to_numpy(dtype=float),
        })

        last_close = float(df["close"].iloc[-1])
        future_dates = [str(d.date()) for d in next_trading_days(horizon, self._last_date(df))]

        # Train each model in its own NeuralForecast and free the GPU between
        # runs, so peak memory is one model's worth (TFT is the heavy one).
        results: dict[str, ForecastResult] = {}
        for mname in self.model_names:
            model = _build_one(mname, horizon, input_size, self.max_steps,
                               self.quantiles, self.accelerator,
                               self.batch_size, self.windows_batch_size)
            if model is None:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    nf = NeuralForecast(models=[model], freq="B")
                    nf.fit(train, val_size=0)
                    fcst = nf.predict()
                except Exception as exc:
                    print(f"  [{ticker}] {mname} forecast error: {exc}")
                    self._free_gpu()
                    continue
            results.update(
                self._parse(fcst, ticker, name, last_close, future_dates, horizon))
            del nf, model
            self._free_gpu()

        return results

    def forecast(self, ticker: str, df: pd.DataFrame, horizon: int,
                 name: str = "") -> ForecastResult | None:
        """Single-result convenience: returns the first model's forecast."""
        out = self.forecast_all(ticker, df, horizon, name)
        return next(iter(out.values()), None)

    def _parse(self, fcst: pd.DataFrame, ticker: str, name: str, last_close: float,
               dates: list[str], horizon: int) -> dict[str, ForecastResult]:
        """Extract median + band columns per model alias from the predict() frame."""
        cols = list(fcst.columns)
        qlo = min(self.quantiles)
        qhi = max(self.quantiles)
        results: dict[str, ForecastResult] = {}

        for alias in {c.split("-")[0] for c in cols if c not in ("unique_id", "ds")}:
            mcols = [c for c in cols if c == alias or c.startswith(alias + "-")]
            point = self._pick_quantile(fcst, mcols, alias, 0.5)
            lower = self._pick_quantile(fcst, mcols, alias, qlo)
            upper = self._pick_quantile(fcst, mcols, alias, qhi)
            if point is None:
                continue
            results[alias] = ForecastResult(
                ticker=ticker, model=alias, name=name, last_close=last_close,
                dates=dates[:len(point)], point=point[:horizon],
                lower=None if lower is None else lower[:horizon],
                upper=None if upper is None else upper[:horizon],
            )
        return results

    @staticmethod
    def _pick_quantile(fcst, mcols, alias, q):
        """MQLoss columns look like 'NHITS-median' / 'NHITS-lo-80' / 'NHITS-hi-80'
        or 'NHITS-q-50'. Be liberal about the naming convention."""
        tag_pct = int(round(q * 100))
        # Direct quantile tag, e.g. NHITS-q-50 / NHITS-50
        for c in mcols:
            tail = c[len(alias):].lstrip("-")
            if tail in (f"q-{tag_pct}", str(tag_pct), f"ql{tag_pct}"):
                return fcst[c].to_numpy(dtype=float)
        # Median / lo / hi naming
        if abs(q - 0.5) < 1e-9:
            for c in mcols:
                if c == alias or c.endswith("-median"):
                    return fcst[c].to_numpy(dtype=float)
        else:
            kind = "lo" if q < 0.5 else "hi"
            band = [c for c in mcols if f"-{kind}-" in c]
            if band:
                return fcst[band[0]].to_numpy(dtype=float)
        return None
