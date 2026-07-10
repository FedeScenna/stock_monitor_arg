"""
Kronos foundation-model wrapper conforming to the shared `Forecaster` interface.

The expensive model load happens once in `__init__`; `forecast` then runs cheap
per-ticker inference, drawing Monte-Carlo sample paths and reducing them to a
point forecast (mean) plus a quantile band.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.forecasting.base import Forecaster, ForecastResult, next_trading_days

TOKENIZER_MAP = {
    "Kronos-mini":  "NeoQuasar/Kronos-Tokenizer-2k",
    "Kronos-small": "NeoQuasar/Kronos-Tokenizer-base",
    "Kronos-base":  "NeoQuasar/Kronos-Tokenizer-base",
}
MAX_CONTEXT_MAP = {"Kronos-mini": 2048, "Kronos-small": 512, "Kronos-base": 512}


class KronosForecaster(Forecaster):
    name = "Kronos"

    def __init__(self, model_name: str = "Kronos-base", device: str | None = None,
                 samples: int = 20, quantiles=(0.1, 0.5, 0.9)):
        import torch
        from model import Kronos, KronosTokenizer, KronosPredictor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.samples = samples
        self.quantiles = list(quantiles)
        self.name = model_name

        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_MAP[model_name])
        kmodel = Kronos.from_pretrained(f"NeoQuasar/{model_name}").to(device)
        self.max_context = MAX_CONTEXT_MAP[model_name]
        self.predictor = KronosPredictor(kmodel, tokenizer, device=device,
                                         max_context=self.max_context)

    def forecast(self, ticker: str, df: pd.DataFrame, horizon: int,
                 name: str = "") -> ForecastResult | None:
        df = self._prepare(df)
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            return None

        hist = df.tail(self.max_context).reset_index(drop=True)
        x_df = hist[["open", "high", "low", "close"]]
        x_ts = pd.Series(pd.to_datetime(hist["date"]).values)
        last_date = self._last_date(hist)
        y_ts = pd.Series(next_trading_days(horizon, from_date=last_date))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                samples = self.predictor.predict(
                    df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=horizon,
                    T=1.0, top_p=0.9, sample_count=self.samples,
                    verbose=False, return_samples=True,
                )
            except Exception as exc:
                print(f"  [{ticker}] Kronos error: {exc}")
                return None

        close_samples = samples[:, :, 3]              # (n_samples, horizon)
        point = close_samples.mean(axis=0)
        lo = np.quantile(close_samples, min(self.quantiles), axis=0)
        hi = np.quantile(close_samples, max(self.quantiles), axis=0)
        last_close = float(hist["close"].iloc[-1])

        return ForecastResult(
            ticker=ticker, model=self.name, name=name, last_close=last_close,
            dates=[str(d.date()) for d in y_ts],
            point=point, lower=lo, upper=hi,
        )
