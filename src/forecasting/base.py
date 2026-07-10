"""
Common forecasting interface shared by every model wrapper.

Each forecaster consumes an OHLCV DataFrame (lowercase columns: date, open, high,
low, close, volume) and returns a `ForecastResult` describing the next `horizon`
trading days of the *close* price, with an optional probabilistic band.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd


# ── Future trading-day timestamps ─────────────────────────────────────────────

def next_trading_days(n: int, from_date: date | None = None) -> pd.DatetimeIndex:
    """Generate n future trading days (Mon-Fri). Holidays are ignored — the
    forecast horizon is a *count* of steps, not a precise calendar mapping."""
    if from_date is None:
        from_date = date.today()
    days: list[pd.Timestamp] = []
    cur = from_date + timedelta(days=1)
    while len(days) < n:
        if cur.weekday() < 5:  # Mon-Fri
            days.append(pd.Timestamp(cur))
        cur += timedelta(days=1)
    return pd.DatetimeIndex(days)


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    """A single model's forecast for one ticker."""
    ticker: str
    model: str
    last_close: float
    dates: list[str]                       # ISO date strings, length = horizon
    point: np.ndarray                      # (horizon,) point forecast of close
    lower: np.ndarray | None = None        # (horizon,) low quantile band
    upper: np.ndarray | None = None        # (horizon,) high quantile band
    name: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return len(self.point)

    def upside_pct(self) -> float:
        """Predicted % change from last close to the final horizon step."""
        if self.last_close == 0 or len(self.point) == 0:
            return float("nan")
        return float(self.point[-1] / self.last_close - 1.0) * 100.0

    def to_rows(self) -> list[dict]:
        """Flatten to one row per forecast day (for CSV output)."""
        rows = []
        for i, (d, c) in enumerate(zip(self.dates, self.point)):
            rows.append({
                "ticker": self.ticker,
                "name": self.name,
                "model": self.model,
                "forecast_date": d,
                "pred_day": i + 1,
                "pred_close": round(float(c), 4),
                "pred_low": round(float(self.lower[i]), 4) if self.lower is not None else np.nan,
                "pred_high": round(float(self.upper[i]), 4) if self.upper is not None else np.nan,
                "last_close": round(float(self.last_close), 4),
                "upside_pct": round((float(c) / self.last_close - 1) * 100, 2),
            })
        return rows


# ── Abstract forecaster ────────────────────────────────────────────────────────

class Forecaster(ABC):
    """A model that maps an OHLCV history to a `ForecastResult`.

    Subclasses may do expensive one-time setup in __init__ (e.g. load a
    foundation model onto the GPU) and cheap per-ticker work in `forecast`.
    """
    name: str = "base"

    @abstractmethod
    def forecast(self, ticker: str, df: pd.DataFrame, horizon: int,
                 name: str = "") -> ForecastResult | None:
        ...

    # Shared helper: standardise + validate an OHLCV frame.
    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def _last_date(df: pd.DataFrame) -> date:
        d = df["date"].iloc[-1]
        return d.date() if hasattr(d, "date") else date.fromisoformat(str(d))
