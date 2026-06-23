"""Balance-sheet fundamentals for the tangible-value screen.

Pulls the handful of fields needed to evaluate Michael Burry's "tangible value"
stack (price vs tangible book per share + balance-sheet quality) from yfinance
and caches them to ``data/fundamentals/fundamentals_{YYYY-MM-DD}.csv``.

yfinance fundamentals are slow and rate-limit-prone (~1-3s per ticker, one
network call each), so this is a once-a-week batch — never call it live from
the Streamlit app.
"""
from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import FUNDAMENTALS_DIR

# Columns written to the cache CSV (also the dict keys returned by ``fetch``).
COLUMNS = [
    "ticker", "currentPrice", "market_cap",
    "tbv", "tbv_prev", "tbv_yoy", "equity", "shares", "tbvps",
    "p_tbv", "price_to_book", "fcf", "debt_to_equity", "current_ratio",
]

# Below this, a computed P/TBV is almost always a data error (e.g. a dual-class
# share-count mis-scale, as yfinance returns for BRK-B) rather than a real net-net.
P_TBV_FLOOR = 0.05


def _bs_get(bs: pd.DataFrame, row: str, col: int) -> float:
    """Safely read a balance-sheet cell (NaN if row/column absent)."""
    try:
        if row in bs.index and col < bs.shape[1]:
            val = bs.loc[row].iloc[col]
            return float(val) if pd.notna(val) else np.nan
    except Exception:
        pass
    return np.nan


class FundamentalsFetcher:
    def __init__(self, data_dir: Path = FUNDAMENTALS_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _csv_path(self, day: date | None = None) -> Path:
        day = day or date.today()
        return self.data_dir / f"fundamentals_{day.isoformat()}.csv"

    # -- single ticker --------------------------------------------------------

    def fetch(self, ticker: str) -> dict:
        """Fetch tangible-value fundamentals for one ticker (graceful on failure)."""
        row = {c: np.nan for c in COLUMNS}
        row["ticker"] = ticker

        try:
            t = yf.Ticker(ticker)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = t.info or {}
                bs = t.balance_sheet
        except Exception as exc:
            print(f"[{ticker}] fundamentals WARNING: {exc}")
            return row

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        row["currentPrice"] = float(price) if price else np.nan
        row["market_cap"] = float(info.get("marketCap")) if info.get("marketCap") else np.nan
        ptb = info.get("priceToBook")
        ptb = float(ptb) if ptb else np.nan
        row["price_to_book"] = ptb
        row["fcf"] = float(info.get("freeCashflow")) if info.get("freeCashflow") is not None else np.nan
        row["current_ratio"] = float(info.get("currentRatio")) if info.get("currentRatio") else np.nan

        # yfinance reports debtToEquity as a percentage (e.g. 79.5 == 0.795x)
        dte = info.get("debtToEquity")
        row["debt_to_equity"] = float(dte) / 100.0 if dte is not None else np.nan

        # Tangible book value, equity, shares from the balance sheet
        tbv = tbv_prev = eq = shares = np.nan
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            tbv = _bs_get(bs, "Tangible Book Value", 0)
            tbv_prev = _bs_get(bs, "Tangible Book Value", 1)
            eq = _bs_get(bs, "Stockholders Equity", 0)
            if pd.isna(eq):
                eq = _bs_get(bs, "Common Stock Equity", 0)
            shares = _bs_get(bs, "Ordinary Shares Number", 0)
        if pd.isna(shares) or not shares:
            so = info.get("sharesOutstanding")
            shares = float(so) if so else np.nan

        row["tbv"] = tbv
        row["tbv_prev"] = tbv_prev
        row["equity"] = eq
        row["shares"] = shares

        # Price-to-tangible-book via a CURRENCY-INVARIANT formula:
        #   P/TBV = priceToBook x (Stockholders Equity / Tangible Book Value)
        # priceToBook (info) is already in the listing currency (USD for ADRs);
        # equity/tbv is a unitless same-currency ratio. This avoids the local-vs-USD
        # mismatch that breaks a raw price / (tbv/shares) calc for foreign ADRs
        # (e.g. Japanese/Korean names reporting their balance sheet in JPY/KRW).
        if pd.notna(ptb) and ptb > 0 and pd.notna(eq) and eq > 0 and pd.notna(tbv) and tbv > 0:
            p_tbv = ptb * (eq / tbv)
            if p_tbv >= P_TBV_FLOOR:
                row["p_tbv"] = round(p_tbv, 4)
                if pd.notna(price) and price > 0:
                    row["tbvps"] = round(float(price) / p_tbv, 4)  # USD tangible book / share
        if pd.notna(tbv) and pd.notna(tbv_prev) and tbv_prev > 0:
            row["tbv_yoy"] = round(tbv / tbv_prev - 1.0, 4)

        return row

    # -- batch ----------------------------------------------------------------

    def update_all(self, tickers: list[str], force: bool = False) -> pd.DataFrame:
        """Fetch fundamentals for every ticker and cache to today's CSV.

        If today's cache already exists and ``force`` is False, it is returned
        as-is (weekly freshness — re-runs the same day are free).
        """
        path = self._csv_path()
        if path.exists() and not force:
            print(f"Fundamentals already cached today -> {path.name}")
            return pd.read_csv(path)

        rows = []
        n = len(tickers)
        print(f"\nFetching fundamentals for {n} tickers (this is slow) ...")
        for i, ticker in enumerate(tickers, 1):
            rows.append(self.fetch(ticker))
            if i % 25 == 0 or i == n:
                print(f"  [{i:>3}/{n}] done")

        df = pd.DataFrame(rows, columns=COLUMNS)
        df.to_csv(path, index=False)
        hits = int(df["p_tbv"].notna().sum())
        print(f"Fundamentals saved -> {path}  ({hits}/{n} with tangible book)")
        return df

    def load_latest(self) -> pd.DataFrame | None:
        """Return the most recent cached fundamentals CSV, or None if absent."""
        files = sorted(self.data_dir.glob("fundamentals_*.csv"))
        if not files:
            return None
        print(f"Loaded cached fundamentals <- {files[-1].name}")
        return pd.read_csv(files[-1])
