import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import DATA_DIR, START_DATE, CEDEAR_TICKERS, CEDEAR_SKIP, PORTFOLIO_STOCKS

_ALL_TICKERS = {k: v for k, v in {**CEDEAR_TICKERS, **PORTFOLIO_STOCKS}.items()
                if k not in CEDEAR_SKIP}


class DataFetcher:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _csv_path(self, ticker: str) -> Path:
        return self.data_dir / f"{ticker}.csv"

    def update(self, ticker: str) -> pd.DataFrame:
        csv_path = self._csv_path(ticker)
        today = date.today()
        existing = None

        if csv_path.exists():
            existing = pd.read_csv(csv_path, parse_dates=["Date"])
            existing["Date"] = pd.to_datetime(existing["Date"]).dt.date
            max_date = existing["Date"].max()
            start = max_date + timedelta(days=1)
            if start >= today:
                print(f"[{ticker}] Already up to date (last: {max_date})")
                return existing
        else:
            start = date.fromisoformat(START_DATE)

        end = today + timedelta(days=1)  # yfinance end is exclusive
        print(f"[{ticker}] Fetching {start} to {today} ...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    auto_adjust=True,
                    progress=False,
                )
        except Exception as exc:
            print(f"[{ticker}] WARNING: download failed — {exc}")
            return existing if existing is not None else pd.DataFrame()

        if raw.empty:
            print(f"[{ticker}] No new rows returned")
            return existing if existing is not None else pd.DataFrame()

        # Flatten MultiIndex columns produced by yfinance ≥0.2
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.reset_index()
        raw["Date"] = pd.to_datetime(raw["Date"]).dt.date

        if existing is not None:
            combined = pd.concat([existing, raw], ignore_index=True)
        else:
            combined = raw

        combined = (
            combined.drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )

        combined.to_csv(csv_path, index=False)
        new_rows = len(raw)
        print(f"[{ticker}] +{new_rows} rows  {combined['Date'].min()} to {combined['Date'].max()}")
        return combined

    def update_all(self, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
        ticker_list = tickers if tickers is not None else list(_ALL_TICKERS.keys())
        results = {}
        for ticker in ticker_list:
            try:
                results[ticker] = self.update(ticker)
            except Exception as exc:
                print(f"[{ticker}] WARNING: unexpected error — {exc}")
                results[ticker] = pd.DataFrame()
        return results
