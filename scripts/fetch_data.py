#!/usr/bin/env python
"""CLI entry point for daily market data updates."""
import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CEDEAR_TICKERS, CEDEAR_SKIP, PORTFOLIO_STOCKS
from src.data.fetcher import DataFetcher

_VALID_TICKERS = {k for k in {**CEDEAR_TICKERS, **PORTFOLIO_STOCKS} if k not in CEDEAR_SKIP}


def main():
    parser = argparse.ArgumentParser(description="Fetch / update market data CSVs")
    parser.add_argument(
        "--ticker",
        metavar="SYMBOL",
        help="Update a single ticker instead of all",
    )
    args = parser.parse_args()

    fetcher = DataFetcher()

    if args.ticker:
        ticker = args.ticker.upper()
        if ticker not in _VALID_TICKERS:
            print(f"Unknown ticker '{ticker}'. Valid tickers: {', '.join(sorted(_VALID_TICKERS))}")
            sys.exit(1)
        fetcher.update(ticker)
    else:
        fetcher.update_all()


if __name__ == "__main__":
    main()
