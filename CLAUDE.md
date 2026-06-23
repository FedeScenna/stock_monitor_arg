# spy_model — Claude Code Guide

## Python interpreter
```
C:\Users\feder\anaconda3\python.exe
```

## How to run scripts
```bash
# Update OHLCV market data (all CEDEARs or a single ticker)
/c/Users/feder/anaconda3/python.exe scripts/fetch_data.py
/c/Users/feder/anaconda3/python.exe scripts/fetch_data.py --ticker AAPL

# Rank all CEDEARs by momentum / RSI / Sharpe
/c/Users/feder/anaconda3/python.exe scripts/analyze_cedears.py
/c/Users/feder/anaconda3/python.exe scripts/analyze_cedears.py --top 20 --no-fetch

# Screen full CEDEAR universe for buy candidates (Kronos + momentum)
/c/Users/feder/anaconda3/python.exe scripts/cedear_full_screen.py
/c/Users/feder/anaconda3/python.exe scripts/cedear_full_screen.py --top 30

# Forecast next 21 trading days for portfolio holdings
/c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py

# Walk-forward backtest (RMSE / MAE / directional accuracy)
/c/Users/feder/anaconda3/python.exe scripts/kronos_backtest.py

# Launch Streamlit dashboard
/c/Users/feder/anaconda3/python.exe -m streamlit run app.py
```

## Architecture
```
config/settings.py          — CEDEAR_TICKERS (full Comafi list), CEDEAR_SKIP,
                              PORTFOLIO_CEDEARS, PORTFOLIO_STOCKS, DATA_DIR, etc.
model/
  kronos.py                 — Autoregressive Transformer (Kronos foundation model)
  module.py                 — VQ-VAE tokenizer + Transformer blocks
  __init__.py               — Exports: Kronos, KronosTokenizer, KronosPredictor
src/data/
  fetcher.py                — DataFetcher: incremental OHLCV download + CSV cache
scripts/
  fetch_data.py             — CLI: update OHLCV CSVs for all CEDEARs or one ticker
  analyze_cedears.py        — Rank full CEDEAR universe by momentum/RSI/Sharpe
  cedear_full_screen.py     — Buy candidates: Kronos forecast + momentum screen
  kronos_forecast.py        — 21-day price forecasts for portfolio holdings
  kronos_backtest.py        — Walk-forward backtest with RMSE/MAE metrics
app.py                      — Streamlit dashboard: charts + portfolio overview
```

## Data formats
- **OHLCV data:** `data/cedears/{TICKER}.csv` — columns: Date, Open, High, Low, Close, Volume
- **Analysis outputs:** `data/portfolio/` — dated CSVs (rankings, forecasts, backtests)

## CEDEAR universe
Defined in `config/settings.py` as `CEDEAR_TICKERS` (~240 tickers).
Source: comafi.com.ar/custodiaglobal/Programas-CEDEARs-2483.note.aspx

- Keys are **yfinance-compatible tickers** (may differ from the BYMA ticker).
- `BYMA_TO_YFINANCE` maps BYMA ticker → yfinance ticker for reference.
- `CEDEAR_SKIP` lists tickers excluded from downloads (Russian sanctioned, bankrupt, no yfinance mapping).

## Portfolio holdings
Defined in `config/settings.py`:
- `PORTFOLIO_CEDEARS`: AAPL, AMZN, AVGO, GOOGL, MELI, MSFT, NU, NVDA, TSM, VIST, XLE
- `PORTFOLIO_STOCKS` (Argentine ADRs): PAM (Pampa), YPF

## After fetching data
Load the quant-analyst skill (`.claude/.skills/quant-analyst/SKILL.md`) and ask Claude to
analyze the CEDEARs alongside OHLCV data for momentum, RSI, and risk-adjusted recommendations.
