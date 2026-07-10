# spy_model — Claude Code Guide

## Python interpreter
```
C:\Users\feder\anaconda3\python.exe
```

## How to run scripts
```bash
# One-shot pipeline: refresh all data + Kronos forecast + weekly screen
/c/Users/feder/anaconda3/python.exe scripts/refresh_all.py
/c/Users/feder/anaconda3/python.exe scripts/refresh_all.py --skip-kronos   # skip the heavy forecast

# Update OHLCV market data (all CEDEARs or a single ticker)
/c/Users/feder/anaconda3/python.exe scripts/fetch_data.py
/c/Users/feder/anaconda3/python.exe scripts/fetch_data.py --ticker AAPL

# Rank all CEDEARs by momentum / RSI / Sharpe
/c/Users/feder/anaconda3/python.exe scripts/analyze_cedears.py
/c/Users/feder/anaconda3/python.exe scripts/analyze_cedears.py --top 20 --no-fetch

# Screen full CEDEAR universe for buy candidates (Kronos + momentum)
/c/Users/feder/anaconda3/python.exe scripts/cedear_full_screen.py
/c/Users/feder/anaconda3/python.exe scripts/cedear_full_screen.py --top 30

# Murphy technical-analysis signals (ADX/DMI, Stochastic, Williams %R, OBV, Donchian…)
/c/Users/feder/anaconda3/python.exe scripts/technical_signals.py --no-fetch            # full universe
/c/Users/feder/anaconda3/python.exe scripts/technical_signals.py --no-fetch --portfolio

# Forecast next 21 trading days for portfolio holdings
/c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py

# Walk-forward backtest (RMSE / MAE / directional accuracy)
/c/Users/feder/anaconda3/python.exe scripts/kronos_backtest.py

# Benchmark Kronos vs Nixtla deep models (N-HiTS / PatchTST / TFT), walk-forward
/c/Users/feder/anaconda3/python.exe scripts/forecast_benchmark.py
/c/Users/feder/anaconda3/python.exe scripts/forecast_benchmark.py --tickers AAPL NVDA --no-kronos

# Ensemble 21-day forecast (Kronos + Nixtla deep models, blended)
/c/Users/feder/anaconda3/python.exe scripts/ensemble_forecast.py              # portfolio
/c/Users/feder/anaconda3/python.exe scripts/ensemble_forecast.py --universe   # all CEDEARs (slow)

# Launch Streamlit dashboard
/c/Users/feder/anaconda3/python.exe -m streamlit run app.py
```

## Forecasting models
Two complementary approaches live behind a shared `Forecaster` interface
(`src/forecasting/base.py`), so they can be benchmarked head-to-head and blended:
- **Kronos** (`model/kronos.py`) — pretrained autoregressive Transformer foundation
  model; zero-shot inference, no per-ticker training.
- **Nixtla deep models** (`src/forecasting/neural.py`) — N-HiTS, PatchTST, TFT via
  `neuralforecast`; trained per-ticker on GPU with a quantile (MQLoss) band.
- **Ensemble** (`src/forecasting/ensemble.py`) — inverse-error blend weighted by the
  latest `forecast_benchmark_*.csv` (equal-weight if none exists).

Install the deep-model deps with `requirements-kronos.txt` (needs `neuralforecast`
+ `coreforecast==0.0.15` — newer coreforecast Windows wheels crash on import).

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
  fundamentals.py           — FundamentalsFetcher: tangible book / FCF / debt
src/screening/
  screens.py                — shared indicators + value/technical screen functions
  murphy.py                 — Murphy TA toolkit (ADX/DMI, Stochastic, Williams %R,
                              ROC, OBV, ATR, Donchian) + weight-of-evidence signal score
src/forecasting/
  base.py                   — Forecaster ABC + ForecastResult (shared interface)
  neural.py                 — NeuralForecaster: Nixtla N-HiTS / PatchTST / TFT
  kronos_model.py           — KronosForecaster: Kronos wrapped in the same interface
  ensemble.py               — blend forecasts; inverse-error weights from backtest
scripts/
  fetch_data.py             — CLI: update OHLCV CSVs for all CEDEARs or one ticker
  analyze_cedears.py        — Rank full CEDEAR universe by momentum/RSI/Sharpe
  cedear_full_screen.py     — Buy candidates: Kronos forecast + momentum screen
  weekly_screen.py          — Value (tangible-book) + technical weekly screen
  technical_signals.py      — Murphy TA signals -> data/portfolio/technical_signals_DATE.csv
  kronos_forecast.py        — 21-day Kronos price forecasts (universe or portfolio)
  kronos_backtest.py        — Kronos walk-forward backtest with RMSE/MAE metrics
  forecast_benchmark.py     — Kronos vs Nixtla deep models, walk-forward comparison
  ensemble_forecast.py      — Blended multi-model 21-day forecast -> portfolio CSV
  refresh_all.py            — One-shot pipeline (data + forecasts + screen)
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
