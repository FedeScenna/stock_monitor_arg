# spy_model

Argentine CEDEAR portfolio analysis toolkit using the [Kronos](https://github.com/shiyu-coder/Kronos) foundation model for OHLCV price forecasting, combined with momentum/RSI signals and a Streamlit dashboard.

## Requirements

- Python 3.11
- Dashboard / screening / data deps: `pip install -r requirements.txt`
  (`streamlit`, `plotly`, `pandas`, `numpy`, `yfinance`, `scipy`)
- Kronos forecasting scripts only: `pip install -r requirements-kronos.txt`
  plus PyTorch with the right CUDA wheel (RTX 3060 Laptop GPU used automatically)

> The Streamlit dashboard (`app.py`) does **not** require PyTorch — only the
> `kronos_*` / `cedear_full_screen` scripts do. That keeps the deployed app lean.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (already at `FedeScenna/stock_monitor_arg`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → **Create app**.
3. Repo `FedeScenna/stock_monitor_arg`, branch `main`, main file `app.py` → **Deploy**.

Streamlit Cloud installs `requirements.txt` (the lean app stack). This repo is **code-only**
(no `data/` committed), so on the deployed app the **Stock Charts** page works live via
yfinance, while **Weekly Screen / Kronos Forecast / Portfolio** show "run the script" notices
until their CSVs exist. To populate them, run the scripts locally (e.g. `scripts/weekly_screen.py`,
`scripts/kronos_forecast.py`) and commit the resulting `data/portfolio` + `data/fundamentals`
files, or generate them in your own deployment.

---

## Scripts

### 1. Rank all BYMA CEDEARs (momentum + RSI)

Fetches USD price history for all ~290 BYMA CEDEARs from US exchanges and ranks them by a composite score: momentum (3M/6M), RSI zone, Sharpe ratio, and moving average structure.

```bash
/c/Users/feder/anaconda3/python.exe scripts/analyze_cedears.py
/c/Users/feder/anaconda3/python.exe scripts/analyze_cedears.py --top 30
/c/Users/feder/anaconda3/python.exe scripts/analyze_cedears.py --no-fetch   # use cached CSVs
```

Output: `data/portfolio/cedear_ranking_YYYY-MM-DD.csv`

---

### 2. Kronos price forecast (portfolio assets)

Runs the Kronos-small foundation model to forecast the next 21 trading days for each asset in your current portfolio (see `config/settings.py`).

```bash
/c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py
/c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py --pred-days 10
/c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py --no-fetch
/c/Users/feder/anaconda3/python.exe scripts/kronos_forecast.py --model Kronos-base
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `Kronos-small` | `Kronos-mini` / `Kronos-small` / `Kronos-base` |
| `--pred-days` | `21` | Trading days to forecast |
| `--samples` | `20` | Monte Carlo sample paths (more = smoother) |
| `--no-fetch` | off | Skip download, use cached CSVs |

Output: `data/portfolio/kronos_forecast_YYYY-MM-DD.csv`

---

### 3. Kronos walk-forward backtest (portfolio assets)

Evaluates Kronos forecast accuracy per portfolio asset using a rolling walk-forward strategy over the last 2 years. Reports RMSE, MAE, MAPE%, and directional accuracy per ticker.

```bash
/c/Users/feder/anaconda3/python.exe scripts/kronos_backtest.py
/c/Users/feder/anaconda3/python.exe scripts/kronos_backtest.py --test-days 252 --stride 10
```

| Option | Default | Description |
|--------|---------|-------------|
| `--context` | `400` | Lookback window in trading days (max 512) |
| `--pred-len` | `21` | Forecast horizon per step |
| `--stride` | `21` | Days between steps (21 = non-overlapping) |
| `--test-days` | `504` | Size of test window (~2 years) |
| `--samples` | `10` | Monte Carlo paths per window |

Output: `data/portfolio/kronos_backtest_YYYY-MM-DD.csv`

---

### 4. Full CEDEAR screen — buy recommendations

Downloads USD price history for all ~290 BYMA CEDEARs **not** currently in your portfolio, runs Kronos-small forecasts on each, and outputs ranked buy candidates combining Kronos 21-day upside (40%), RSI zone (15%), and MA structure (10%).

```bash
/c/Users/feder/anaconda3/python.exe scripts/cedear_full_screen.py
/c/Users/feder/anaconda3/python.exe scripts/cedear_full_screen.py --top 30
/c/Users/feder/anaconda3/python.exe scripts/cedear_full_screen.py --no-fetch   # fast re-run
```

| Option | Default | Description |
|--------|---------|-------------|
| `--top` | `25` | Number of top candidates to display |
| `--pred-days` | `21` | Forecast horizon |
| `--samples` | `10` | Monte Carlo paths per asset |
| `--no-fetch` | off | Use cached CSVs (skip download) |

Output: `data/portfolio/cedear_screen_YYYY-MM-DD.csv`

---

### 5. Weekly value + technical screen

Scans the full CEDEAR universe (underlying US equities, USD) and flags each name
against several **independent, non-exclusive** screens:

| # | Screen | Rule |
|---|--------|------|
| A | 💎 Tangible value (Michael Burry) | `P/TBV ≤ 1` **and** FCF > 0 **and** debt/equity < 1 **and** current ratio ≥ 1 **and** tangible book growing YoY |
| B | 📉 Below SMA200 | last close < 200-day SMA |
| C | 📉 Below 20-week MA | last weekly close < 20-week MA |
| D | 🔀 EMA9/21 + MACD cross | EMA9×EMA21 **and** MACD×signal cross, same direction, last 5 bars |
| E | ⚖️ RSI buy / sell | RSI14 ≤ 30 (buy) · RSI14 ≥ 70 (sell) |

A **confluence** score counts how many buy-oriented screens each name hits.

```bash
/c/Users/feder/anaconda3/python.exe scripts/weekly_screen.py
/c/Users/feder/anaconda3/python.exe scripts/weekly_screen.py --no-fetch              # use cached OHLCV + fundamentals
/c/Users/feder/anaconda3/python.exe scripts/weekly_screen.py --skip-fundamentals     # fast, technical screens only
/c/Users/feder/anaconda3/python.exe scripts/weekly_screen.py --min-dollar-vol 5e6    # liquidity floor
```

| Option | Default | Description |
|--------|---------|-------------|
| `--no-fetch` | off | Use cached OHLCV / fundamentals (skip downloads) |
| `--skip-fundamentals` | off | Technical screens only (no tangible-value screen) |
| `--rsi-buy` / `--rsi-sell` | `30` / `70` | RSI buy / sell thresholds |
| `--cross-lookback` | `5` | A cross counts if it occurred within the last N trading days |
| `--min-dollar-vol` | `0` | Drop names below this 20-day avg $ volume (US stock) |
| `--limit` | — | Only screen the first N tickers (testing) |

Output: `data/portfolio/weekly_screen_YYYY-MM-DD.csv` (one row per CEDEAR, boolean flag
columns per screen + metrics). Fundamentals are cached weekly in
`data/fundamentals/fundamentals_YYYY-MM-DD.csv`.

> Fundamentals come from yfinance (`Tangible Book Value`, shares, FCF, debt/equity,
> current ratio) and are slow to fetch across ~240 names — run weekly, then re-use with
> `--no-fetch`. All prices/volumes are the **underlying US equity in USD**, not the BYMA
> peso CEDEAR price.

---

### 6. Streamlit dashboard

Interactive chart viewer (candlesticks), portfolio overview, Kronos forecasts, and the
**Weekly Screen** page — tabs for each screen above plus a confluence master table, a
liquidity slider, and CSV download.

```bash
/c/Users/feder/anaconda3/python.exe -m streamlit run app.py
```

---

## Project structure

```
spy_model/
├── config/
│   └── settings.py              # CEDEAR_TICKERS (full Comafi list), CEDEAR_SKIP,
│                                #   PORTFOLIO_CEDEARS, PORTFOLIO_STOCKS, DATA_DIR
├── model/
│   ├── __init__.py              # KronosTokenizer, Kronos, KronosPredictor
│   ├── kronos.py                # Autoregressive Transformer model
│   └── module.py                # VQ-VAE tokenizer + support modules
├── scripts/
│   ├── fetch_data.py            # CLI: update OHLCV CSVs (all CEDEARs or one ticker)
│   ├── analyze_cedears.py       # Momentum/RSI ranking for all BYMA CEDEARs
│   ├── cedear_full_screen.py    # Full universe screen + buy recommendations
│   ├── weekly_screen.py         # Weekly value + technical screen (tangible value, MAs, MACD, RSI)
│   ├── kronos_forecast.py       # 21-day Kronos forecast for portfolio
│   └── kronos_backtest.py       # Walk-forward backtest (RMSE/MAE per ticker)
├── src/
│   ├── data/
│   │   ├── fetcher.py           # DataFetcher: incremental OHLCV CSV download
│   │   └── fundamentals.py      # FundamentalsFetcher: tangible book / FCF / debt (yfinance)
│   └── screening/
│       └── screens.py           # Shared indicators + tangible-value / technical screen logic
├── data/
│   ├── cedears/                 # OHLCV CSVs — one file per ticker (~290 tickers)
│   ├── fundamentals/            # Weekly fundamentals snapshots (tangible book, FCF, debt)
│   └── portfolio/               # Forecast, backtest, and screen outputs
├── app.py                       # Streamlit dashboard
└── CLAUDE.md                    # Claude Code session guide
```

---

## Configuration

Edit `config/settings.py` to update the portfolio or CEDEAR universe:

```python
# Full official CEDEAR list (~290 tickers, source: Banco Comafi)
# Keys are yfinance-compatible tickers (may differ from BYMA tickers).
CEDEAR_TICKERS = { "AAPL": "Apple Inc", ... }

# Tickers excluded from downloads (Russian sanctioned, bankrupt, no yfinance mapping)
CEDEAR_SKIP = {"OGZD", "LKOD", ...}

# Your current holdings — used by forecast/backtest/screen scripts
PORTFOLIO_CEDEARS = {
    "AAPL": "Apple Inc",
    "NVDA": "Nvidia Corp",
    # ... add/remove as needed
}

PORTFOLIO_STOCKS = {
    "PAM": "Pampa Energia (PAMP ADR)",
    "YPF": "YPF SA (YPFD ADR)",
}
```

All prices are **USD** from US exchanges — not Argentine peso CEDEAR prices.

---

## Kronos model

Kronos is an open-source OHLCV foundation model (decoder-only autoregressive Transformer) trained on 12B+ K-line records from 45+ global exchanges. Downloaded automatically from HuggingFace Hub on first run.

| Variant | Params | Context | HuggingFace ID |
|---------|--------|---------|----------------|
| Kronos-mini | 4.1M | 2048 bars | `NeoQuasar/Kronos-mini` |
| Kronos-small | 24.7M | 512 bars | `NeoQuasar/Kronos-small` |
| Kronos-base | 102.3M | 512 bars | `NeoQuasar/Kronos-base` |

GPU (CUDA) is used automatically when available. Treat Kronos forecasts as probabilistic signals — combine with fundamental analysis before trading.
