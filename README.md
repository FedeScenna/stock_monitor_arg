# spy_model

Argentine CEDEAR portfolio analysis toolkit. Forecasts OHLCV prices with the
[Kronos](https://github.com/shiyu-coder/Kronos) foundation model **and** Nixtla
deep time-series models (N-HiTS / PatchTST / TFT) — benchmarked head-to-head and
blended into an ensemble — alongside a full technical-analysis signal engine
(John Murphy's toolkit) and value/momentum screens, all surfaced in a Streamlit
dashboard.

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

### 0. One-shot pipeline — refresh everything

Runs the whole flow in order (each source refreshed once; analyses run `--no-fetch`):

1. **OHLCV refresh** — full CEDEAR universe + portfolio
2. **Fundamentals** — tangible book / FCF / debt for the value screen
3. **Kronos forecast** — 21-day price forecast
4. **Weekly screen** — value + technical
5. **Technical signals** — Murphy TA weight-of-evidence
6. **Ensemble forecast** — Kronos + Nixtla deep models *(opt-in: `--with-ensemble`)*

```bash
/c/Users/feder/anaconda3/python.exe scripts/refresh_all.py
/c/Users/feder/anaconda3/python.exe scripts/refresh_all.py --no-fetch            # caches only
/c/Users/feder/anaconda3/python.exe scripts/refresh_all.py --skip-fundamentals   # faster, no value screen
/c/Users/feder/anaconda3/python.exe scripts/refresh_all.py --skip-kronos
/c/Users/feder/anaconda3/python.exe scripts/refresh_all.py --with-ensemble       # add the multi-model ensemble
/c/Users/feder/anaconda3/python.exe scripts/refresh_all.py --kronos-model Kronos-base --kronos-samples 50
```

Outputs: `data/portfolio/{kronos_forecast,weekly_screen,technical_signals}_YYYY-MM-DD.csv`
(plus `ensemble_forecast_YYYY-MM-DD.csv` with `--with-ensemble`). Requires the Kronos
deps (`requirements-kronos.txt`) unless you pass `--skip-kronos`.

---

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
| `--model` | `Kronos-base` | `Kronos-mini` / `Kronos-small` / `Kronos-base` |
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

### 6. Murphy technical-analysis signals

Scores each name with the classic indicator toolkit from John J. Murphy's *Technical
Analysis of the Financial Markets*, combined into a net weight-of-evidence rating
(STRONG BUY / BUY / HOLD / SELL / STRONG SELL). Ten rules each vote +1 / −1 / 0, split
into two families:

| Family | Rules |
|--------|-------|
| **Trend-following** | 50/200 Golden/Death cross · MA alignment · ADX / +DI / −DI (Wilder) · MACD · On-Balance Volume · Donchian 52-week breakout |
| **Oscillators** | RSI · Slow Stochastic · Williams %R · Rate of Change |

The DI-cross vote is gated on **ADX ≥ 25** (Murphy: trend tools only work in trending
markets). Wilder ATR is also computed for volatility-scaled stops.

```bash
/c/Users/feder/anaconda3/python.exe scripts/technical_signals.py               # full universe
/c/Users/feder/anaconda3/python.exe scripts/technical_signals.py --no-fetch     # use cached OHLCV
/c/Users/feder/anaconda3/python.exe scripts/technical_signals.py --portfolio    # holdings only
```

Output: `data/portfolio/technical_signals_YYYY-MM-DD.csv` (rating, `trend_score`,
`osc_score`, `murphy_score`, per-rule votes + indicator readings).

---

### 7. Multi-model forecast benchmark + ensemble

Two complementary forecasters live behind a shared `Forecaster` interface
(`src/forecasting/base.py`) so they can be scored head-to-head and blended:

- **Kronos** — pretrained autoregressive Transformer (zero-shot, no per-ticker training).
- **Nixtla deep models** — N-HiTS / PatchTST / TFT via `neuralforecast`, trained per
  ticker on GPU with a quantile (MQLoss) band.
- **Ensemble** — inverse-error blend weighted by the latest walk-forward benchmark.

```bash
# Walk-forward benchmark: Kronos vs deep models, strictly out-of-sample
/c/Users/feder/anaconda3/python.exe scripts/forecast_benchmark.py
/c/Users/feder/anaconda3/python.exe scripts/forecast_benchmark.py --tickers AAPL NVDA --no-kronos

# Blended 21-day ensemble forecast (weights from the benchmark)
/c/Users/feder/anaconda3/python.exe scripts/ensemble_forecast.py                # portfolio
/c/Users/feder/anaconda3/python.exe scripts/ensemble_forecast.py --universe     # all CEDEARs (slow)
```

Outputs: `data/portfolio/forecast_benchmark_YYYY-MM-DD.csv` (RMSE / MAE / MAPE /
directional accuracy per model) and `ensemble_forecast_YYYY-MM-DD.csv` (per-model +
blended paths). Deep models need `requirements-kronos.txt` — which pins
`coreforecast==0.0.15` (newer Windows wheels segfault on import).

---

### 8. Streamlit dashboard

Interactive chart viewer (candlesticks), portfolio overview, Kronos forecasts, plus:

- **Weekly Screen** — tabs per screen, confluence master table, liquidity slider, CSV download.
- **Technical Signals** — Murphy rating cards, colour-coded per-rule vote table, and a
  3-panel indicator chart (price+SMA+Donchian / ADX+DI / RSI+Stochastic).
- **Ensemble Forecast** — per-model upside table + blended forecast paths with a band.
- **Backtest Results** — model leaderboard, MAPE / directional-accuracy bars, per-ticker heatmap.

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
│   ├── technical_signals.py     # Murphy TA signals (ADX/DMI, Stochastic, %R, OBV, Donchian…)
│   ├── kronos_forecast.py       # 21-day Kronos forecast for portfolio
│   ├── kronos_backtest.py       # Walk-forward backtest (RMSE/MAE per ticker)
│   ├── forecast_benchmark.py    # Kronos vs Nixtla deep models, walk-forward comparison
│   ├── ensemble_forecast.py     # Blended multi-model 21-day forecast
│   └── refresh_all.py           # One-shot pipeline (data + forecasts + screens + signals)
├── src/
│   ├── data/
│   │   ├── fetcher.py           # DataFetcher: incremental OHLCV CSV download
│   │   └── fundamentals.py      # FundamentalsFetcher: tangible book / FCF / debt (yfinance)
│   ├── screening/
│   │   ├── screens.py           # Shared indicators + tangible-value / technical screen logic
│   │   └── murphy.py            # Murphy TA toolkit + weight-of-evidence signal score
│   └── forecasting/
│       ├── base.py              # Forecaster ABC + ForecastResult (shared interface)
│       ├── neural.py            # NeuralForecaster: Nixtla N-HiTS / PatchTST / TFT
│       ├── kronos_model.py      # KronosForecaster wrapped in the same interface
│       └── ensemble.py          # Inverse-error blend; weights from the benchmark
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
