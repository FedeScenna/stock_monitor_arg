"""Streamlit dashboard — Stock Charts + Portfolio Overview."""
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import DATA_DIR, PORTFOLIO_DIR, TICKERS, RSI_BUY, RSI_SELL
from src.screening.screens import compute_indicators
from src.screening.murphy import TREND_SIGNALS, OSC_SIGNALS

ALL_TICKERS = dict(sorted(TICKERS.items()))

st.set_page_config(page_title="Spy Model", layout="wide", page_icon="📈")

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_ohlcv(ticker: str) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{ticker}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["Date"])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(ticker, start="2000-01-01", auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def slice_by_range(df: pd.DataFrame, time_range: str) -> pd.DataFrame:
    if time_range == "All" or df.empty:
        return df
    months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}[time_range]
    cutoff = df["Date"].max() - pd.DateOffset(months=months)
    return df[df["Date"] >= cutoff]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Stock Charts", "Weekly Screen", "Technical Signals", "Portfolio Overview",
     "Kronos Forecast", "Ensemble Forecast", "Backtest Results"],
)

if page == "Stock Charts":
    ticker = st.sidebar.selectbox("Ticker", list(ALL_TICKERS.keys()),
                                  format_func=lambda t: f"{t} — {ALL_TICKERS[t]}")
    time_range = st.sidebar.radio("Time range", ["1M", "3M", "6M", "1Y", "All"], index=3)

if page == "Kronos Forecast":
    forecast_files = sorted(PORTFOLIO_DIR.glob("kronos_forecast_*.csv"))
    if forecast_files:
        selected_file = st.sidebar.selectbox(
            "Forecast date",
            forecast_files,
            index=len(forecast_files) - 1,
            format_func=lambda p: p.stem.replace("kronos_forecast_", ""),
        )

if page == "Ensemble Forecast":
    ensemble_files = sorted(PORTFOLIO_DIR.glob("ensemble_forecast_*.csv"))
    if ensemble_files:
        selected_ens_file = st.sidebar.selectbox(
            "Forecast date",
            ensemble_files,
            index=len(ensemble_files) - 1,
            format_func=lambda p: p.stem.replace("ensemble_forecast_", ""),
        )

if page == "Backtest Results":
    bench_files = sorted(PORTFOLIO_DIR.glob("forecast_benchmark_*.csv"))
    if bench_files:
        selected_bench_file = st.sidebar.selectbox(
            "Benchmark run",
            bench_files,
            index=len(bench_files) - 1,
            format_func=lambda p: p.stem.replace("forecast_benchmark_", ""),
        )

# ---------------------------------------------------------------------------
# Page 1 — Stock Charts
# ---------------------------------------------------------------------------

if page == "Stock Charts":
    st.title(f"{ticker} — {ALL_TICKERS.get(ticker, '')}")

    df_full = load_ohlcv(ticker)
    if df_full.empty:
        st.error(f"No data available for {ticker}.")
        st.stop()

    df_full = compute_indicators(df_full)
    df = slice_by_range(df_full, time_range)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.17, 0.17],
        vertical_spacing=0.02,
    )

    # --- Row 1: Candlestick + SMAs + BBands ---
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)

    # Bollinger band fill
    fig.add_trace(go.Scatter(
        x=pd.concat([df["Date"], df["Date"][::-1]]),
        y=pd.concat([df["BB_upper"], df["BB_lower"][::-1]]),
        fill="toself", fillcolor="rgba(150,150,150,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="BB", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_upper"], line=dict(color="grey", width=1, dash="dot"), name="BB upper", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_lower"], line=dict(color="grey", width=1, dash="dot"), name="BB lower", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], line=dict(color="orange", width=1.5), name="SMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA200"], line=dict(color="#4fc3f7", width=1.5), name="SMA 200"), row=1, col=1)

    # --- Row 2: Volume ---
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], marker_color=colors, name="Volume", showlegend=False), row=2, col=1)

    # --- Row 3: RSI ---
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], line=dict(color="#ab47bc", width=1.5), name="RSI 14"), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="red", dash="dash", width=1), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="green", dash="dash", width=1), row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # --- Row 4: MACD ---
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_hist"], marker_color=hist_colors, name="MACD hist", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], line=dict(color="#29b6f6", width=1.5), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_signal"], line=dict(color="orange", width=1.5), name="Signal"), row=4, col=1)

    fig.update_layout(
        height=900,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 2 — Portfolio Overview
# ---------------------------------------------------------------------------

elif page == "Portfolio Overview":
    st.title("Portfolio Overview")

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    portfolio_files = sorted(PORTFOLIO_DIR.glob("portfolio_*.csv"))

    if not portfolio_files:
        st.info(
            "No portfolio CSV found. Place a file named `portfolio_YYYYMMDD.csv` "
            f"inside `{PORTFOLIO_DIR}` to see your holdings here.\n\n"
            "Expected columns: `category`, `ticker`, `description`, `quantity`, `price_usd`, `value_usd`"
        )
        st.stop()

    latest = portfolio_files[-1]
    df_port = pd.read_csv(latest)

    # Normalise column names
    df_port.columns = [c.strip().lower() for c in df_port.columns]
    required = {"category", "ticker", "description", "quantity", "price_usd", "value_usd"}
    missing = required - set(df_port.columns)
    if missing:
        st.error(f"Portfolio CSV is missing columns: {missing}")
        st.stop()

    total_value = df_port["value_usd"].sum()
    largest = df_port.loc[df_port["value_usd"].idxmax()]
    n_holdings = len(df_port)

    # Metric cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Value", f"${total_value:,.2f}")
    c2.metric("Largest Position", f"{largest['ticker']} — ${largest['value_usd']:,.2f}")
    c3.metric("# Holdings", n_holdings)

    st.divider()

    # Pie chart by category
    cat_totals = df_port.groupby("category")["value_usd"].sum().reset_index()
    fig_pie = go.Figure(go.Pie(
        labels=cat_totals["category"],
        values=cat_totals["value_usd"],
        hole=0.35,
        textinfo="label+percent",
    ))
    fig_pie.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
    )
    st.subheader("Allocation by Category")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Holdings table
    st.subheader("Holdings")
    display = df_port[["category", "ticker", "description", "quantity", "price_usd", "value_usd"]].copy()
    display = display.sort_values("value_usd", ascending=False).reset_index(drop=True)
    display["price_usd"] = display["price_usd"].map("${:,.2f}".format)
    display["value_usd"] = display["value_usd"].map("${:,.2f}".format)
    st.dataframe(display, use_container_width=True)

    st.caption(f"Source: `{latest.name}`")

# ---------------------------------------------------------------------------
# Page 3 — Kronos Forecast
# ---------------------------------------------------------------------------

elif page == "Kronos Forecast":
    st.title("Kronos Price Forecast")

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    forecast_files = sorted(PORTFOLIO_DIR.glob("kronos_forecast_*.csv"))

    if not forecast_files:
        st.info(
            "No forecast CSV found. Run `scripts/kronos_forecast.py` to generate one.\n\n"
            f"Expected location: `{PORTFOLIO_DIR}/kronos_forecast_YYYY-MM-DD.csv`"
        )
        st.stop()

    df_fc = pd.read_csv(selected_file, parse_dates=["forecast_date"])

    forecast_date = df_fc["forecast_date"].iloc[0].date()

    # --- Summary table ---
    summary = (
        df_fc.groupby("ticker")
        .agg(
            name=("name", "first"),
            last_close=("last_close", "first"),
            pred_7d=("pred_close", lambda s: s.iloc[6] if len(s) >= 7 else s.iloc[-1]),
            pred_14d=("pred_close", lambda s: s.iloc[13] if len(s) >= 14 else s.iloc[-1]),
            pred_21d=("pred_close", lambda s: s.iloc[20] if len(s) >= 21 else s.iloc[-1]),
        )
        .reset_index()
    )
    has_conf = "confidence" in df_fc.columns
    if has_conf:
        conf_map = df_fc.groupby("ticker")["confidence"].first()
        summary["confidence"] = summary["ticker"].map(conf_map)
    summary["up_7d"]  = (summary["pred_7d"]  / summary["last_close"] - 1) * 100
    summary["up_14d"] = (summary["pred_14d"] / summary["last_close"] - 1) * 100
    summary["up_21d"] = (summary["pred_21d"] / summary["last_close"] - 1) * 100
    summary = summary.sort_values("up_21d", ascending=False).reset_index(drop=True)

    st.subheader(f"21-day Outlook — {forecast_date}")
    st.caption("Forecast generated by the Kronos foundation model. "
               "Confidence = share of sample paths agreeing on the 21-day direction.")

    # Confidence gate: Kronos should be at least 55% confident to count as a signal.
    MIN_CONF = 0.55
    if has_conf:
        n_total = len(summary)
        n_ok = int((summary["confidence"] >= MIN_CONF).sum())
        only_conf = st.checkbox(
            f"Only show predictions Kronos is ≥{MIN_CONF*100:.0f}% confident about "
            f"({n_ok} of {n_total} qualify)",
            value=True,
        )
        if only_conf:
            summary = summary[summary["confidence"] >= MIN_CONF].reset_index(drop=True)
        if summary.empty:
            st.info(f"No predictions reach the ≥{MIN_CONF*100:.0f}% confidence bar for this date.")
            st.stop()
    else:
        st.caption("⚠️ This forecast file predates the confidence feature — "
                   "re-run `scripts/kronos_forecast.py` to populate it.")

    # Colour-coded metric cards (top row) — cap to the top movers so a
    # full-universe forecast (~240 tickers) doesn't blow up the layout.
    N_CARDS = 8
    card_rows = summary.head(min(N_CARDS, len(summary)))
    cols = st.columns(len(card_rows))
    for col, (_, row) in zip(cols, card_rows.iterrows()):
        col.metric(
            label=row["ticker"],
            value=f"${row['pred_21d']:.2f}",
            delta=f"{row['up_21d']:+.1f}%",
        )
    if len(summary) > N_CARDS:
        st.caption(f"Showing top {N_CARDS} of {len(summary)} assets by 21-day upside — full list below.")

    st.divider()

    # Summary data table
    summary_cols = ["ticker", "name", "last_close", "pred_7d", "up_7d", "pred_14d", "up_14d", "pred_21d", "up_21d"]
    new_names = ["Ticker", "Name", "Last", "7d Price", "7d %", "14d Price", "14d %", "21d Price", "21d %"]
    if has_conf:
        summary_cols.append("confidence")
        new_names.append("Conf")
    display_summary = summary[summary_cols].copy()
    display_summary.columns = new_names

    def color_pct(val):
        if isinstance(val, float):
            color = "#26a69a" if val >= 0 else "#ef5350"
            return f"color: {color}"
        return ""

    def color_conf(val):
        # green if >= 55%, amber otherwise
        if isinstance(val, float):
            return f"color: {'#26a69a' if val >= MIN_CONF else '#ffb74d'}"
        return ""

    fmt = {
        "Last":     "${:.2f}",
        "7d Price": "${:.2f}",
        "14d Price":"${:.2f}",
        "21d Price":"${:.2f}",
        "7d %":     "{:+.1f}%",
        "14d %":    "{:+.1f}%",
        "21d %":    "{:+.1f}%",
    }
    if has_conf:
        fmt["Conf"] = "{:.0%}"
    styled = display_summary.style.format(fmt).map(color_pct, subset=["7d %", "14d %", "21d %"])
    if has_conf:
        styled = styled.map(color_conf, subset=["Conf"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # --- Per-ticker forecast charts ---
    st.subheader("Forecast Charts")
    selected_ticker = st.selectbox(
        "Select asset",
        summary["ticker"].tolist(),
        format_func=lambda t: f"{t} — {summary.loc[summary['ticker']==t, 'name'].values[0]}",
    )

    df_ticker_fc = df_fc[df_fc["ticker"] == selected_ticker].copy()
    last_close = df_ticker_fc["last_close"].iloc[0]

    # Load recent OHLCV history for context (last 60 trading days)
    hist_csv = DATA_DIR / f"{selected_ticker}.csv"
    if hist_csv.exists():
        df_hist = pd.read_csv(hist_csv, parse_dates=["Date"])
        df_hist = df_hist.sort_values("Date").tail(60).reset_index(drop=True)
        hist_dates = df_hist["Date"].tolist()
        hist_close = df_hist["Close"].tolist()
    else:
        df_hist = pd.DataFrame()
        hist_dates, hist_close = [], []

    # Build forecast dates (trading days starting from forecast_date)
    fc_dates = pd.bdate_range(start=forecast_date, periods=len(df_ticker_fc)).tolist()
    fc_close = df_ticker_fc["pred_close"].tolist()

    fig_fc = go.Figure()

    # Historical line
    if hist_dates:
        fig_fc.add_trace(go.Scatter(
            x=hist_dates, y=hist_close,
            mode="lines",
            line=dict(color="#4fc3f7", width=2),
            name="Historical close",
        ))
        # Anchor dot at last close
        fig_fc.add_trace(go.Scatter(
            x=[hist_dates[-1]], y=[last_close],
            mode="markers",
            marker=dict(color="#4fc3f7", size=8),
            showlegend=False,
        ))

    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x=fc_dates, y=fc_close,
        mode="lines+markers",
        line=dict(color="#ab47bc", width=2, dash="dot"),
        marker=dict(size=4),
        name="Kronos forecast",
    ))

    # Horizontal reference at last close
    fig_fc.add_hline(
        y=last_close,
        line=dict(color="grey", dash="dash", width=1),
        annotation_text=f"Last close ${last_close:.2f}",
        annotation_position="bottom right",
    )

    # Day markers: 7, 14, 21. Pass x as epoch milliseconds — plotly's add_vline
    # chokes on date strings / Timestamps under pandas 2.x.
    for day, label in [(7, "7d"), (14, "14d"), (21, "21d")]:
        if day <= len(fc_dates):
            fig_fc.add_vline(
                x=int(fc_dates[day - 1].timestamp() * 1000),
                line=dict(color="rgba(255,255,255,0.2)", dash="dot"),
                annotation_text=label,
                annotation_position="top",
            )

    row_summary = summary[summary["ticker"] == selected_ticker].iloc[0]
    up21 = row_summary["up_21d"]
    arrow = "+" if up21 >= 0 else ""
    fig_fc.update_layout(
        title=f"{selected_ticker} — {arrow}{up21:.1f}% predicted in 21 trading days",
        template="plotly_dark",
        height=450,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 4 — Weekly Screen (tangible value + technical screens)
# ---------------------------------------------------------------------------

elif page == "Weekly Screen":
    st.title("Weekly CEDEAR Screen")
    st.caption(
        "Independent, non-exclusive screens over the underlying US equities (USD). "
        "Generate / refresh with `scripts/weekly_screen.py`."
    )

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    screen_files = sorted(PORTFOLIO_DIR.glob("weekly_screen_*.csv"))
    if not screen_files:
        st.info(
            "No screen found. Run "
            "`python scripts/weekly_screen.py` (add `--skip-fundamentals` for a fast "
            "technical-only run) to generate "
            f"`{PORTFOLIO_DIR}/weekly_screen_YYYY-MM-DD.csv`."
        )
        st.stop()

    selected = st.selectbox(
        "Screen date", screen_files, index=len(screen_files) - 1,
        format_func=lambda p: p.stem.replace("weekly_screen_", ""),
    )
    df = pd.read_csv(selected)

    # --- Liquidity filter (underlying US stock's 20-day avg $ volume) -------
    if "dollar_vol_20d" in df.columns and df["dollar_vol_20d"].notna().any():
        max_dv = float(df["dollar_vol_20d"].max())
        dv_floor_m = st.slider(
            "Min liquidity — 20-day avg $ volume (US stock), $M", 0.0,
            round(max_dv / 1e6, 1), 0.0, step=1.0,
        )
        if dv_floor_m > 0:
            df = df[df["dollar_vol_20d"] >= dv_floor_m * 1e6]
    df = df.reset_index(drop=True)

    # --- Helpers -----------------------------------------------------------
    def _check(s):
        return s.map(lambda v: "✓" if bool(v) else "")

    def _money(v):
        if pd.isna(v):
            return "—"
        for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
            if abs(v) >= div:
                return f"${v / div:.1f}{unit}"
        return f"${v:.0f}"

    has_fund = "p_tbv" in df.columns and df["p_tbv"].notna().any()

    # --- Metric cards ------------------------------------------------------
    cols = st.columns(7)
    cols[0].metric("Universe", len(df))
    cols[1].metric("Tangible value", int(df.get("value_hit", pd.Series(dtype=bool)).sum()))
    cols[2].metric("Below SMA200", int(df.get("below_sma200", pd.Series(dtype=bool)).sum()))
    cols[3].metric("Below 20-wk MA", int(df.get("below_wma20", pd.Series(dtype=bool)).sum()))
    cols[4].metric("Cross ↑", int(df.get("cross_up", pd.Series(dtype=bool)).sum()))
    cols[5].metric(f"RSI buy ≤{RSI_BUY:.0f}", int(df.get("rsi_buy", pd.Series(dtype=bool)).sum()))
    cols[6].metric(f"RSI sell ≥{RSI_SELL:.0f}", int(df.get("rsi_sell", pd.Series(dtype=bool)).sum()))

    st.download_button(
        "⬇ Download screen CSV", df.to_csv(index=False).encode(),
        file_name=selected.name, mime="text/csv",
    )
    st.divider()

    tabs = st.tabs([
        "💎 Tangible Value", "📉 Below SMA200", "📉 Below 20-wk MA",
        "🔀 EMA9/21 + MACD", "⚖️ RSI buy / sell", "🎯 Confluence",
    ])

    # --- A. Tangible value -------------------------------------------------
    with tabs[0]:
        st.caption(
            "Burry stack: **P/TBV ≤ 1** AND positive free cash flow AND debt/equity < 1 "
            "AND current ratio ≥ 1 AND tangible book growing YoY. "
            "Table below ranks every name by cheapest price-to-tangible-book."
        )
        if not has_fund:
            st.info("This screen was generated with `--skip-fundamentals`. "
                    "Re-run `python scripts/weekly_screen.py` (without that flag) to populate it.")
        else:
            hits = df[df["value_hit"] == True]  # noqa: E712
            st.markdown(f"**Full value hits: {len(hits)}**")
            ranked = df[df["p_tbv"].notna()].sort_values("p_tbv").head(50).copy()
            show = pd.DataFrame({
                "Ticker": ranked["ticker"], "Name": ranked["name"].str.slice(0, 26),
                "Price": ranked["last_close"].map(lambda v: f"${v:,.2f}"),
                "P/TBV": ranked["p_tbv"].map(lambda v: f"{v:.2f}"),
                "TBV/sh": ranked["tbvps"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—"),
                "FCF>0": _check(ranked["q_fcf"]), "D/E<1": _check(ranked["q_dte"]),
                "CR≥1": _check(ranked["q_cr"]), "TBV↑": _check(ranked["q_tbv_growth"]),
                "VALUE": _check(ranked["value_hit"]),
                "$Vol": ranked["dollar_vol_20d"].map(_money),
            })
            st.dataframe(show, use_container_width=True, hide_index=True)

    # --- generic technical-list renderer -----------------------------------
    def _tech_table(sub, extra_col, extra_label, fmt, ascending_sort, sort_col):
        if sub.empty:
            st.info("No names match this screen for the selected date.")
            return
        sub = sub.sort_values(sort_col, ascending=ascending_sort)
        show = pd.DataFrame({
            "Ticker": sub["ticker"], "Name": sub["name"].str.slice(0, 28),
            "Price": sub["last_close"].map(lambda v: f"${v:,.2f}"),
            extra_label: sub[extra_col].map(fmt),
            "RSI": sub["rsi"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "—"),
            "$Vol": sub["dollar_vol_20d"].map(_money),
            "Conf": sub["confluence"],
        })
        st.dataframe(show, use_container_width=True, hide_index=True)

    # --- B. Below SMA200 ---------------------------------------------------
    with tabs[1]:
        st.caption("Last close below the 200-day simple moving average.")
        sub = df[df["below_sma200"] == True].copy()  # noqa: E712
        sub["pct_below"] = (sub["last_close"] / sub["sma200"] - 1) * 100
        _tech_table(sub, "pct_below", "vs SMA200",
                    lambda v: f"{v:+.1f}%", True, "pct_below")

    # --- C. Below 20-week MA ----------------------------------------------
    with tabs[2]:
        st.caption("Last weekly close below the 20-week moving average.")
        sub = df[df["below_wma20"] == True].copy()  # noqa: E712
        sub["pct_below"] = (sub["last_close"] / sub["wma20"] - 1) * 100
        _tech_table(sub, "pct_below", "vs 20wMA",
                    lambda v: f"{v:+.1f}%", True, "pct_below")

    # --- D. EMA9/21 + MACD cross ------------------------------------------
    with tabs[3]:
        st.caption("EMA9 crossed EMA21 **and** MACD crossed its signal, same direction, "
                   "within the last 5 trading days.")
        sub = df[(df["cross_up"] == True) | (df["cross_down"] == True)].copy()  # noqa: E712
        if sub.empty:
            st.info("No fresh crosses for the selected date.")
        else:
            sub["dir"] = sub["cross_up"].map(lambda v: "▲ Bullish" if v else "▼ Bearish")
            sub = sub.sort_values("cross_up", ascending=False)
            show = pd.DataFrame({
                "Ticker": sub["ticker"], "Name": sub["name"].str.slice(0, 26),
                "Direction": sub["dir"],
                "Price": sub["last_close"].map(lambda v: f"${v:,.2f}"),
                "EMA9": sub["ema9"].map(lambda v: f"{v:,.2f}"),
                "EMA21": sub["ema21"].map(lambda v: f"{v:,.2f}"),
                "RSI": sub["rsi"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "—"),
                "Conf": sub["confluence"],
            })
            st.dataframe(show, use_container_width=True, hide_index=True)

    # --- E. RSI buy / sell -------------------------------------------------
    with tabs[4]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Buy zone — RSI ≤ {RSI_BUY:.0f} (oversold)**")
            sub = df[df["rsi_buy"] == True].copy()  # noqa: E712
            _tech_table(sub, "rsi", "RSI", lambda v: f"{v:.0f}", True, "rsi")
        with c2:
            st.markdown(f"**Sell zone — RSI ≥ {RSI_SELL:.0f} (overbought)**")
            sub = df[df["rsi_sell"] == True].copy()  # noqa: E712
            _tech_table(sub, "rsi", "RSI", lambda v: f"{v:.0f}", False, "rsi")

    # --- F. Confluence -----------------------------------------------------
    with tabs[5]:
        st.caption("Names hitting multiple buy-oriented screens "
                   "(tangible value, below SMA200, below 20-wk MA, bullish cross, RSI buy).")
        sub = df[df["confluence"] > 0].sort_values(
            ["confluence", "p_tbv"], ascending=[False, True]).copy()
        if sub.empty:
            st.info("No names hit a buy screen for the selected date.")
        else:
            show = pd.DataFrame({
                "Ticker": sub["ticker"], "Name": sub["name"].str.slice(0, 24),
                "Conf": sub["confluence"],
                "Value": _check(sub.get("value_hit", False)),
                "<SMA200": _check(sub["below_sma200"]),
                "<20wMA": _check(sub["below_wma20"]),
                "Cross↑": _check(sub["cross_up"]),
                "RSI-buy": _check(sub["rsi_buy"]),
                "P/TBV": sub["p_tbv"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
                "RSI": sub["rsi"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "—"),
            })
            st.dataframe(show, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Page 5 — Ensemble Forecast (Kronos + Nixtla deep models)
# ---------------------------------------------------------------------------

elif page == "Ensemble Forecast":
    st.title("Ensemble Price Forecast")
    st.caption("Blend of the Kronos foundation model and the Nixtla deep models "
               "(N-HiTS, PatchTST, TFT), weighted by walk-forward backtest accuracy. "
               "The shaded band is the ensemble's quantile range (model disagreement + uncertainty).")

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    ensemble_files = sorted(PORTFOLIO_DIR.glob("ensemble_forecast_*.csv"))
    if not ensemble_files:
        st.info(
            "No ensemble forecast found. Run `scripts/ensemble_forecast.py` "
            "(or `scripts/refresh_all.py --with-ensemble`) to generate one.\n\n"
            f"Expected location: `{PORTFOLIO_DIR}/ensemble_forecast_YYYY-MM-DD.csv`"
        )
        st.stop()

    df_e = pd.read_csv(selected_ens_file, parse_dates=["forecast_date"])
    forecast_date = df_e["forecast_date"].iloc[0].date()

    # --- Summary: final-day upside per ticker, per model (ensemble first) ---
    last_day = df_e.groupby(["ticker", "model"])["pred_day"].transform("max")
    finals = df_e[df_e["pred_day"] == last_day]
    pivot = finals.pivot_table(index="ticker", columns="model",
                               values="upside_pct", aggfunc="first")
    name_map = df_e.groupby("ticker")["name"].first()
    last_map = df_e.groupby("ticker")["last_close"].first()
    pivot = pivot.reset_index()
    pivot.insert(1, "name", pivot["ticker"].map(name_map))
    pivot.insert(2, "last_close", pivot["ticker"].map(last_map))
    if "Ensemble" in pivot.columns:
        pivot = pivot.sort_values("Ensemble", ascending=False).reset_index(drop=True)

    st.subheader(f"21-day Outlook — {forecast_date}")

    # Metric cards (top movers by ensemble upside)
    N_CARDS = 8
    card_rows = pivot.head(min(N_CARDS, len(pivot)))
    cols = st.columns(len(card_rows))
    for col, (_, row) in zip(cols, card_rows.iterrows()):
        up = row.get("Ensemble", float("nan"))
        col.metric(label=row["ticker"],
                   value=f"${row['last_close']:.2f}",
                   delta=f"{up:+.1f}%" if pd.notna(up) else "—")
    if len(pivot) > N_CARDS:
        st.caption(f"Showing top {N_CARDS} of {len(pivot)} assets by ensemble 21-day upside.")

    st.divider()

    # Per-model upside table (%), ensemble first
    model_cols = [c for c in ["Ensemble", "NHITS", "PatchTST", "TFT",
                              "Kronos-base", "Kronos-small", "Kronos-mini"]
                  if c in pivot.columns]
    show = pivot[["ticker", "name", "last_close"] + model_cols].copy()
    show.columns = ["Ticker", "Name", "Last"] + [f"{m} %" for m in model_cols]

    def _color_pct(val):
        if isinstance(val, float) and pd.notna(val):
            return f"color: {'#26a69a' if val >= 0 else '#ef5350'}"
        return ""

    fmt = {"Last": "${:.2f}"}
    fmt.update({f"{m} %": "{:+.1f}%" for m in model_cols})
    styled = show.style.format(fmt, na_rep="—").map(
        _color_pct, subset=[f"{m} %" for m in model_cols])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.download_button("Download forecast CSV", df_e.to_csv(index=False).encode(),
                       file_name=selected_ens_file.name, mime="text/csv")

    st.divider()

    # --- Per-ticker forecast chart: ensemble band + member model lines ---
    st.subheader("Forecast Path")
    sel = st.selectbox("Select asset", pivot["ticker"].tolist(),
                       format_func=lambda t: f"{t} — {name_map.get(t, '')}")

    g = df_e[df_e["ticker"] == sel].copy()
    last_close = float(g["last_close"].iloc[0])
    g_ens = g[g["model"] == "Ensemble"].sort_values("pred_day")

    fig = go.Figure()
    if not g_ens.empty and g_ens["pred_high"].notna().any():
        fig.add_trace(go.Scatter(
            x=list(g_ens["pred_day"]) + list(g_ens["pred_day"][::-1]),
            y=list(g_ens["pred_high"]) + list(g_ens["pred_low"][::-1]),
            fill="toself", fillcolor="rgba(99,110,250,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="Ensemble band", hoverinfo="skip"))
    for m in [x for x in g["model"].unique() if x != "Ensemble"]:
        gm = g[g["model"] == m].sort_values("pred_day")
        fig.add_trace(go.Scatter(x=gm["pred_day"], y=gm["pred_close"], mode="lines",
                                 line=dict(width=1, dash="dot"), name=m, opacity=0.6))
    if not g_ens.empty:
        fig.add_trace(go.Scatter(x=g_ens["pred_day"], y=g_ens["pred_close"], mode="lines+markers",
                                 line=dict(width=3, color="#636efa"), name="Ensemble"))
    fig.add_hline(y=last_close, line_dash="dash", line_color="gray",
                  annotation_text=f"last ${last_close:.2f}")
    fig.update_layout(height=460, xaxis_title="Trading days ahead",
                      yaxis_title="Predicted close (USD)",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 6 — Backtest Results (walk-forward model comparison)
# ---------------------------------------------------------------------------

elif page == "Backtest Results":
    st.title("Forecast Model Backtest")
    st.caption(
        "Walk-forward out-of-sample benchmark comparing Kronos, N-HiTS, PatchTST, TFT, "
        "and their equal-weight Ensemble. Each window retrains neural models on the prior "
        "~400 days and scores the next 21-day horizon. No look-ahead bias."
    )

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    bench_files = sorted(PORTFOLIO_DIR.glob("forecast_benchmark_*.csv"))
    if not bench_files:
        st.info(
            "No benchmark results found. Run `scripts/forecast_benchmark.py` to generate one.\n\n"
            f"Expected location: `{PORTFOLIO_DIR}/forecast_benchmark_YYYY-MM-DD.csv`\n\n"
            "Example: `python scripts/forecast_benchmark.py` (uses the portfolio tickers by default)"
        )
        st.stop()

    df_b = pd.read_csv(selected_bench_file)
    run_date = selected_bench_file.stem.replace("forecast_benchmark_", "")

    # ── Summary leaderboard (mean across all tickers) ─────────────────────────
    st.subheader(f"Leaderboard — {run_date}")

    MODEL_ORDER = ["Ensemble", "TFT", "PatchTST", "NHITS", "Kronos-base",
                   "Kronos-small", "Kronos-mini"]
    present_models = [m for m in MODEL_ORDER if m in df_b["model"].unique()]
    other_models = [m for m in df_b["model"].unique() if m not in MODEL_ORDER]
    all_models = present_models + other_models

    agg = (df_b.groupby("model")
               .agg(RMSE=("rmse", "mean"), MAE=("mae", "mean"),
                    MAPE=("mape", "mean"), DirAcc=("dir_acc", "mean"),
                    Tickers=("ticker", "nunique"), Windows=("n_windows", "sum"))
               .reindex(all_models).dropna(how="all").reset_index())
    agg.rename(columns={"model": "Model"}, inplace=True)
    agg = agg.sort_values("MAPE")

    # Metric cards for top-3
    cols3 = st.columns(min(3, len(agg)))
    medals = ["🥇", "🥈", "🥉"]
    for i, (col, (_, row)) in enumerate(zip(cols3, agg.head(3).iterrows())):
        col.metric(
            label=f"{medals[i]} {row['Model']}",
            value=f"MAPE {row['MAPE']:.2f}%",
            delta=f"DirAcc {row['DirAcc']:.1f}%",
            delta_color="normal",
        )

    st.divider()

    # Full leaderboard table
    def _color_mape(val):
        if pd.isna(val):
            return ""
        mn, mx = agg["MAPE"].min(), agg["MAPE"].max()
        ratio = (val - mn) / max(mx - mn, 1e-9)
        r = int(239 * ratio + 38 * (1 - ratio))
        g = int(83 * ratio + 166 * (1 - ratio))
        b = int(80 * ratio + 154 * (1 - ratio))
        return f"color: rgb({r},{g},{b})"

    def _color_dir(val):
        if pd.isna(val):
            return ""
        return f"color: {'#26a69a' if val >= 50 else '#ef5350'}"

    styled_agg = (
        agg.style
        .format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE": "{:.2f}%",
                 "DirAcc": "{:.1f}%", "Tickers": "{:.0f}", "Windows": "{:.0f}"}, na_rep="—")
        .map(_color_mape, subset=["MAPE"])
        .map(_color_dir, subset=["DirAcc"])
    )
    st.dataframe(styled_agg, use_container_width=True, hide_index=True)

    st.divider()

    # ── Bar chart: MAPE by model ──────────────────────────────────────────────
    st.subheader("MAPE by Model (lower is better)")
    colors = {"Ensemble": "#636efa", "TFT": "#ef553b", "PatchTST": "#00cc96",
              "NHITS": "#ab63fa", "Kronos-base": "#ffa15a",
              "Kronos-small": "#19d3f3", "Kronos-mini": "#ff6692"}

    fig_bar = go.Figure()
    for _, row in agg.iterrows():
        fig_bar.add_trace(go.Bar(
            x=[row["Model"]], y=[row["MAPE"]],
            name=row["Model"],
            marker_color=colors.get(row["Model"], "#8c8c8c"),
            text=[f"{row['MAPE']:.2f}%"],
            textposition="outside",
        ))
    fig_bar.update_layout(
        height=380, showlegend=False,
        yaxis_title="Mean MAPE % (across tickers)",
        xaxis_title="Model",
        yaxis=dict(range=[0, agg["MAPE"].max() * 1.25]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Directional accuracy bar chart ────────────────────────────────────────
    st.subheader("Directional Accuracy by Model")
    fig_dir = go.Figure()
    for _, row in agg.iterrows():
        fig_dir.add_trace(go.Bar(
            x=[row["Model"]], y=[row["DirAcc"]],
            name=row["Model"],
            marker_color=colors.get(row["Model"], "#8c8c8c"),
            text=[f"{row['DirAcc']:.1f}%"],
            textposition="outside",
        ))
    fig_dir.add_hline(y=50, line_dash="dash", line_color="gray",
                      annotation_text="50% (random)")
    fig_dir.update_layout(
        height=380, showlegend=False,
        yaxis_title="Directional accuracy %",
        xaxis_title="Model",
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig_dir, use_container_width=True)

    st.divider()

    # ── Per-ticker breakdown ──────────────────────────────────────────────────
    st.subheader("Per-Ticker Detail")

    metric_choice = st.radio("Metric", ["MAPE", "MAE", "RMSE", "DirAcc"],
                             horizontal=True, index=0)
    tickers_in = sorted(df_b["ticker"].unique())
    ticker_sel = st.multiselect("Filter tickers", tickers_in, default=tickers_in)

    sub_b = df_b[df_b["ticker"].isin(ticker_sel)].copy()

    fig_heat = go.Figure(data=go.Heatmap(
        z=sub_b.pivot_table(index="ticker", columns="model",
                            values=metric_choice.lower(), aggfunc="mean")
                .reindex(columns=all_models).values,
        x=[m for m in all_models if m in sub_b["model"].unique()],
        y=sub_b.pivot_table(index="ticker", columns="model",
                            values=metric_choice.lower(), aggfunc="mean")
                .reindex(columns=all_models).index.tolist(),
        colorscale="RdYlGn_r" if metric_choice != "DirAcc" else "RdYlGn",
        text=sub_b.pivot_table(index="ticker", columns="model",
                               values=metric_choice.lower(), aggfunc="mean")
                   .reindex(columns=all_models).values.round(2),
        texttemplate="%{text}",
        showscale=True,
    ))
    fig_heat.update_layout(
        height=max(350, 30 * len(ticker_sel) + 100),
        xaxis_title="Model", yaxis_title="Ticker",
        title=f"{metric_choice} heatmap — lower is better"
               if metric_choice != "DirAcc" else "Directional Accuracy heatmap — higher is better",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.download_button(
        "Download benchmark CSV", df_b.to_csv(index=False).encode(),
        file_name=selected_bench_file.name, mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Page 7 — Technical Signals (Murphy indicator toolkit + weight-of-evidence)
# ---------------------------------------------------------------------------

elif page == "Technical Signals":
    st.title("Murphy Technical Signals")
    st.caption(
        "Weight-of-evidence scoring from John J. Murphy's *Technical Analysis of the "
        "Financial Markets*. Ten rules vote +1/–1/0, split into trend-following "
        "(MA cross, MA alignment, ADX/DI, MACD, OBV, Donchian breakout) and oscillators "
        "(RSI, Stochastic, Williams %R, ROC). Generate with `scripts/technical_signals.py`."
    )

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    sig_files = sorted(PORTFOLIO_DIR.glob("technical_signals_*.csv"))
    if not sig_files:
        st.info(
            "No technical-signal run found. Generate one with "
            "`python scripts/technical_signals.py --no-fetch` "
            "(add `--portfolio` for just your holdings).\n\n"
            f"Expected: `{PORTFOLIO_DIR}/technical_signals_YYYY-MM-DD.csv`"
        )
        st.stop()

    selected_sig_file = st.selectbox(
        "Signal date", sig_files, index=len(sig_files) - 1,
        format_func=lambda p: p.stem.replace("technical_signals_", ""),
    )
    df_s = pd.read_csv(selected_sig_file)

    # --- Rating distribution cards -----------------------------------------
    RATINGS = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
    counts = df_s["rating"].value_counts()
    cols = st.columns(len(RATINGS))
    for col, r in zip(cols, RATINGS):
        col.metric(r.title(), int(counts.get(r, 0)))

    st.divider()

    # --- Signal table with rating + per-rule votes -------------------------
    sig_cols = [f"sig_{k}" for k in (*TREND_SIGNALS, *OSC_SIGNALS)]
    # Distinct labels — must not collide with the raw "ADX"/"RSI" value columns,
    # or pandas Styler fails on non-unique columns.
    pretty = {
        "sig_ma_cross": "GC/DC", "sig_ma_align": "MA≡", "sig_adx_di": "DI",
        "sig_macd": "MACD", "sig_obv": "OBV", "sig_donchian": "Donch",
        "sig_rsi": "RSI±", "sig_stoch": "Stoch", "sig_williams": "%R", "sig_roc": "ROC",
    }

    view_cols = ["ticker", "name", "last_close", "rating", "murphy_score",
                 "trend_score", "osc_score", "adx", "rsi"] + sig_cols
    show = df_s[[c for c in view_cols if c in df_s.columns]].copy()
    show = show.rename(columns={
        "ticker": "Ticker", "name": "Name", "last_close": "Last",
        "rating": "Rating", "murphy_score": "Score",
        "trend_score": "Trend", "osc_score": "Osc",
        "adx": "ADX", "rsi": "RSI", **pretty,
    })

    def _color_rating(val):
        colors = {"STRONG BUY": "#1b7f3b", "BUY": "#26a69a", "HOLD": "#9e9e9e",
                  "SELL": "#ef5350", "STRONG SELL": "#b71c1c"}
        return f"color: {colors.get(val, '')}; font-weight: 600"

    def _color_vote(val):
        if val == 1:
            return "color: #26a69a"
        if val == -1:
            return "color: #ef5350"
        return "color: #6b6b6b"

    def _color_score(val):
        if isinstance(val, (int, float)) and pd.notna(val):
            return f"color: {'#26a69a' if val > 0 else '#ef5350' if val < 0 else '#9e9e9e'}"
        return ""

    vote_labels = [pretty[c] for c in sig_cols]
    fmt = {"Last": "${:.2f}", "ADX": "{:.0f}", "RSI": "{:.0f}"}
    fmt.update({lbl: "{:+d}" for lbl in vote_labels})
    styled = (
        show.style.format(fmt, na_rep="—")
        .map(_color_rating, subset=["Rating"])
        .map(_color_score, subset=["Score", "Trend", "Osc"])
        .map(_color_vote, subset=vote_labels)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=560)
    st.download_button("Download signals CSV", df_s.to_csv(index=False).encode(),
                       file_name=selected_sig_file.name, mime="text/csv")

    st.caption("Vote legend: +1 bullish · −1 bearish · 0 neutral. "
               "Trend tools lead in trending markets (ADX ≥ 25); oscillators lead in ranges.")

    st.divider()

    # --- Per-ticker indicator detail chart ---------------------------------
    st.subheader("Indicator Detail")
    sel = st.selectbox("Select asset", df_s["ticker"].tolist(),
                       format_func=lambda t: f"{t} — "
                       f"{df_s.loc[df_s['ticker'] == t, 'name'].iloc[0]}")
    row = df_s[df_s["ticker"] == sel].iloc[0]

    df_ohlc = load_ohlcv(sel)
    if not df_ohlc.empty:
        from src.screening.murphy import murphy_indicators
        ind = murphy_indicators(df_ohlc).tail(252)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.04,
            subplot_titles=("Price · SMA50 · SMA200 · Donchian", "ADX / +DI / −DI", "RSI · Stochastic"),
        )
        # Price panel
        fig.add_trace(go.Scatter(x=ind["Date"], y=ind["Close"], name="Close",
                                 line=dict(color="#e0e0e0", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ind["Date"], y=ind["SMA50"], name="SMA50",
                                 line=dict(color="#42a5f5", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ind["Date"], y=ind["SMA200"], name="SMA200",
                                 line=dict(color="#ffa726", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ind["Date"], y=ind["donchian_high"], name="Donchian hi",
                                 line=dict(color="#66bb6a", width=0.7, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ind["Date"], y=ind["donchian_low"], name="Donchian lo",
                                 line=dict(color="#ef5350", width=0.7, dash="dot")), row=1, col=1)
        # ADX / DI panel
        fig.add_trace(go.Scatter(x=ind["Date"], y=ind["ADX"], name="ADX",
                                 line=dict(color="#ab47bc", width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=ind["Date"], y=ind["plus_di"], name="+DI",
                                 line=dict(color="#26a69a", width=0.9)), row=2, col=1)
        fig.add_trace(go.Scatter(x=ind["Date"], y=ind["minus_di"], name="−DI",
                                 line=dict(color="#ef5350", width=0.9)), row=2, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="gray", row=2, col=1)
        # RSI / Stochastic panel
        fig.add_trace(go.Scatter(x=ind["Date"], y=ind["RSI"], name="RSI",
                                 line=dict(color="#29b6f6", width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ind["Date"], y=ind["stoch_k"], name="Stoch %K",
                                 line=dict(color="#ffca28", width=0.9)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="gray", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", row=3, col=1)

        fig.update_layout(height=680, hovermode="x unified",
                          legend=dict(orientation="h", y=1.04, yanchor="bottom"),
                          margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"**{sel}** — rating **{row['rating']}** "
        f"(score {int(row['murphy_score'])}: trend {int(row['trend_score'])}, "
        f"oscillator {int(row['osc_score'])}). "
        f"ADX {row['adx']:.0f}, RSI {row['rsi']:.0f}, "
        f"+DI {row['plus_di']:.0f} / −DI {row['minus_di']:.0f}."
    )
