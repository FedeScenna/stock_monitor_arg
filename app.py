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
    ["Stock Charts", "Weekly Screen", "Portfolio Overview", "Kronos Forecast"],
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

    tickers_in_fc = df_fc["ticker"].unique().tolist()
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
    summary["up_7d"]  = (summary["pred_7d"]  / summary["last_close"] - 1) * 100
    summary["up_14d"] = (summary["pred_14d"] / summary["last_close"] - 1) * 100
    summary["up_21d"] = (summary["pred_21d"] / summary["last_close"] - 1) * 100
    summary = summary.sort_values("up_21d", ascending=False).reset_index(drop=True)

    st.subheader(f"21-day Outlook — {forecast_date}")
    st.caption("Forecast generated by Kronos-small foundation model. Treat as probabilistic signal.")

    # Colour-coded metric cards (top row)
    cols = st.columns(len(tickers_in_fc))
    for col, (_, row) in zip(cols, summary.iterrows()):
        delta_str = f"{row['up_21d']:+.1f}%"
        col.metric(
            label=row["ticker"],
            value=f"${row['pred_21d']:.2f}",
            delta=delta_str,
        )

    st.divider()

    # Summary data table
    display_summary = summary[["ticker", "name", "last_close", "pred_7d", "up_7d", "pred_14d", "up_14d", "pred_21d", "up_21d"]].copy()
    display_summary.columns = ["Ticker", "Name", "Last", "7d Price", "7d %", "14d Price", "14d %", "21d Price", "21d %"]

    def color_pct(val):
        if isinstance(val, float):
            color = "#26a69a" if val >= 0 else "#ef5350"
            return f"color: {color}"
        return ""

    styled = (
        display_summary.style
        .format({
            "Last":     "${:.2f}",
            "7d Price": "${:.2f}",
            "14d Price":"${:.2f}",
            "21d Price":"${:.2f}",
            "7d %":     "{:+.1f}%",
            "14d %":    "{:+.1f}%",
            "21d %":    "{:+.1f}%",
        })
        .applymap(color_pct, subset=["7d %", "14d %", "21d %"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # --- Per-ticker forecast charts ---
    st.subheader("Forecast Charts")
    selected_ticker = st.selectbox(
        "Select asset",
        tickers_in_fc,
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

    # Day markers: 7, 14, 21
    for day, label in [(7, "7d"), (14, "14d"), (21, "21d")]:
        if day <= len(fc_dates):
            fig_fc.add_vline(
                x=str(fc_dates[day - 1].date()),
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
