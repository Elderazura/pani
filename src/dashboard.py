"""Streamlit monitoring dashboard for the futures trading system.

Run with: streamlit run src/dashboard.py
(From project root: cd futures-intel && PYTHONPATH=. streamlit run src/dashboard.py)
"""

import asyncio
from datetime import date, datetime, timedelta

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .config import get_settings
from .db import db
from .engine import TradingEngine
from .backtest import Backtester

# Dark theme + green/red accents
CHART_TEMPLATE = "plotly_dark"
BULLISH = "#22c55e"
BEARISH = "#ef4444"
PENDING = "#eab308"
NEUTRAL = "#6b7280"

st.set_page_config(page_title="Futures Intel", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(ttl=60)
def _check_binance() -> bool:
    try:
        import ccxt
        ex = ccxt.binanceusdm({"options": {"sandbox": get_settings().BINANCE_TESTNET}})
        ex.fetch_time()
        return True
    except Exception:
        return False


@st.cache_data(ttl=60)
def _check_ollama() -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{get_settings().OLLAMA_URL.rstrip('/')}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=60)
def _check_duckdb() -> bool:
    try:
        with db.connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@st.cache_data(ttl=30)
def _get_proposals_cached(status: str = "all") -> list[dict]:
    return db.get_proposals(status)


@st.cache_data(ttl=30)
def _get_scans_cached(limit: int = 50) -> list[dict]:
    return db.get_latest_scans(limit)


@st.cache_data(ttl=30)
def _get_signals_cached(limit: int = 50) -> list[dict]:
    return db.get_latest_signals(limit)


def _status_dot(ok: bool) -> str:
    return "🟢" if ok else "🔴"


def _sidebar():
    st.sidebar.title("Futures Intel")
    st.sidebar.markdown("---")

    binance_ok = _check_binance()
    ollama_ok = _check_ollama()
    duckdb_ok = _check_duckdb()

    st.sidebar.markdown("### System Status")
    st.sidebar.markdown(f"{_status_dot(binance_ok)} Binance")
    st.sidebar.markdown(f"{_status_dot(ollama_ok)} Ollama")
    st.sidebar.markdown(f"{_status_dot(duckdb_ok)} DuckDB")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Filters")
    date_range = st.sidebar.date_input(
        "Date range",
        value=(date.today() - timedelta(days=7), date.today()),
        max_value=date.today(),
    )
    if not isinstance(date_range, (tuple, list)):
        date_range = (date_range, date_range)
    symbols_raw = _get_proposals_cached("all")
    all_symbols = sorted(set(p.get("symbol", "") for p in symbols_raw if p.get("symbol")))
    symbol_filter = st.sidebar.selectbox("Symbol", ["All"] + all_symbols)
    status_filter = st.sidebar.selectbox("Status", ["all", "pending", "approved", "rejected", "expired"])

    st.sidebar.markdown("---")
    if st.sidebar.button("Run Pipeline Now", use_container_width=True):
        with st.spinner("Running pipeline..."):
            async def _run():
                engine = TradingEngine()
                await engine.run_pipeline()
                await engine.shutdown()
            asyncio.run(_run())
            st.sidebar.success("Pipeline complete!")
            st.cache_data.clear()

    if st.sidebar.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    return date_range, symbol_filter, status_filter


def _prop_date(p: dict) -> date:
    ct = p.get("created_at")
    return ct.date() if ct and hasattr(ct, "date") else date.min


def _tab_overview(date_range, symbol_filter, status_filter):
    all_proposals = _get_proposals_cached(status_filter)
    proposals = all_proposals
    if symbol_filter != "All":
        proposals = [p for p in proposals if p.get("symbol") == symbol_filter]
    if isinstance(date_range, (tuple, list)) and len(date_range) >= 2:
        start_d, end_d = date_range[0], date_range[1]
        proposals = [p for p in proposals if start_d <= _prop_date(p) <= end_d]

    today_proposals = [p for p in _get_proposals_cached("all") if _prop_date(p) == date.today()]
    today_pnl = db.get_daily_pnl(date.today())
    signals = _get_signals_cached(20)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Proposals Today", len(today_proposals))
    with col2:
        win_rate = 0.0
        if "backtest_result" in st.session_state:
            win_rate = st.session_state["backtest_result"].win_rate
        st.metric("Win Rate (Backtest)", f"{win_rate:.1%}")
    with col3:
        st.metric("Active Signals", len(signals))
    with col4:
        pnl_val = today_pnl["realized_pnl"] if today_pnl else 0.0
        st.metric("Portfolio PnL", f"{pnl_val:.2f}%", delta=None)

    if "equity_curve" in st.session_state and st.session_state["equity_curve"]:
        eq = st.session_state["equity_curve"]
        fig = go.Figure(data=[go.Scatter(y=eq, mode="lines", line=dict(color=BULLISH, width=2))])
        fig.update_layout(template=CHART_TEMPLATE, title="Equity Curve", height=300, margin=dict(l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    latest = proposals[:10]
    if latest:
        df = pd.DataFrame(latest)
        cols = ["symbol", "action", "entry_price", "stop_loss", "take_profit", "status", "created_at"]
        display_cols = [c for c in cols if c in df.columns]
        if display_cols:
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


def _tab_live_scanner():
    scans = _get_scans_cached(30)
    signals = _get_signals_cached(30)

    if not scans:
        st.info("No scan data. Run the pipeline or backfill first.")
        return

    df = pd.DataFrame(scans)
    scan_cols = ["symbol", "volume_24h", "price", "funding_rate", "open_interest"]
    display_cols = [c for c in scan_cols if c in df.columns]
    if display_cols:
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    for s in signals[:10]:
        with st.expander(f"{s['symbol']} — {s.get('signal_type', 'N/A')} (conf: {s.get('confidence', 0):.2f})"):
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi = s.get("rsi", 50)
                st.metric("RSI", f"{rsi:.1f}")
                st.progress(min(max((rsi - 30) / 40, 0), 1))
            with col2:
                macd = s.get("macd_histogram", 0)
                st.metric("MACD Hist", f"{macd:.4f}")
            with col3:
                sent = 0.0
                props = [p for p in _get_proposals_cached("all") if p.get("symbol") == s["symbol"]]
                if props and props[0].get("sentiment_score") is not None:
                    sent = props[0]["sentiment_score"]
                st.metric("Sentiment", f"{sent:.2f}")
                st.progress((sent + 1) / 2)


def _tab_backtest_lab():
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Symbol", "BTC/USDT:USDT")
        days = st.number_input("Days", value=90, min_value=7, max_value=365)
        sl = st.number_input("Stop Loss %", value=1.5, min_value=0.5, step=0.5)
        tp = st.number_input("Take Profit %", value=4.5, min_value=1.0, step=0.5)
        leverage = st.number_input("Leverage", value=3, min_value=1, max_value=20)

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            bt = Backtester()
            end = datetime.utcnow()
            start = end - timedelta(days=int(days))
            result = bt.run(
                symbols=[symbol],
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                stop_loss_pct=sl,
                take_profit_pct=tp,
                leverage=int(leverage),
            )
            trades, equity_curve, _ = bt.get_last_run_details()
            st.session_state["backtest_result"] = result
            st.session_state["backtest_trades"] = trades
            st.session_state["equity_curve"] = equity_curve

    if "backtest_result" in st.session_state:
        r = st.session_state["backtest_result"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trades", r.total_trades)
        c2.metric("Win Rate", f"{r.win_rate:.1%}")
        c3.metric("Sharpe", f"{r.sharpe_ratio:.2f}")
        c4.metric("Max DD", f"{r.max_drawdown_pct:.1f}%")
        c5.metric("Return", f"{r.total_return_pct:.1f}%")

        if st.session_state.get("equity_curve"):
            fig = go.Figure(data=[go.Scatter(y=st.session_state["equity_curve"], mode="lines", line=dict(color=BULLISH))])
            fig.update_layout(template=CHART_TEMPLATE, title="Equity Curve", height=300)
            st.plotly_chart(fig, use_container_width=True)

        eq_curve = st.session_state.get("equity_curve") or []
        peak = eq_curve[0] if eq_curve else 1
        dd = []
        for eq in eq_curve:
            peak = max(peak, eq)
            dd.append((peak - eq) / peak * 100 if peak > 0 else 0)
        fig_dd = go.Figure(data=[go.Scatter(y=dd, mode="lines", line=dict(color=BEARISH))])
        fig_dd.update_layout(template=CHART_TEMPLATE, title="Drawdown %", height=200)
        st.plotly_chart(fig_dd, use_container_width=True)

        if st.session_state.get("backtest_trades"):
            st.dataframe(pd.DataFrame(st.session_state["backtest_trades"]), use_container_width=True, hide_index=True)

    if st.button("Run Optimization"):
        with st.spinner("Optimizing..."):
            bt = Backtester()
            end = datetime.utcnow()
            start = end - timedelta(days=180)
            opt_df = bt.optimize(
                symbol,
                param_grid={"stop_loss_pct": [1.0, 1.5, 2.0], "take_profit_pct": [3.0, 4.5, 6.0], "leverage": [2, 3]},
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            )
            if not opt_df.empty:
                st.session_state["optimize_df"] = opt_df
                pivot = opt_df.pivot_table(
                    values="sharpe_ratio", index="take_profit_pct", columns="stop_loss_pct", aggfunc="mean"
                )
                fig_heat = px.imshow(
                    pivot, color_continuous_scale="RdYlGn", title="Sharpe by SL/TP",
                    labels=dict(x="Stop Loss %", y="Take Profit %", color="Sharpe")
                )
                fig_heat.update_layout(template=CHART_TEMPLATE)
                st.plotly_chart(fig_heat, use_container_width=True)
                st.dataframe(opt_df, use_container_width=True, hide_index=True)


def _tab_trade_log(date_range, symbol_filter, status_filter):
    proposals = _get_proposals_cached(status_filter)
    if symbol_filter != "All":
        proposals = [p for p in proposals if p.get("symbol") == symbol_filter]
    if isinstance(date_range, (tuple, list)) and len(date_range) >= 2:
        start_d, end_d = date_range[0], date_range[1]
        def _proposal_date(p):
            ct = p.get("created_at")
            return ct.date() if ct and hasattr(ct, "date") else date.today()
        proposals = [p for p in proposals if start_d <= _proposal_date(p) <= end_d]

    if proposals:
        df = pd.DataFrame(proposals)
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False)
        st.download_button("Export to CSV", csv, file_name="proposals.csv", mime="text/csv")
    else:
        st.info("No proposals match filters.")


def main():
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; }
        </style>
    """, unsafe_allow_html=True)

    date_range, symbol_filter, status_filter = _sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Live Scanner", "Backtest Lab", "Trade Log"])

    with tab1:
        _tab_overview(date_range, symbol_filter, status_filter)
    with tab2:
        _tab_live_scanner()
    with tab3:
        _tab_backtest_lab()
    with tab4:
        _tab_trade_log(date_range, symbol_filter, status_filter)


if __name__ == "__main__":
    main()
