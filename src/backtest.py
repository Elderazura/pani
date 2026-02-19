"""Backtesting engine using vectorbt for strategy validation."""

import itertools
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import vectorbt as vbt
except ImportError:
    vbt = None
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings
from .db import db
from .models import BacktestResult
from .pipeline.analyst import QuantAnalyst

TAKER_FEE_PCT = 0.06
ROUND_TRIP_FEE_PCT = TAKER_FEE_PCT * 2
HOURLY_PERIODS_PER_YEAR = 365 * 24


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Sort OHLCV ascending by time, ensure required columns."""
    if df is None or df.empty:
        return df
    out = df.copy()
    time_col = "bar_time" if "bar_time" in out.columns else "timestamp"
    if time_col in out.columns:
        out = out.sort_values(time_col).reset_index(drop=True)
    return out


class Backtester:
    """Backtests the full strategy pipeline on historical data."""

    def __init__(self, db_instance=None):
        """Initialize with database and analyst."""
        self.db = db_instance if db_instance is not None else db
        self.analyst = QuantAnalyst(get_settings())

    def run(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        start_date: str | None = None,
        end_date: str | None = None,
        stop_loss_pct: float = 1.5,
        take_profit_pct: float = 4.5,
        leverage: int = 3,
        min_confidence: float = 0.6,
        initial_capital: float = 10_000,
    ) -> BacktestResult:
        """
        Run full backtest across multiple symbols.
        Uses rolling 100-bar window, analyst.analyze(), SL/TP, leverage, fees.
        """
        all_trades: list[dict[str, Any]] = []
        start_dt: datetime | None = None
        end_dt: datetime | None = None

        for symbol in symbols:
            df = self.db.get_ohlcv(symbol, timeframe, limit=20_000)
            df = _prepare_ohlcv(df)
            if df is None or len(df) < 101:
                logger.warning("{}: insufficient data ({} rows), skipping", symbol, len(df) if df is not None else 0)
                continue

            if start_date or end_date:
                time_col = "bar_time" if "bar_time" in df.columns else "timestamp"
                df[time_col] = pd.to_datetime(df[time_col])
                if start_date:
                    start = pd.Timestamp(start_date)
                    df = df[df[time_col] >= start]
                if end_date:
                    end = pd.Timestamp(end_date)
                    df = df[df[time_col] <= end]
                if df.empty or len(df) < 101:
                    continue

            if df.empty:
                continue
            time_col = "bar_time" if "bar_time" in df.columns else "timestamp"
            sym_start = df[time_col].min()
            sym_end = df[time_col].max()
            if start_dt is None or sym_start < start_dt:
                start_dt = sym_start
            if end_dt is None or sym_end > end_dt:
                end_dt = sym_end

            in_position = False
            entry_bar = 0
            entry_price = 0.0
            stop_loss = 0.0
            take_profit = 0.0
            pos_leverage = 1

            i = 99
            while i < len(df) - 1:
                if in_position:
                    bar = df.iloc[i]
                    low = float(bar["low"])
                    high = float(bar["high"])
                    exit_price: float | None = None
                    exit_reason = ""

                    if low <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "SL"
                    elif high >= take_profit:
                        exit_price = take_profit
                        exit_reason = "TP"

                    if exit_price is not None:
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                        pnl_pct *= pos_leverage
                        pnl_pct -= ROUND_TRIP_FEE_PCT
                        bars_held = i - entry_bar
                        all_trades.append({
                            "symbol": symbol,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl_pct": pnl_pct,
                            "bars_held": bars_held,
                            "win": pnl_pct > 0,
                            "exit_reason": exit_reason,
                        })
                        in_position = False

                if not in_position:
                    window = df.iloc[i - 99 : i + 1].copy()
                    signal = self.analyst.analyze(symbol, window, timeframe=timeframe)
                    if signal and signal.confidence >= min_confidence:
                        next_bar = df.iloc[i + 1]
                        entry_price = float(next_bar["open"])
                        stop_loss = entry_price * (1 - stop_loss_pct / 100)
                        take_profit = entry_price * (1 + take_profit_pct / 100)
                        entry_bar = i + 1
                        pos_leverage = leverage
                        in_position = True
                        i += 1
                        continue

                i += 1

        if not all_trades:
            logger.warning("No trades generated")
            return self._calculate_metrics(
                [], initial_capital,
                start_dt or datetime.utcnow(),
                end_dt or datetime.utcnow(),
                {"stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct, "leverage": leverage},
            )

        result = self._calculate_metrics(
            all_trades,
            initial_capital,
            start_dt or datetime.utcnow(),
            end_dt or datetime.utcnow(),
            {"stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct, "leverage": leverage},
        )
        self._last_trades = all_trades
        self._last_equity_curve = self._build_equity_from_trades(all_trades, initial_capital)
        self._last_start_dt = start_dt or datetime.utcnow()
        return result

    def _build_equity_from_trades(
        self, trades: list[dict[str, Any]], initial_capital: float
    ) -> list[float]:
        equity = initial_capital
        curve = [equity]
        for t in trades:
            equity *= 1 + t["pnl_pct"] / 100
            curve.append(equity)
        return curve

    def get_last_run_details(self) -> tuple[list[dict], list[float], datetime | None]:
        """Return (trades, equity_curve, start_dt) from last run()."""
        return (
            getattr(self, "_last_trades", []) or [],
            getattr(self, "_last_equity_curve", []) or [],
            getattr(self, "_last_start_dt", None),
        )

    def _calculate_metrics(
        self,
        trades: list[dict[str, Any]],
        initial_capital: float = 10_000,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        params: dict[str, Any] | None = None,
    ) -> BacktestResult:
        """
        From list of trade dicts compute metrics and return BacktestResult.
        """
        params = params or {}
        start_date = start_date or datetime.utcnow()
        end_date = end_date or datetime.utcnow()

        if not trades:
            return BacktestResult(
                strategy_name="pipeline",
                params=params,
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                total_return_pct=0.0,
                avg_trade_duration_minutes=0.0,
                start_date=start_date,
                end_date=end_date,
            )

        total = len(trades)
        wins = sum(1 for t in trades if t.get("win", t.get("pnl_pct", 0) > 0))
        win_rate = wins / total if total else 0.0

        winning_pnl = sum(t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) > 0)
        losing_pnl = sum(t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) < 0)
        profit_factor = winning_pnl / abs(losing_pnl) if losing_pnl != 0 else (float("inf") if winning_pnl > 0 else 0.0)
        if profit_factor == float("inf"):
            profit_factor = 99.0

        equity = initial_capital
        equity_curve = [equity]
        for t in trades:
            equity *= 1 + t["pnl_pct"] / 100
            equity_curve.append(equity)

        equity_series = pd.Series(equity_curve)
        idx = pd.date_range(start=start_date, periods=len(equity_curve), freq="1h")
        equity_series.index = idx
        try:
            if vbt is not None:
                pf = vbt.Portfolio.from_holding(equity_series, init_cash=initial_capital)
                sharpe = float(pf.sharpe_ratio().iloc[-1]) if not pf.sharpe_ratio().empty else 0.0
                max_dd = float(pf.max_drawdown().iloc[-1] * 100) if not pf.max_drawdown().empty else 0.0
            else:
                raise ImportError("vectorbt not installed")
        except Exception:
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            returns = returns[~np.isnan(returns)]
            sharpe = 0.0
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(HOURLY_PERIODS_PER_YEAR))
            peak = equity_curve[0]
            max_dd = 0.0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

        total_return_pct = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100 if equity_curve else 0.0
        avg_bars = np.mean([t["bars_held"] for t in trades])
        avg_trade_duration_minutes = float(avg_bars * 60)

        return BacktestResult(
            strategy_name="pipeline",
            params=params,
            total_trades=total,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            total_return_pct=total_return_pct,
            avg_trade_duration_minutes=avg_trade_duration_minutes,
            start_date=start_date,
            end_date=end_date,
        )

    def optimize(
        self,
        symbol: str,
        param_grid: dict[str, list[Any]],
        timeframe: str = "1h",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Grid search over parameter combinations.
        Returns DataFrame sorted by Sharpe ratio descending.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        results: list[dict[str, Any]] = []
        for combo in combos:
            params = dict(zip(keys, combo))
            sl = params.get("stop_loss_pct", 1.5)
            tp = params.get("take_profit_pct", 4.5)
            lev = params.get("leverage", 3)
            r = self.run(
                symbols=[symbol],
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                stop_loss_pct=sl,
                take_profit_pct=tp,
                leverage=lev,
            )
            results.append({
                **params,
                "sharpe_ratio": r.sharpe_ratio,
                "total_return_pct": r.total_return_pct,
                "win_rate": r.win_rate,
                "total_trades": r.total_trades,
                "max_drawdown_pct": r.max_drawdown_pct,
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
        return df

    def walk_forward(
        self,
        symbol: str,
        train_days: int = 60,
        test_days: int = 30,
        timeframe: str = "1h",
        param_grid: dict[str, list[Any]] | None = None,
    ) -> list[BacktestResult]:
        """
        Walk-forward optimization: optimize on train, test on out-of-sample.
        Returns list of out-of-sample BacktestResults.
        """
        param_grid = param_grid or {
            "stop_loss_pct": [1.0, 1.5, 2.0],
            "take_profit_pct": [3.0, 4.5, 6.0],
            "leverage": [2, 3],
        }

        df = self.db.get_ohlcv(symbol, timeframe, limit=20_000)
        df = _prepare_ohlcv(df)
        if df is None or len(df) < 200:
            logger.warning("Insufficient data for walk-forward")
            return []

        time_col = "bar_time" if "bar_time" in df.columns else "timestamp"
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

        start = df[time_col].min()
        results: list[BacktestResult] = []

        while True:
            train_end = start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)
            train_df = df[(df[time_col] >= start) & (df[time_col] < train_end)]
            test_df = df[(df[time_col] >= train_end) & (df[time_col] < test_end)]

            if len(train_df) < 100 or len(test_df) < 50:
                break

            train_start_str = start.strftime("%Y-%m-%d")
            train_end_str = train_end.strftime("%Y-%m-%d")
            test_start_str = train_end.strftime("%Y-%m-%d")
            test_end_str = test_end.strftime("%Y-%m-%d")

            opt_df = self.optimize(
                symbol,
                param_grid,
                timeframe=timeframe,
                start_date=train_start_str,
                end_date=train_end_str,
            )
            if opt_df.empty:
                break

            best = opt_df.iloc[0]
            sl = best.get("stop_loss_pct", 1.5)
            tp = best.get("take_profit_pct", 4.5)
            lev = best.get("leverage", 3)

            test_result = self.run(
                symbols=[symbol],
                timeframe=timeframe,
                start_date=test_start_str,
                end_date=test_end_str,
                stop_loss_pct=sl,
                take_profit_pct=tp,
                leverage=lev,
            )
            results.append(test_result)
            logger.info(
                "Walk-forward test {}: Sharpe={:.2f}, Return={:.1f}%",
                test_end_str,
                test_result.sharpe_ratio,
                test_result.total_return_pct,
            )

            start = train_end

        return results

    def report(self, result: BacktestResult) -> None:
        """Print a Rich formatted report with key metrics."""
        console = Console()
        table = Table(title="Backtest Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Strategy", result.strategy_name)
        table.add_row("Total Trades", str(result.total_trades))
        table.add_row("Win Rate", f"{result.win_rate:.1%}")
        table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
        table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        table.add_row("Max Drawdown", f"{result.max_drawdown_pct:.1f}%")
        table.add_row("Total Return", f"{result.total_return_pct:.1f}%")
        table.add_row("Avg Trade Duration", f"{result.avg_trade_duration_minutes:.0f} min")
        table.add_row("Start Date", result.start_date.strftime("%Y-%m-%d"))
        table.add_row("End Date", result.end_date.strftime("%Y-%m-%d"))
        console.print(Panel(table, title="[bold]Backtest Summary[/bold]"))
        if result.params:
            console.print("\n[bold]Parameters:[/bold]", result.params)


def _run_backtest(
    symbols: list[str],
    days: int = 180,
    sl: float = 1.5,
    tp: float = 4.5,
    leverage: int = 3,
) -> BacktestResult:
    """Helper for CLI."""
    bt = Backtester()
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    return bt.run(
        symbols=symbols,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        stop_loss_pct=sl,
        take_profit_pct=tp,
        leverage=leverage,
    )


if __name__ == "__main__":
    import typer

    app = typer.Typer(help="Backtesting engine")

    @app.command()
    def run(
        symbols: str = typer.Option("BTC/USDT:USDT", "--symbols", help="Comma-separated symbols"),
        days: int = typer.Option(180, "--days", help="Days of history"),
        sl: float = typer.Option(1.5, "--sl", help="Stop loss %"),
        tp: float = typer.Option(4.5, "--tp", help="Take profit %"),
        leverage: int = typer.Option(3, "--leverage", help="Leverage"),
    ):
        """Run backtest."""
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        result = _run_backtest(symbol_list, days, sl, tp, leverage)
        bt = Backtester()
        bt.report(result)

    @app.command("optimize")
    def optimize_cmd(
        symbol: str = typer.Option("BTC/USDT:USDT", "--symbol", help="Symbol to optimize"),
        days: int = typer.Option(180, "--days", help="Days of history"),
    ):
        """Grid search optimization."""
        bt = Backtester()
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        df = bt.optimize(
            symbol,
            param_grid={
                "stop_loss_pct": [1.0, 1.5, 2.0],
                "take_profit_pct": [3.0, 4.5, 6.0],
                "leverage": [2, 3],
            },
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
        console = Console()
        console.print(Panel(df.to_string(), title="[bold]Optimization Results (by Sharpe)[/bold]"))

    @app.command("walk-forward")
    def walk_forward_cmd(
        symbol: str = typer.Option("BTC/USDT:USDT", "--symbol", help="Symbol"),
        train_days: int = typer.Option(60, "--train-days", help="Training window days"),
        test_days: int = typer.Option(30, "--test-days", help="Test window days"),
    ):
        """Walk-forward optimization."""
        bt = Backtester()
        results = bt.walk_forward(symbol, train_days, test_days)
        console = Console()
        for i, r in enumerate(results):
            console.print(f"\n[bold]Fold {i+1}:[/bold] Sharpe={r.sharpe_ratio:.2f}, Return={r.total_return_pct:.1f}%")
        if results:
            avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
            avg_ret = sum(r.total_return_pct for r in results) / len(results)
            console.print(f"\n[bold]Average:[/bold] Sharpe={avg_sharpe:.2f}, Return={avg_ret:.1f}%")

    app()
