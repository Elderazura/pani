"""Main CLI entry point for the futures trading system."""

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import typer
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .backtest import Backtester
from .config import get_settings
from .db import db
from .engine import TradingEngine
from .pipeline.data_collector import DataCollector

app = typer.Typer(
    help="Futures Intel — Crypto futures trading system",
    no_args_is_help=True,
)

console = Console()


def _run_async(coro):
    """Run async coroutine from sync context."""
    return asyncio.run(coro)


def _setup_logging(verbose: bool = False) -> None:
    """Configure loguru based on verbose flag."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    dry_run: bool | None = typer.Option(None, "--dry-run", help="Override: dry run mode"),
    live_mode: bool | None = typer.Option(None, "--live", help="Override: live trading mode"),
):
    """Futures Intel CLI."""
    _setup_logging(verbose)
    if dry_run is not None or live_mode is not None:
        # Override DRY_RUN via env-like; config is loaded at import
        import os
        if dry_run is not None:
            os.environ["DRY_RUN"] = str(dry_run).lower()
        if live_mode is not None:
            os.environ["DRY_RUN"] = "false"


@app.command()
def scan():
    """Run the full pipeline once."""
    async def _scan():
        engine = TradingEngine()
        proposals = await engine.run_pipeline()
        await engine.shutdown()
        return proposals

    proposals = _run_async(_scan())
    console.print(f"\n[green]Pipeline complete. {len(proposals)} proposal(s) generated.[/green]")


@app.command()
def backfill(
    symbols: str = typer.Option("", "--symbols", help="Comma-separated symbols (omit for top 50)"),
    timeframe: str = typer.Option("1h", "--timeframe", help="OHLCV timeframe"),
    days: int = typer.Option(180, "--days", help="Days of history"),
):
    """Collect historical data for backtesting."""
    collector = DataCollector()
    if not symbols.strip():
        total = _run_async(collector.backfill_top_coins(50, timeframe, days))
        console.print(f"[green]Backfilled top 50 coins: {total} bars[/green]")
    else:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        total = _run_async(collector.backfill(symbol_list, timeframe, days))
        console.print(f"[green]Backfilled {len(symbol_list)} symbols: {total} bars[/green]")


@app.command()
def backtest(
    symbols: str = typer.Option("BTC/USDT:USDT", "--symbols", help="Comma-separated symbols"),
    days: int = typer.Option(90, "--days", help="Days of history"),
    sl: float = typer.Option(1.5, "--sl", help="Stop loss %"),
    tp: float = typer.Option(4.5, "--tp", help="Take profit %"),
):
    """Run backtester with Rich report."""
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    bt = Backtester()
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    result = bt.run(
        symbols=symbol_list,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        stop_loss_pct=sl,
        take_profit_pct=tp,
    )
    bt.report(result)


@app.command()
def optimize(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to optimize (required)"),
    days: int = typer.Option(180, "--days", help="Days of history"),
):
    """Run parameter optimization."""
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
    console.print(Panel(df.to_string(), title="[bold]Optimization Results (by Sharpe)[/bold]"))


@app.command()
def proposals(
    status: str = typer.Option("pending", "--status", help="pending|approved|rejected|expired|all"),
):
    """List proposals with optional status filter."""
    items = db.get_proposals(status)
    if not items:
        console.print(f"[yellow]No {status} proposals found.[/yellow]")
        return

    table = Table(title=f"Proposals ({status})")
    table.add_column("ID", style="dim")
    table.add_column("Symbol")
    table.add_column("Action")
    table.add_column("Entry", justify="right")
    table.add_column("SL", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("R:R", justify="right")
    table.add_column("Status")
    table.add_column("Created")

    for p in items:
        table.add_row(
            str(p.get("id", ""))[:8] + "...",
            str(p.get("symbol", "")),
            str(p.get("action", "")),
            f"{p.get('entry_price', 0):,.2f}",
            f"{p.get('stop_loss', 0):,.2f}",
            f"{p.get('take_profit', 0):,.2f}",
            f"{p.get('risk_reward', 0):.2f}",
            str(p.get("status", "")),
            str(p.get("created_at", ""))[:19] if p.get("created_at") else "",
        )
    console.print(table)


@app.command()
def approve(proposal_id: str = typer.Argument(..., help="Proposal ID to approve")):
    """Approve a proposal."""
    prop = db.get_proposal_by_id(proposal_id)
    if not prop:
        console.print(f"[red]Proposal {proposal_id} not found.[/red]")
        raise typer.Exit(1)
    db.update_proposal_status(proposal_id, "APPROVED")
    console.print("[green]Proposal approved.[/green]")
    table = Table(title="Trade Details")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Symbol", str(prop.get("symbol", "")))
    table.add_row("Action", str(prop.get("action", "")))
    table.add_row("Entry", f"{prop.get('entry_price', 0):,.2f}")
    table.add_row("Stop Loss", f"{prop.get('stop_loss', 0):,.2f}")
    table.add_row("Take Profit", f"{prop.get('take_profit', 0):,.2f}")
    table.add_row("Leverage", str(prop.get("leverage", "")))
    console.print(table)


@app.command()
def reject(proposal_id: str = typer.Argument(..., help="Proposal ID to reject")):
    """Reject a proposal."""
    prop = db.get_proposal_by_id(proposal_id)
    if not prop:
        console.print(f"[red]Proposal {proposal_id} not found.[/red]")
        raise typer.Exit(1)
    db.update_proposal_status(proposal_id, "REJECTED")
    console.print(f"[yellow]Proposal {proposal_id} rejected.[/yellow]")


@app.command()
def status():
    """System health check."""
    config = get_settings()
    checks: list[tuple[str, str, str]] = []

    # Binance
    try:
        import ccxt
        ex = ccxt.binanceusdm({"options": {"sandbox": config.BINANCE_TESTNET}})
        ts = ex.fetch_time()
        checks.append(("Binance API", "OK", f"Server time: {datetime.utcfromtimestamp(ts/1000).isoformat()}"))
    except Exception as e:
        checks.append(("Binance API", "FAIL", str(e)[:80]))

    # Ollama
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{config.OLLAMA_URL.rstrip('/')}/api/tags")
            r.raise_for_status()
        checks.append(("Ollama", "OK", config.OLLAMA_URL))
    except Exception as e:
        checks.append(("Ollama", "FAIL", str(e)[:80]))

    # DuckDB
    try:
        db_path = Path(config.DB_PATH)
        size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0
        with db.connection() as conn:
            tables = ["scans", "ohlcv", "signals", "proposals", "daily_pnl", "funding_history"]
            counts = []
            for t in tables:
                try:
                    r = conn.execute(f"SELECT count(*) FROM {t}").fetchone()
                    counts.append(f"{t}: {r[0] if r else 0}")
                except Exception:
                    counts.append(f"{t}: N/A")
        checks.append(("DuckDB", "OK", f"{size_mb:.2f} MB | " + ", ".join(counts)))
    except Exception as e:
        checks.append(("DuckDB", "FAIL", str(e)[:80]))

    # Cloud LLM
    key_present = bool(config.CLOUD_LLM_API_KEY and config.CLOUD_LLM_API_KEY.strip())
    checks.append(("Cloud LLM Key", "OK" if key_present else "MISSING", config.CLOUD_LLM_PROVIDER))

    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")
    for name, st, detail in checks:
        style = "green" if st == "OK" else ("red" if st == "FAIL" else "yellow")
        table.add_row(name, f"[{style}]{st}[/{style}]", detail)
    console.print(Panel(table, title="[bold]Health Check[/bold]"))


@app.command()
def live():
    """Start continuous paper trading daemon (runs pipeline every hour)."""
    last_scan: datetime | None = None
    last_proposals: list = []
    today_pnl: float | None = None

    def _make_dashboard() -> Panel:
        config = get_settings()
        table = Table(title="Live Dashboard")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Last Scan", last_scan.strftime("%Y-%m-%d %H:%M") if last_scan else "Never")
        table.add_row("Pending Proposals", str(len(last_proposals)))
        table.add_row("Today PnL", f"{today_pnl:.2f}%" if today_pnl is not None else "N/A")
        table.add_row("DRY_RUN", str(config.DRY_RUN))
        table.add_row("Max Positions", str(config.MAX_OPEN_POSITIONS))
        return Panel(table, title="[bold]Futures Intel Live[/bold] — Ctrl+C to stop")

    async def _run_pipeline():
        nonlocal last_scan, last_proposals, today_pnl
        engine = TradingEngine()
        proposals = await engine.run_pipeline()
        await engine.shutdown()
        last_scan = datetime.utcnow()
        last_proposals = db.get_pending_proposals()
        pnl_rec = db.get_daily_pnl(date.today())
        today_pnl = pnl_rec["realized_pnl"] if pnl_rec else None

    def _scheduled_run():
        _run_async(_run_pipeline())

    scheduler = AsyncIOScheduler()
    scheduler.add_job(_scheduled_run, "interval", hours=1, id="pipeline")
    scheduler.start()

    # Initial run
    _run_async(_run_pipeline())

    try:
        with Live(_make_dashboard(), refresh_per_second=0.5, console=console) as live:
            while True:
                import time
                time.sleep(2)
                live.update(_make_dashboard())
    except KeyboardInterrupt:
        pass
    finally:
        scheduler.shutdown(wait=False)
        console.print("\n[yellow]Daemon stopped.[/yellow]")


if __name__ == "__main__":
    app()
