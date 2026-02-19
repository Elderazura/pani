"""Backfills historical data into DuckDB for backtesting."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import ccxt.async_support as ccxt
from loguru import logger

from ..config import get_settings
from ..db import db
from ..models import OHLCVBar
from .scanner import MarketScanner


class DataCollector:
    """Backfills historical OHLCV and funding rate data."""

    def __init__(self, config=None):
        """Init ccxt Binance (same as scanner)."""
        self.config = config or get_settings()
        self._exchange: ccxt.binanceusdm | None = None

    def _create_exchange(self) -> ccxt.binanceusdm:
        """Return configured ccxt.binanceusdm instance."""
        options: dict = {
            "enableRateLimit": True,
            "defaultType": "swap",
        }
        if self.config.BINANCE_TESTNET:
            options["sandbox"] = True

        return ccxt.binanceusdm(
            {
                "apiKey": self.config.BINANCE_API_KEY or None,
                "secret": self.config.BINANCE_SECRET or None,
                "options": options,
            }
        )

    @property
    def exchange(self) -> ccxt.binanceusdm:
        """Lazy-init exchange instance."""
        if self._exchange is None:
            self._exchange = self._create_exchange()
        return self._exchange

    async def backfill(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        days: int = 180,
    ) -> int:
        """
        For each symbol, fetch OHLCV in batches of 1000 until current time.
        Returns total bars collected.
        """
        rate_limit_ms = getattr(self.exchange, "rateLimit", 1000) or 1000
        sleep_sec = rate_limit_ms / 1000.0

        end_ts = int(datetime.utcnow().timestamp() * 1000)
        start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        total_bars = 0

        for symbol in symbols:
            since = start_ts
            symbol_bars = 0

            while since < end_ts:
                try:
                    candles = await self.exchange.fetch_ohlcv(
                        symbol, timeframe, since=since, limit=1000
                    )
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    logger.exception("Failed to fetch OHLCV for {}: {}", symbol, e)
                    break

                if not candles:
                    break

                bars = [
                    OHLCVBar(
                        symbol=symbol,
                        timeframe=timeframe,
                        open=c[1],
                        high=c[2],
                        low=c[3],
                        close=c[4],
                        volume=c[5],
                        timestamp=datetime.utcfromtimestamp(c[0] / 1000),
                    )
                    for c in candles
                ]

                db.save_ohlcv(bars)
                symbol_bars += len(bars)
                total_bars += len(bars)

                last_ts = candles[-1][0]
                since = last_ts + 1

                await asyncio.sleep(sleep_sec)

            start_dt = datetime.utcfromtimestamp(start_ts / 1000)
            end_dt = datetime.utcfromtimestamp(
                (since - 1) / 1000 if symbol_bars else start_ts / 1000
            )
            logger.info(
                "Backfilled {}: {} bars from {} to {}",
                symbol,
                symbol_bars,
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
            )

        return total_bars

    async def backfill_funding_rates(
        self,
        symbols: list[str],
        days: int = 180,
    ) -> int:
        """
        Fetch funding rate history for each symbol.
        Returns total records collected.
        """
        rate_limit_ms = getattr(self.exchange, "rateLimit", 1000) or 1000
        sleep_sec = rate_limit_ms / 1000.0

        start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        total_records = 0

        for symbol in symbols:
            since = start_ts
            symbol_records = 0

            while True:
                try:
                    history = await self.exchange.fetch_funding_rate_history(
                        symbol, since=since, limit=1000
                    )
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    logger.debug("Funding history for {}: {}", symbol, e)
                    break
                except AttributeError:
                    logger.warning(
                        "fetch_funding_rate_history not supported for this exchange"
                    )
                    return total_records

                if not history:
                    break

                records = []
                for h in history:
                    rate = h.get("fundingRate")
                    ts = h.get("fundingTimestamp") or h.get("timestamp")
                    if rate is not None and ts is not None:
                        records.append(
                            (symbol, float(rate), datetime.utcfromtimestamp(ts / 1000))
                        )

                if records:
                    db.save_funding_history(records)
                    symbol_records += len(records)
                    total_records += len(records)
                    last_ts = max(r[2] for r in records)
                    since = int(last_ts.timestamp() * 1000) + 1

                if len(history) < 1000 or not records:
                    break

                await asyncio.sleep(sleep_sec)

            logger.info(
                "Backfilled funding rates for {}: {} records",
                symbol,
                symbol_records,
            )

        return total_records

    async def backfill_top_coins(
        self,
        count: int = 50,
        timeframe: str = "1h",
        days: int = 180,
    ) -> int:
        """
        Run scanner, take top `count` symbols, backfill OHLCV.
        Returns total bars collected.
        """
        scanner = MarketScanner(self.config)
        scans = await scanner.scan(limit=count)
        symbols = [s.symbol for s in scans]

        if not symbols:
            logger.warning("No symbols from scanner, nothing to backfill")
            return 0

        total_bars = await self.backfill(symbols, timeframe, days)

        db_path = Path(self.config.DB_PATH)
        size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0

        logger.info(
            "{} symbols, {} bars collected, DB size: {:.2f}MB",
            len(symbols),
            total_bars,
            size_mb,
        )
        return total_bars


def _run_async(coro):
    """Run async coroutine from sync context."""
    return asyncio.run(coro)


if __name__ == "__main__":
    import typer

    app = typer.Typer(help="Backfill historical data")

    @app.command()
    def backfill(
        symbols: str = typer.Option(
            "",
            "--symbols",
            help="Comma-separated symbols (e.g. BTC/USDT:USDT,ETH/USDT:USDT)",
        ),
        timeframe: str = typer.Option("1h", "--timeframe", help="OHLCV timeframe"),
        days: int = typer.Option(180, "--days", help="Days of history"),
    ):
        """Backfill OHLCV for given symbols."""
        if not symbols.strip():
            typer.echo("Error: --symbols is required")
            raise typer.Exit(1)
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        collector = DataCollector()
        total = _run_async(collector.backfill(symbol_list, timeframe, days))
        typer.echo(f"Backfilled {total} bars")

    @app.command("backfill-top")
    def backfill_top(
        count: int = typer.Option(50, "--count", help="Number of top symbols"),
        timeframe: str = typer.Option("1h", "--timeframe", help="OHLCV timeframe"),
        days: int = typer.Option(180, "--days", help="Days of history"),
    ):
        """Backfill OHLCV for top coins by volume."""
        collector = DataCollector()
        total = _run_async(collector.backfill_top_coins(count, timeframe, days))
        typer.echo(f"Backfilled {total} bars")

    app()
