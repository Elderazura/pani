"""Stage 1: Market scanner for Binance Futures."""

import asyncio
from datetime import datetime
from pathlib import Path

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger

from ..config import get_settings
from ..db import db
from ..models import CoinScan


def _with_retry(max_retries: int = 3):
    """Decorator for exponential backoff on ccxt errors."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = 2**attempt
                        logger.warning(
                            "{} (attempt {}/{}), retrying in {}s: {}",
                            type(e).__name__,
                            attempt + 1,
                            max_retries,
                            delay,
                            str(e)[:100],
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.exception("All retries exhausted: {}", e)
                        raise last_error
            raise last_error

        return wrapper

    return decorator


class MarketScanner:
    """Finds the most active coins on Binance Futures."""

    def __init__(self, config=None):
        """Load settings and init ccxt Binance with sandbox, rate limiting, API keys."""
        self.config = config or get_settings()
        self._exchange: ccxt.binanceusdm | None = None

    def _create_exchange(self) -> ccxt.binanceusdm:
        """Return configured ccxt.binanceusdm instance with sandbox/live settings."""
        options: dict = {
            "enableRateLimit": True,
            "defaultType": "swap",
        }
        if self.config.BINANCE_TESTNET:
            options["sandbox"] = True

        exchange = ccxt.binanceusdm(
            {
                "apiKey": self.config.BINANCE_API_KEY or None,
                "secret": self.config.BINANCE_SECRET or None,
                "options": options,
            }
        )
        return exchange

    @property
    def exchange(self) -> ccxt.binanceusdm:
        """Lazy-init exchange instance."""
        if self._exchange is None:
            self._exchange = self._create_exchange()
        return self._exchange

    def _is_perpetual(self, symbol: str) -> bool:
        """True if symbol is USDT perpetual (not quarterly)."""
        return symbol.endswith(":USDT") and "_" not in symbol

    @_with_retry(max_retries=3)
    async def scan(self, limit: int = 5) -> list[CoinScan]:
        """
        Fetch top USDT-margined perpetual futures by volume.
        Saves all results to DuckDB, returns top `limit`.
        """
        exchange = self.exchange
        min_volume = self.config.MIN_VOLUME_USD
        now = datetime.utcnow()

        tickers = await exchange.fetch_tickers()
        perpetual = {
            s: t
            for s, t in tickers.items()
            if self._is_perpetual(s) and t.get("quoteVolume") is not None
        }
        volume_filtered = {
            s: t
            for s, t in perpetual.items()
            if (t.get("quoteVolume") or 0) >= min_volume
        }

        total = len(perpetual)
        filtered = len(volume_filtered)

        # Sort by volume descending
        sorted_symbols = sorted(
            volume_filtered.keys(),
            key=lambda s: volume_filtered[s].get("quoteVolume", 0) or 0,
            reverse=True,
        )

        scans: list[CoinScan] = []
        for symbol in sorted_symbols:
            ticker = volume_filtered[symbol]
            funding_rate: float | None = None
            open_interest: float | None = None

            try:
                fr = await exchange.fetch_funding_rate(symbol)
                if fr and "fundingRate" in fr:
                    funding_rate = float(fr["fundingRate"])
            except Exception as e:
                logger.debug("No funding rate for {}: {}", symbol, e)

            try:
                oi = await exchange.fetch_open_interest(symbol)
                if oi and "openInterest" in oi:
                    open_interest = float(oi["openInterest"])
            except Exception as e:
                logger.debug("No open interest for {}: {}", symbol, e)

            price = float(ticker.get("last") or ticker.get("close") or 0)
            volume_24h = float(ticker.get("quoteVolume") or 0)

            scans.append(
                CoinScan(
                    symbol=symbol,
                    volume_24h=volume_24h,
                    price=price,
                    funding_rate=funding_rate,
                    open_interest=open_interest,
                    long_short_ratio=None,
                    timestamp=now,
                )
            )

        db.save_scans(scans)
        logger.info(
            "Scanned {} pairs, {} passed volume filter, returning top {}",
            total,
            filtered,
            limit,
        )

        return scans[:limit]

    async def scan_from_csv(self, filepath: str) -> list[CoinScan]:
        """
        Load OHLCV from CSV for backtesting.
        CSV columns: symbol, open, high, low, close, volume, timestamp
        Group by symbol, sum volume, apply same filters, return top 5.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {filepath}")

        df = pd.read_csv(path)
        required = {"symbol", "open", "high", "low", "close", "volume", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        agg = (
            df.groupby("symbol")
            .agg(
                volume_24h=("volume", "sum"),
                price=("close", "last"),
            )
            .reset_index()
        )

        min_volume = self.config.MIN_VOLUME_USD
        filtered = agg[agg["volume_24h"] >= min_volume].copy()
        filtered = filtered.sort_values("volume_24h", ascending=False)

        now = datetime.utcnow()
        scans = [
            CoinScan(
                symbol=row["symbol"],
                volume_24h=float(row["volume_24h"]),
                price=float(row["price"]),
                funding_rate=None,
                open_interest=None,
                long_short_ratio=None,
                timestamp=now,
            )
            for _, row in filtered.head(5).iterrows()
        ]

        if scans:
            db.save_scans(scans)

        logger.info(
            "Scanned {} symbols from CSV, {} passed volume filter, returning top 5",
            len(agg),
            len(filtered),
        )
        return scans
