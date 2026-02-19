"""DuckDB persistence layer."""

from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from loguru import logger

from .config import get_settings
from .models import CoinScan, OHLCVBar, TechnicalSignal, TradeProposal


class Database:
    """DuckDB database manager with singleton connection."""

    _instance: "Database | None" = None
    _conn: duckdb.DuckDBPyConnection | None = None

    def __new__(cls) -> "Database":
        """Singleton: return existing instance or create new one."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize with db_path from config. Connection created on first use."""
        settings = get_settings()
        self._db_path = Path(settings.DB_PATH)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        if Database._conn is None:
            self._init_connection()

    def _init_connection(self) -> None:
        """Create and store the singleton connection."""
        Database._conn = duckdb.connect(str(self._db_path))
        self._ensure_schema(Database._conn)
        logger.info("Database connection initialized: {}", self._db_path)

    def _ensure_schema(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create tables if they do not exist."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                symbol VARCHAR,
                volume_24h DOUBLE,
                price DOUBLE,
                funding_rate DOUBLE,
                open_interest DOUBLE,
                long_short_ratio DOUBLE,
                scanned_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR,
                timeframe VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                bar_time TIMESTAMP,
                PRIMARY KEY (symbol, timeframe, bar_time)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                symbol VARCHAR,
                signal_type VARCHAR,
                rsi DOUBLE,
                macd_value DOUBLE,
                macd_signal DOUBLE,
                macd_histogram DOUBLE,
                atr DOUBLE,
                funding_rate DOUBLE,
                oi_change_pct DOUBLE,
                confidence DOUBLE,
                timeframe VARCHAR,
                detected_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS proposals (
                id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                action VARCHAR,
                entry_price DOUBLE,
                stop_loss DOUBLE,
                take_profit DOUBLE,
                leverage INTEGER,
                risk_reward DOUBLE,
                sentiment_score DOUBLE,
                sentiment_reason VARCHAR,
                status VARCHAR,
                created_at TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date DATE PRIMARY KEY,
                realized_pnl DOUBLE,
                trade_count INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS funding_history (
                symbol VARCHAR,
                rate DOUBLE,
                timestamp TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        conn.commit()
        logger.debug("Database schema ensured")

    @contextmanager
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Context manager yielding the singleton connection."""
        if Database._conn is None:
            self._init_connection()
        yield Database._conn

    def save_scans(self, scans: list[CoinScan]) -> int:
        """Bulk insert scan records. Returns count of rows inserted."""
        if not scans:
            return 0
        try:
            with self.connection() as conn:
                rows = [
                    (
                        s.symbol,
                        s.volume_24h,
                        s.price,
                        s.funding_rate,
                        s.open_interest,
                        s.long_short_ratio,
                        s.timestamp,
                    )
                    for s in scans
                ]
                conn.executemany(
                    """
                    INSERT INTO scans (symbol, volume_24h, price, funding_rate, open_interest, long_short_ratio, scanned_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()
                logger.info("Saved {} scan(s)", len(rows))
                return len(rows)
        except Exception as e:
            logger.exception("Failed to save scans: {}", e)
            raise

    def save_funding_history(
        self, records: list[tuple[str, float, datetime]]
    ) -> int:
        """Upsert funding rate history. Records: (symbol, rate, timestamp). Returns count."""
        if not records:
            return 0
        try:
            with self.connection() as conn:
                for symbol, rate, ts in records:
                    conn.execute(
                        """
                        INSERT INTO funding_history (symbol, rate, timestamp)
                        VALUES (?, ?, ?)
                        ON CONFLICT (symbol, timestamp) DO UPDATE SET rate = excluded.rate
                        """,
                        [symbol, rate, ts],
                    )
                conn.commit()
                logger.info("Saved {} funding history record(s)", len(records))
                return len(records)
        except Exception as e:
            logger.exception("Failed to save funding history: {}", e)
            raise

    def save_ohlcv(self, bars: list[OHLCVBar]) -> int:
        """Upsert OHLCV bars using INSERT OR REPLACE semantics. Returns count."""
        if not bars:
            return 0
        try:
            with self.connection() as conn:
                for bar in bars:
                    conn.execute(
                        """
                        INSERT INTO ohlcv (symbol, timeframe, open, high, low, close, volume, bar_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (symbol, timeframe, bar_time) DO UPDATE SET
                            open = excluded.open,
                            high = excluded.high,
                            low = excluded.low,
                            close = excluded.close,
                            volume = excluded.volume
                        """,
                        [
                            bar.symbol,
                            bar.timeframe,
                            bar.open,
                            bar.high,
                            bar.low,
                            bar.close,
                            bar.volume,
                            bar.timestamp,
                        ],
                    )
                conn.commit()
                logger.info("Saved {} OHLCV bar(s)", len(bars))
                return len(bars)
        except Exception as e:
            logger.exception("Failed to save OHLCV: {}", e)
            raise

    def save_signal(self, signal: TechnicalSignal) -> None:
        """Insert a technical signal."""
        try:
            with self.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO signals (symbol, signal_type, rsi, macd_value, macd_signal, macd_histogram, atr, funding_rate, oi_change_pct, confidence, timeframe, detected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        signal.symbol,
                        signal.signal_type.value,
                        signal.rsi,
                        signal.macd_value,
                        signal.macd_signal,
                        signal.macd_histogram,
                        signal.atr,
                        signal.funding_rate,
                        signal.open_interest_change_pct,
                        signal.confidence,
                        signal.timeframe,
                        signal.timestamp,
                    ],
                )
                conn.commit()
                logger.info("Saved signal: {} {}", signal.symbol, signal.signal_type.value)
        except Exception as e:
            logger.exception("Failed to save signal: {}", e)
            raise

    def save_proposal(self, proposal: TradeProposal) -> None:
        """Insert a trade proposal."""
        try:
            with self.connection() as conn:
                sentiment_score = proposal.sentiment.score if proposal.sentiment else None
                sentiment_reason = (
                    proposal.sentiment.reason if proposal.sentiment else ""
                )
                conn.execute(
                    """
                    INSERT INTO proposals (id, symbol, action, entry_price, stop_loss, take_profit, leverage, risk_reward, sentiment_score, sentiment_reason, status, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        proposal.id,
                        proposal.symbol,
                        proposal.action.value,
                        proposal.entry_price,
                        proposal.stop_loss,
                        proposal.take_profit,
                        proposal.leverage,
                        proposal.risk_reward_ratio,
                        sentiment_score,
                        sentiment_reason,
                        proposal.status,
                        proposal.created_at,
                        proposal.expires_at,
                    ],
                )
                conn.commit()
                logger.info("Saved proposal {}: {} {}", proposal.id, proposal.symbol, proposal.action.value)
        except Exception as e:
            logger.exception("Failed to save proposal: {}", e)
            raise

    def get_pending_proposals(self) -> list[dict[str, Any]]:
        """Return all proposals with status PENDING and not expired."""
        return self.get_proposals("pending")

    def get_proposals(
        self, status_filter: str = "pending"
    ) -> list[dict[str, Any]]:
        """Return proposals filtered by status: pending, approved, rejected, expired, all."""
        try:
            with self.connection() as conn:
                now = datetime.utcnow()
                status_filter = (status_filter or "pending").lower()
                if status_filter == "pending":
                    sql = """
                        SELECT id, symbol, action, entry_price, stop_loss, take_profit, leverage, risk_reward, sentiment_score, sentiment_reason, status, created_at, expires_at
                        FROM proposals WHERE status = 'PENDING' AND expires_at > ?
                        ORDER BY created_at DESC
                    """
                    params = [now]
                elif status_filter == "approved":
                    sql = "SELECT id, symbol, action, entry_price, stop_loss, take_profit, leverage, risk_reward, sentiment_score, sentiment_reason, status, created_at, expires_at FROM proposals WHERE status = 'APPROVED' ORDER BY created_at DESC"
                    params = []
                elif status_filter == "rejected":
                    sql = "SELECT id, symbol, action, entry_price, stop_loss, take_profit, leverage, risk_reward, sentiment_score, sentiment_reason, status, created_at, expires_at FROM proposals WHERE status = 'REJECTED' ORDER BY created_at DESC"
                    params = []
                elif status_filter == "expired":
                    sql = """
                        SELECT id, symbol, action, entry_price, stop_loss, take_profit, leverage, risk_reward, sentiment_score, sentiment_reason, status, created_at, expires_at
                        FROM proposals WHERE status = 'EXPIRED' OR (status = 'PENDING' AND expires_at <= ?)
                        ORDER BY created_at DESC
                    """
                    params = [now]
                else:
                    sql = "SELECT id, symbol, action, entry_price, stop_loss, take_profit, leverage, risk_reward, sentiment_score, sentiment_reason, status, created_at, expires_at FROM proposals ORDER BY created_at DESC"
                    params = []
                result = conn.execute(sql, params).fetchdf()
                if result.empty:
                    return []
                return result.to_dict(orient="records")
        except Exception as e:
            logger.exception("Failed to get proposals: {}", e)
            raise

    def get_proposal_by_id(self, proposal_id: str) -> dict[str, Any] | None:
        """Return a single proposal by id, or None."""
        try:
            with self.connection() as conn:
                result = conn.execute(
                    "SELECT id, symbol, action, entry_price, stop_loss, take_profit, leverage, risk_reward, sentiment_score, sentiment_reason, status, created_at, expires_at FROM proposals WHERE id = ?",
                    [proposal_id],
                ).fetchone()
                if result is None:
                    return None
                cols = ["id", "symbol", "action", "entry_price", "stop_loss", "take_profit", "leverage", "risk_reward", "sentiment_score", "sentiment_reason", "status", "created_at", "expires_at"]
                return dict(zip(cols, result))
        except Exception as e:
            logger.exception("Failed to get proposal: {}", e)
            raise

    def get_latest_scans(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return latest scan results per symbol (most recent scan batch)."""
        try:
            with self.connection() as conn:
                df = conn.execute(
                    """
                    SELECT symbol, volume_24h, price, funding_rate, open_interest, long_short_ratio, scanned_at
                    FROM scans
                    ORDER BY scanned_at DESC
                    LIMIT ?
                    """,
                    [limit],
                ).fetchdf()
                if df.empty:
                    return []
                return df.to_dict(orient="records")
        except Exception as e:
            logger.exception("Failed to get latest scans: {}", e)
            raise

    def get_latest_signals(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return latest technical signals."""
        try:
            with self.connection() as conn:
                df = conn.execute(
                    """
                    SELECT symbol, signal_type, rsi, macd_value, macd_signal, macd_histogram, atr, confidence, timeframe, detected_at
                    FROM signals
                    ORDER BY detected_at DESC
                    LIMIT ?
                    """,
                    [limit],
                ).fetchdf()
                if df.empty:
                    return []
                return df.to_dict(orient="records")
        except Exception as e:
            logger.exception("Failed to get latest signals: {}", e)
            raise

    def get_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> pd.DataFrame:
        """Return OHLCV bars for symbol/timeframe, most recent first, limited."""
        try:
            with self.connection() as conn:
                df = conn.execute(
                    """
                    SELECT symbol, timeframe, open, high, low, close, volume, bar_time
                    FROM ohlcv
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY bar_time DESC
                    LIMIT ?
                    """,
                    [symbol, timeframe, limit],
                ).fetchdf()
                logger.debug("Fetched {} OHLCV rows for {}/{}", len(df), symbol, timeframe)
                return df
        except Exception as e:
            logger.exception("Failed to get OHLCV: {}", e)
            raise

    def update_proposal_status(self, id: str, status: str) -> None:
        """Update proposal status by id."""
        try:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE proposals SET status = ? WHERE id = ?",
                    [status, id],
                )
                conn.commit()
                logger.info("Updated proposal {} status to {}", id, status)
        except Exception as e:
            logger.exception("Failed to update proposal status: {}", e)
            raise

    def get_daily_pnl(self, d: date) -> dict[str, Any] | None:
        """Return daily PNL record for date, or None."""
        try:
            with self.connection() as conn:
                result = conn.execute(
                    "SELECT date, realized_pnl, trade_count FROM daily_pnl WHERE date = ?",
                    [d],
                ).fetchone()
                if result is None:
                    return None
                return {
                    "date": result[0],
                    "realized_pnl": result[1],
                    "trade_count": result[2],
                }
        except Exception as e:
            logger.exception("Failed to get daily PNL: {}", e)
            raise

    def update_daily_pnl(
        self, d: date, pnl: float, trade_count: int
    ) -> None:
        """Insert or replace daily PNL record."""
        try:
            with self.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO daily_pnl (date, realized_pnl, trade_count)
                    VALUES (?, ?, ?)
                    ON CONFLICT (date) DO UPDATE SET
                        realized_pnl = excluded.realized_pnl,
                        trade_count = excluded.trade_count
                    """,
                    [d, pnl, trade_count],
                )
                conn.commit()
                logger.info("Updated daily PNL for {}: pnl={}, trades={}", d, pnl, trade_count)
        except Exception as e:
            logger.exception("Failed to update daily PNL: {}", e)
            raise

    def expire_old_proposals(self) -> int:
        """Set status to EXPIRED where expires_at < now and status is PENDING. Returns count updated."""
        try:
            with self.connection() as conn:
                now = datetime.utcnow()
                count_result = conn.execute(
                    "SELECT count(*) FROM proposals WHERE status = 'PENDING' AND expires_at < ?",
                    [now],
                ).fetchone()
                count = int(count_result[0]) if count_result else 0
                if count > 0:
                    conn.execute(
                        """
                        UPDATE proposals SET status = 'EXPIRED'
                        WHERE status = 'PENDING' AND expires_at < ?
                        """,
                        [now],
                    )
                    conn.commit()
                logger.info("Expired {} old proposal(s)", count)
                return count
        except Exception as e:
            logger.exception("Failed to expire old proposals: {}", e)
            raise


db = Database()
