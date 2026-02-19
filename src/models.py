"""Pydantic v2 models for data boundaries."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class Side(str, Enum):
    """Order side."""

    LONG = "long"
    SHORT = "short"


class OrderStatus(str, Enum):
    """Order lifecycle status."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    CLOSED = "closed"


class SignalType(str, Enum):
    """Technical signal types."""

    OVERSOLD = "oversold"
    BULLISH_DIVERGENCE = "bullish_divergence"
    MACD_CROSS = "macd_cross"
    FUNDING_EXTREME = "funding_extreme"
    OI_SPIKE = "oi_spike"


class TradeAction(str, Enum):
    """Trade action recommendation."""

    LONG = "long"
    SHORT = "short"
    SKIP = "skip"


# -----------------------------------------------------------------------------
# Market & Scan Models
# -----------------------------------------------------------------------------


class CoinScan(BaseModel):
    """Market scan snapshot for a symbol."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    volume_24h: float
    price: float
    funding_rate: float | None = None
    open_interest: float | None = None
    long_short_ratio: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OHLCVBar(BaseModel):
    """OHLCV candle bar with timeframe."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# -----------------------------------------------------------------------------
# Signal Models
# -----------------------------------------------------------------------------


class TechnicalSignal(BaseModel):
    """Technical analysis signal with indicators."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    signal_type: SignalType
    rsi: float
    macd_value: float
    macd_signal: float
    macd_histogram: float
    atr: float
    funding_rate: float | None = None
    open_interest_change_pct: float | None = None
    timeframe: str = "1h"
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SentimentResult(BaseModel):
    """LLM sentiment analysis result."""

    model_config = ConfigDict(from_attributes=True)

    score: float = Field(ge=-1.0, le=1.0)
    reason: str
    source: str = "ollama"
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# -----------------------------------------------------------------------------
# Trade Proposal
# -----------------------------------------------------------------------------


class TradeProposal(BaseModel):
    """Trade proposal combining technical signal and optional sentiment."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    action: TradeAction
    signal: TechnicalSignal
    sentiment: SentimentResult | None = None
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    risk_reward_ratio: float
    position_size_usd: float | None = None
    status: Literal["PENDING", "APPROVED", "REJECTED", "EXPIRED"] = "PENDING"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def set_expires_from_created(self) -> "TradeProposal":
        """Set expires_at to 2 hours after created_at when not provided."""
        if self.expires_at is None:
            object.__setattr__(
                self, "expires_at", self.created_at + timedelta(hours=2)
            )
        return self

    @computed_field
    @property
    def is_expired(self) -> bool:
        """True if current time is past expires_at."""
        return datetime.utcnow() > self.expires_at


# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------


class BacktestResult(BaseModel):
    """Backtest run result with performance metrics."""

    model_config = ConfigDict(from_attributes=True)

    strategy_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_return_pct: float
    avg_trade_duration_minutes: float
    start_date: datetime
    end_date: datetime


# -----------------------------------------------------------------------------
# Legacy models (for db.py compatibility)
# -----------------------------------------------------------------------------


class Position(BaseModel):
    """Open or closed position."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    side: Side
    size: float
    entry_price: float
    leverage: int = 1
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: datetime | None = None
    pnl: float | None = None


class Signal(BaseModel):
    """Trading signal from strategy or LLM."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    side: Side
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class OHLCV(BaseModel):
    """OHLCV candle data (legacy)."""

    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
