"""Main orchestrator for the three-stage trading funnel."""

from datetime import date

from loguru import logger
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .db import db
from .models import CoinScan, SentimentResult, TechnicalSignal, TradeAction, TradeProposal
from .pipeline.analyst import QuantAnalyst
from .pipeline.scanner import MarketScanner
from .pipeline.sentinel import SentimentScorer


class TradingEngine:
    """Orchestrates the three-stage funnel: scan → analyze → sentiment."""

    def __init__(self):
        """Initialize with config, db, and pipeline components."""
        self.config = get_settings()
        self.db = db
        self.scanner = MarketScanner(self.config)
        self.analyst = QuantAnalyst(self.config)
        self.sentinel = SentimentScorer(self.config)

    async def run_pipeline(self) -> list[TradeProposal]:
        """Execute the full three-stage funnel."""
        logger.info("=== Pipeline Starting ===")

        # Check daily loss limit
        today_pnl = self.db.get_daily_pnl(date.today())
        if today_pnl and today_pnl["realized_pnl"] < -(self.config.DAILY_LOSS_LIMIT_PCT):
            logger.warning("Daily loss limit hit. Pipeline halted.")
            return []

        # Expire old proposals
        self.db.expire_old_proposals()

        # Check current open positions count
        pending = self.db.get_pending_proposals()
        if len(pending) >= self.config.MAX_OPEN_POSITIONS:
            logger.info(
                "Max open positions ({}) reached. Skipping.",
                self.config.MAX_OPEN_POSITIONS,
            )
            return []

        # Stage 1: Scan
        logger.info("Stage 1: Scanning market...")
        coins = await self.scanner.scan()
        if not coins:
            logger.warning("No coins passed volume filter.")
            return []
        logger.info("Stage 1 complete: {} coins found", len(coins))

        # Stage 2: Analyze
        logger.info("Stage 2: Technical analysis...")
        signals = self.analyst.analyze_batch(coins, self.db)
        strong_signals = [s for s in signals if s.confidence >= 0.6]
        if not strong_signals:
            logger.info("No strong signals detected.")
            return []
        logger.info(
            "Stage 2 complete: {} signals with confidence >= 0.6",
            len(strong_signals),
        )

        # Stage 3: Sentiment
        logger.info("Stage 3: Sentiment scoring...")
        sentiments = await self.sentinel.score_batch(strong_signals)
        logger.info("Stage 3 complete: {} scores received", len(sentiments))

        # Build proposals
        proposals: list[TradeProposal] = []
        slots_remaining = self.config.MAX_OPEN_POSITIONS - len(pending)
        for signal, sentiment in zip(strong_signals, sentiments):
            if len(proposals) >= slots_remaining:
                break
            if sentiment.score < -0.3:
                logger.info(
                    "Skipping {}: bearish sentiment ({:.2f})",
                    signal.symbol,
                    sentiment.score,
                )
                continue

            proposal = self._build_proposal(signal, sentiment, coins)
            self.db.save_proposal(proposal)
            proposals.append(proposal)

        self._display_proposals(proposals)
        return proposals

    def _build_proposal(
        self,
        signal: TechnicalSignal,
        sentiment: SentimentResult,
        coins: list[CoinScan],
    ) -> TradeProposal:
        """Build a trade proposal with entry, SL, TP based on ATR."""
        coin = next(c for c in coins if c.symbol == signal.symbol)

        atr_multiplier_sl = self.config.STOP_LOSS_PCT / 1.5
        atr_multiplier_tp = self.config.TAKE_PROFIT_PCT / 1.5

        entry = coin.price
        stop_loss = entry - (signal.atr * atr_multiplier_sl)
        take_profit = entry + (signal.atr * atr_multiplier_tp)

        rr_ratio = (
            (take_profit - entry) / (entry - stop_loss)
            if entry != stop_loss
            else 0.0
        )

        return TradeProposal(
            symbol=signal.symbol,
            action=TradeAction.LONG,
            signal=signal,
            sentiment=sentiment,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=self.config.MAX_LEVERAGE,
            risk_reward_ratio=round(rr_ratio, 2),
            status="PENDING",
        )

    def _display_proposals(self, proposals: list[TradeProposal]) -> None:
        """Use Rich to display a formatted table of proposals."""
        if not proposals:
            return

        console = Console()
        table = Table(
            title="Trade Proposals",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Symbol", style="white")
        table.add_column("Action", style="white")
        table.add_column("Entry", justify="right")
        table.add_column("SL", justify="right")
        table.add_column("TP", justify="right")
        table.add_column("R:R", justify="right")
        table.add_column("Confidence", justify="right")
        table.add_column("Sentiment", justify="right")
        table.add_column("Status", style="white")

        for p in proposals:
            sentiment = p.sentiment
            score = sentiment.score if sentiment else 0.0
            if score >= 0.3:
                sent_style = "green"
            elif score <= -0.3:
                sent_style = "red"
            else:
                sent_style = "yellow"

            table.add_row(
                p.symbol,
                p.action.value.upper(),
                f"{p.entry_price:,.2f}",
                f"{p.stop_loss:,.2f}",
                f"{p.take_profit:,.2f}",
                f"{p.risk_reward_ratio:.2f}",
                f"{p.signal.confidence:.2f}",
                f"[{sent_style}]{score:.2f}[/{sent_style}]",
                p.status,
            )

        console.print(table)

    async def shutdown(self) -> None:
        """Close resources."""
        await self.sentinel.close()
