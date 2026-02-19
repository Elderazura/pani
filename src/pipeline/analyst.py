"""Stage 2: Quant analyst — calculates indicators and detects trading signals."""

from datetime import datetime

import pandas as pd
from loguru import logger
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

from ..config import get_settings
from ..db import Database
from ..models import CoinScan, TechnicalSignal, SignalType


def _compute_indicators(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series] | None:
    """Compute RSI, MACD line, MACD signal, MACD hist, ATR. Returns None if failed."""
    try:
        rsi = RSIIndicator(close=df["close"], window=14).rsi()
        macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd.macd()
        macd_signal = macd.macd_signal()
        macd_hist = macd.macd_diff()
        atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
        return rsi, macd_line, macd_signal, macd_hist, atr
    except Exception:
        return None


def quick_analyze(symbol: str, df: pd.DataFrame) -> dict:
    """
    For backtesting — returns raw indicator values without signal detection.
    Returns {"rsi": float, "macd": float, "macd_signal": float, "macd_hist": float, "atr": float}
    """
    df = _prepare_df(df)
    if len(df) < 100:
        return {}

    result = _compute_indicators(df)
    if result is None:
        return {}

    rsi, macd_line, macd_signal, macd_hist, atr = result
    last = len(df) - 1
    return {
        "rsi": float(rsi.iloc[last]) if pd.notna(rsi.iloc[last]) else 0.0,
        "macd": float(macd_line.iloc[last]) if pd.notna(macd_line.iloc[last]) else 0.0,
        "macd_signal": float(macd_signal.iloc[last]) if pd.notna(macd_signal.iloc[last]) else 0.0,
        "macd_hist": float(macd_hist.iloc[last]) if pd.notna(macd_hist.iloc[last]) else 0.0,
        "atr": float(atr.iloc[last]) if pd.notna(atr.iloc[last]) else 0.0,
    }


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has required columns and is sorted by time ascending."""
    required = {"open", "high", "low", "close", "volume"}
    time_col = "bar_time" if "bar_time" in df.columns else "timestamp"
    cols = {c.lower(): c for c in df.columns}
    out = df.copy()
    for r in required:
        if r not in out.columns and r in cols:
            out[r] = out[cols[r]]
    if time_col in out.columns:
        out = out.sort_values(time_col).reset_index(drop=True)
    return out


class QuantAnalyst:
    """Calculates indicators and detects trading signals."""

    def __init__(self, config=None):
        """Store settings."""
        self.config = config or get_settings()

    def analyze(
        self,
        symbol: str,
        df: pd.DataFrame,
        funding_rate: float | None = None,
        open_interest_change_pct: float | None = None,
        timeframe: str = "1h",
    ) -> TechnicalSignal | None:
        """
        Analyze OHLCV data and return a signal if found.
        Expects df with columns: open, high, low, close, volume, bar_time
        """
        df = _prepare_df(df)
        if len(df) < 100:
            logger.debug("{}: insufficient data ({} rows, need 100)", symbol, len(df))
            return None

        result = _compute_indicators(df)
        if result is None:
            logger.warning("{}: indicator calculation failed", symbol)
            return None

        rsi_series, macd_line, macd_signal, macd_hist, atr_series = result
        n = len(df) - 1
        rsi = float(rsi_series.iloc[n]) if pd.notna(rsi_series.iloc[n]) else 50.0
        macd_val = float(macd_line.iloc[n]) if pd.notna(macd_line.iloc[n]) else 0.0
        macd_sig = float(macd_signal.iloc[n]) if pd.notna(macd_signal.iloc[n]) else 0.0
        macd_hist_val = float(macd_hist.iloc[n]) if pd.notna(macd_hist.iloc[n]) else 0.0
        atr_val = float(atr_series.iloc[n]) if pd.notna(atr_series.iloc[n]) else 0.0

        macd_hist_prev = float(macd_hist.iloc[n - 1]) if n >= 1 and pd.notna(macd_hist.iloc[n - 1]) else 0.0
        macd_3_ago = float(macd_line.iloc[-3]) if n >= 2 and pd.notna(macd_line.iloc[-3]) else 0.0
        sig_3_ago = float(macd_signal.iloc[-3]) if n >= 2 and pd.notna(macd_signal.iloc[-3]) else 0.0

        signal = self._detect_signal(
            symbol=symbol,
            rsi=rsi,
            macd_val=macd_val,
            macd_sig=macd_sig,
            macd_hist=macd_hist_val,
            macd_hist_prev=macd_hist_prev,
            macd_3_ago=macd_3_ago,
            sig_3_ago=sig_3_ago,
            atr_val=atr_val,
            df=df,
            funding_rate=funding_rate,
            open_interest_change_pct=open_interest_change_pct,
            timeframe=timeframe,
        )

        if signal:
            logger.info("{}: signal {} (confidence {:.2f})", symbol, signal.signal_type.value, signal.confidence)
        else:
            logger.debug("{}: no signal", symbol)

        return signal

    def _detect_signal(
        self,
        symbol: str,
        rsi: float,
        macd_val: float,
        macd_sig: float,
        macd_hist: float,
        macd_hist_prev: float,
        macd_3_ago: float,
        sig_3_ago: float,
        atr_val: float,
        df: pd.DataFrame,
        funding_rate: float | None,
        open_interest_change_pct: float | None,
        timeframe: str,
    ) -> TechnicalSignal | None:
        """Check signal conditions in priority order."""
        now = datetime.utcnow()

        # a. OVERSOLD
        if rsi < 30 and macd_hist > macd_hist_prev:
            conf = 0.5 + (30 - rsi) / 50  # RSI 30=0.5, 20=0.7, 10=0.9, 0=1.0
            conf = min(1.0, max(0.5, conf))
            return TechnicalSignal(
                symbol=symbol,
                signal_type=SignalType.OVERSOLD,
                rsi=rsi,
                macd_value=macd_val,
                macd_signal=macd_sig,
                macd_histogram=macd_hist,
                atr=atr_val,
                funding_rate=funding_rate,
                open_interest_change_pct=open_interest_change_pct,
                timeframe=timeframe,
                confidence=conf,
                timestamp=now,
            )

        # b. BULLISH_DIVERGENCE
        if len(df) >= 20:
            rsi_full = RSIIndicator(close=df["close"], window=14).rsi()
            if rsi_full is not None and len(rsi_full) >= 20:
                price_min_last = df["low"].iloc[-10:].min()
                price_min_prev = df["low"].iloc[-20:-10].min()
                rsi_min_last = rsi_full.iloc[-10:].min()
                rsi_min_prev = rsi_full.iloc[-20:-10].min()
                if pd.notna(rsi_min_last) and pd.notna(rsi_min_prev) and price_min_last < price_min_prev and rsi_min_last > rsi_min_prev:
                    conf = 0.7
                    if macd_hist > 0 and macd_hist_prev <= 0:
                        conf += 0.1
                    conf = min(1.0, conf)
                    return TechnicalSignal(
                        symbol=symbol,
                        signal_type=SignalType.BULLISH_DIVERGENCE,
                        rsi=rsi,
                        macd_value=macd_val,
                        macd_signal=macd_sig,
                        macd_histogram=macd_hist,
                        atr=atr_val,
                        funding_rate=funding_rate,
                        open_interest_change_pct=open_interest_change_pct,
                        timeframe=timeframe,
                        confidence=conf,
                        timestamp=now,
                    )

        # c. MACD_CROSS
        if len(df) >= 3 and macd_3_ago < sig_3_ago and macd_val > macd_sig and 35 <= rsi <= 65:
            conf = 0.6
            vol_last_5 = df["volume"].tail(5)
            if len(vol_last_5) == 5 and vol_last_5.is_monotonic_increasing:
                conf += 0.1
            return TechnicalSignal(
                symbol=symbol,
                signal_type=SignalType.MACD_CROSS,
                rsi=rsi,
                macd_value=macd_val,
                macd_signal=macd_sig,
                macd_histogram=macd_hist,
                atr=atr_val,
                funding_rate=funding_rate,
                open_interest_change_pct=open_interest_change_pct,
                timeframe=timeframe,
                confidence=conf,
                timestamp=now,
            )

        # d. FUNDING_EXTREME
        if funding_rate is not None and funding_rate < -0.01:
            return TechnicalSignal(
                symbol=symbol,
                signal_type=SignalType.FUNDING_EXTREME,
                rsi=rsi,
                macd_value=macd_val,
                macd_signal=macd_sig,
                macd_histogram=macd_hist,
                atr=atr_val,
                funding_rate=funding_rate,
                open_interest_change_pct=open_interest_change_pct,
                timeframe=timeframe,
                confidence=0.65,
                timestamp=now,
            )

        # e. OI_SPIKE
        if open_interest_change_pct is not None and open_interest_change_pct > 15 and rsi < 45:
            return TechnicalSignal(
                symbol=symbol,
                signal_type=SignalType.OI_SPIKE,
                rsi=rsi,
                macd_value=macd_val,
                macd_signal=macd_sig,
                macd_histogram=macd_hist,
                atr=atr_val,
                funding_rate=funding_rate,
                open_interest_change_pct=open_interest_change_pct,
                timeframe=timeframe,
                confidence=0.55,
                timestamp=now,
            )

        return None

    def analyze_batch(
        self, scans: list[CoinScan], db: Database
    ) -> list[TechnicalSignal]:
        """
        For each CoinScan, fetch OHLCV from DuckDB, run analyze, save signals.
        Returns signals sorted by confidence descending.
        """
        signals: list[TechnicalSignal] = []
        for scan in scans:
            df = db.get_ohlcv(scan.symbol, "1h", 500)
            if df is None or len(df) < 100:
                logger.warning(
                    "{}: insufficient OHLCV data ({} rows), skipping",
                    scan.symbol,
                    len(df) if df is not None else 0,
                )
                continue
            signal = self.analyze(
                symbol=scan.symbol,
                df=df,
                funding_rate=scan.funding_rate,
                open_interest_change_pct=None,
                timeframe="1h",
            )
            if signal is not None:
                db.save_signal(signal)
                signals.append(signal)
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
