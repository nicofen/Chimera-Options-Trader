"""
chimera_v12/strategies/sector/engine.py
═══════════════════════════════════════════════════════════════════════════════
SECTOR STRATEGY ENGINE

Four independent strategy modules, each optimized for its asset class.
Signals feed into the VotingAgent for final resolution.

Sector A — Crypto   : On-chain inflow/outflow + memecoin volume spike
Sector B — Stocks   : Squeeze Probability Score + multi-factor model
Sector C — Forex    : NLP momentum bias on EMA + carry trade filter
Sector D — Futures  : Value Area mean reversion + AVWAP + seasonality
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import numpy as np

from chimera_v12.core.state import SharedState, TechnicalSignals
from chimera_v12.utils.ta import (
    ema, rsi, adx, atr_value, bollinger_squeeze,
    detect_rsi_divergence, vwap, volume_profile, anchored_vwap,
)
from chimera_v12.utils.logger import setup_logger

log = setup_logger("strategies.sector")


# ══════════════════════════════════════════════════════════════════════════════
# Sector A: Crypto Strategy
# ══════════════════════════════════════════════════════════════════════════════

class CryptoStrategy:
    """
    Signals based on:
    1. Exchange inflow spikes (whale risk-off signal)
    2. Memecoin volume spikes (risk appetite indicator)
    3. EMA crossover + RSI with funding rate filter
    4. BTC dominance trend (alt-season detector)

    Edge: Crypto funding rates create predictable mean-reversion cycles.
    Extreme positive funding (>1% per 8h) → crowded longs → contrarian SHORT.
    """

    def __init__(self, config: dict):
        self.cfg = config.get("crypto", {})

    def evaluate(self, symbol: str, bars: dict, whale_inflow: float = 0,
                 memecoin_vol: float = 0, funding_rate: float = 0.0) -> TechnicalSignals:
        closes = np.array(bars.get("close", []), dtype=float)
        highs  = np.array(bars.get("high",  []), dtype=float)
        lows   = np.array(bars.get("low",   []), dtype=float)

        if len(closes) < 50:
            return TechnicalSignals(symbol=symbol, direction="HOLD")

        fast = self.cfg.get("ema_fast", 9)
        slow = self.cfg.get("ema_slow", 21)
        ema_f = ema(closes, fast)
        ema_s = ema(closes, slow)
        rsi_v = rsi(closes, self.cfg.get("rsi_period", 14))
        atr_v = atr_value(highs, lows, closes)

        # EMA crossover direction
        ema_bull = ema_f[-1] > ema_s[-1]
        ema_bear = ema_f[-1] < ema_s[-1]

        direction = "HOLD"

        # Risk-off: whale inflow to exchanges → potential sell pressure
        if whale_inflow > self.cfg.get("btc_inflow_threshold", 1_000_000):
            direction = "HOLD"
            log.debug(f"{symbol} crypto: whale inflow ${whale_inflow:,.0f} → risk-off")
        elif memecoin_vol > self.cfg.get("sol_memecoin_spike_threshold", 50_000_000):
            # Extreme memecoin speculation → risk appetite peak → fade longs
            direction = "SELL" if ema_bear else "HOLD"
        elif rsi_v > self.cfg.get("rsi_overbought", 70):
            # Funding rate extreme + overbought → contrarian short
            if funding_rate > self.cfg.get("funding_rate_extreme", 0.01):
                direction = "SELL"
            else:
                direction = "HOLD"
        elif rsi_v < self.cfg.get("rsi_oversold", 30) and ema_bull:
            direction = "BUY"
        elif ema_bull and rsi_v > 50:
            direction = "BUY"
        elif ema_bear and rsi_v < 50:
            direction = "SELL"

        return TechnicalSignals(
            symbol=symbol, direction=direction,
            rsi=rsi_v, atr=atr_v,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Sector B: Stocks Strategy
# ══════════════════════════════════════════════════════════════════════════════

def squeeze_probability_score(
    short_interest: float,
    rvol:           float,
    sentiment_z:    float,
    cfg:            dict,
) -> float:
    """
    Sp = (SI × 0.4) + (Vvelocity × 0.3) + (Ssentiment × 0.3)
    Inputs normalized to 0–1 before weighting.
    """
    si_cap   = cfg.get("si_cap",   0.50)
    rvol_cap = cfg.get("rvol_cap", 10.0)
    si_norm   = min(short_interest / si_cap,        1.0)
    rvol_norm = min(max((rvol - 1.0) / (rvol_cap - 1.0), 0.0), 1.0)
    sent_norm = min(max(sentiment_z / 5.0,          0.0), 1.0)
    return round(
        si_norm   * cfg.get("si_weight",   0.40)
        + rvol_norm * cfg.get("rvol_weight", 0.30)
        + sent_norm * cfg.get("sentiment_weight", 0.30),
        4,
    )


class StocksStrategy:
    """
    Multi-signal stock strategy:
    1. Squeeze Probability Score (short-term catalyst)
    2. Bollinger Squeeze (momentum setup)
    3. RSI divergence (reversal early warning)
    4. EMA trend structure (200/21/9)
    5. ADX trend strength filter
    6. Composite factor score gate (quality × value × momentum)
    """

    def __init__(self, config: dict):
        self.cfg   = config.get("stocks", {})
        self.sq_cfg = self.cfg.get("squeeze", {})

    def evaluate(
        self,
        symbol:         str,
        bars:           dict,
        short_interest: float = 0.10,
        rvol:           float = 1.0,
        sentiment_z:    float = 0.0,
        composite_score: float = 0.50,
    ) -> TechnicalSignals:
        closes  = np.array(bars.get("close",  []), dtype=float)
        highs   = np.array(bars.get("high",   []), dtype=float)
        lows    = np.array(bars.get("low",    []), dtype=float)
        volumes = np.array(bars.get("volume", []), dtype=float)

        if len(closes) < 200:
            return TechnicalSignals(symbol=symbol, direction="HOLD")

        # Squeeze Probability
        sp = squeeze_probability_score(short_interest, rvol, sentiment_z, self.sq_cfg)

        # Technical indicators
        rsi_v  = rsi(closes)
        adx_v  = adx(highs, lows, closes)
        atr_v  = atr_value(highs, lows, closes)
        is_sq  = bollinger_squeeze(closes)
        div    = detect_rsi_divergence(closes, np.array([rsi(closes[:i+1]) for i in range(len(closes))][-14:]))

        ema_9   = ema(closes, 9)[-1]
        ema_21  = ema(closes, 21)[-1]
        ema_200 = ema(closes, 200)[-1]
        price   = closes[-1]

        above_200 = price > ema_200
        ema_bull  = ema_9 > ema_21
        adx_trend = adx_v > self.cfg.get("adx_trending", 25)

        direction = "HOLD"

        # Squeeze play: high Sp + bollinger squeeze + above 200 EMA
        if sp >= self.sq_cfg.get("min_sp_score", 0.60) and is_sq and above_200:
            direction = "BUY"
        # Quality trend: EMA bull + trend strength + good factor score
        elif ema_bull and above_200 and adx_trend and composite_score >= 0.55:
            direction = "BUY"
        # Oversold RSI divergence bounce
        elif div == "bull" and rsi_v < 35 and above_200:
            direction = "BUY"
        # Distribution: bear EMA + overbought + below 200
        elif not ema_bull and not above_200 and rsi_v > 70:
            direction = "SELL"
        # Bear trend
        elif not ema_bull and not above_200 and adx_trend and composite_score < 0.35:
            direction = "SELL"

        return TechnicalSignals(
            symbol=symbol, direction=direction,
            rsi=rsi_v, adx=adx_v, atr=atr_v,
            squeeze_prob=sp,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Sector C: Forex Strategy
# ══════════════════════════════════════════════════════════════════════════════

class ForexStrategy:
    """
    Forex signals combining:
    1. EMA momentum (20/50 EMA with RSI filter)
    2. NLP news bias (provided by NewsAgent)
    3. Carry trade interest rate differential
    4. Session filter (only trade during active sessions)

    Edge: Carry trades have a well-documented positive risk premium.
    High-yield currencies (AUD, NZD) vs low-yield (JPY) show persistent drift.
    """

    def __init__(self, config: dict):
        self.cfg = config.get("forex", {})

    def evaluate(
        self,
        pair:        str,
        bars:        dict,
        news_bias:   float = 0.0,   # -1 to +1, from LLM news sentiment
        rate_diff:   float = 0.0,   # interest rate differential (carry)
        session_ok:  bool  = True,
    ) -> TechnicalSignals:
        closes = np.array(bars.get("close", []), dtype=float)
        highs  = np.array(bars.get("high",  []), dtype=float)
        lows   = np.array(bars.get("low",   []), dtype=float)

        if len(closes) < 50 or (self.cfg.get("session_filter") and not session_ok):
            return TechnicalSignals(symbol=pair, direction="HOLD")

        ema_f = ema(closes, self.cfg.get("ema_fast", 20))[-1]
        ema_s = ema(closes, self.cfg.get("ema_slow", 50))[-1]
        rsi_v = rsi(closes, self.cfg.get("rsi_period", 14))
        atr_v = atr_value(highs, lows, closes)

        ema_bull = ema_f > ema_s
        rsi_bull = rsi_v > self.cfg.get("rsi_momentum_bull", 55)
        rsi_bear = rsi_v < self.cfg.get("rsi_momentum_bear", 45)

        # Composite: EMA direction + RSI confirmation + news bias + carry
        bull_score = (
            (1.0 if ema_bull else 0.0) * 0.40
            + (1.0 if rsi_bull else 0.0) * 0.20
            + max(news_bias, 0.0)  * self.cfg.get("news_bias_weight", 0.40) * 0.30
            + max(rate_diff, 0.0)  * self.cfg.get("carry_weight", 0.20) * 0.10
        )
        bear_score = (
            (1.0 if not ema_bull else 0.0) * 0.40
            + (1.0 if rsi_bear else 0.0)   * 0.20
            + max(-news_bias, 0.0) * self.cfg.get("news_bias_weight", 0.40) * 0.30
            + max(-rate_diff, 0.0) * self.cfg.get("carry_weight", 0.20) * 0.10
        )

        if bull_score >= 0.60:
            direction = "BUY"
        elif bear_score >= 0.60:
            direction = "SELL"
        else:
            direction = "HOLD"

        return TechnicalSignals(symbol=pair, direction=direction, rsi=rsi_v, atr=atr_v)


# ══════════════════════════════════════════════════════════════════════════════
# Sector D: Futures Strategy
# ══════════════════════════════════════════════════════════════════════════════

class FuturesStrategy:
    """
    Futures signals:
    1. Value Area mean reversion (buy VAL, sell VAH)
    2. Anchored VWAP (weekly anchor) as dynamic S/R
    3. COT (Commitment of Traders) positioning — contrarian at extremes
    4. Seasonal patterns (monthly/quarterly tendencies)
    5. Rollover management (close 5 days before expiry)

    Edge: Institutional futures traders rotate positions predictably.
    COT extreme long positioning in commercial hedgers → bullish for underlying.
    """

    def __init__(self, config: dict):
        self.cfg = config.get("futures", {})

    def evaluate(
        self,
        symbol:      str,
        bars:        dict,
        cot_net:     float = 0.0,   # net commercial trader position (normalized -1 to +1)
        days_to_exp: int   = 999,
        seasonal:    float = 0.0,   # seasonal bias score -1 to +1
    ) -> TechnicalSignals:
        closes  = np.array(bars.get("close",  []), dtype=float)
        highs   = np.array(bars.get("high",   []), dtype=float)
        lows    = np.array(bars.get("low",    []), dtype=float)
        volumes = np.array(bars.get("volume", []), dtype=float)

        if len(closes) < 30:
            return TechnicalSignals(symbol=symbol, direction="HOLD")

        # Rollover: close position if too close to expiry
        if days_to_exp <= self.cfg.get("rollover_days_before", 5):
            return TechnicalSignals(symbol=symbol, direction="CLOSE")

        price = closes[-1]
        atr_v = atr_value(highs, lows, closes)
        rsi_v = rsi(closes)
        adx_v = adx(highs, lows, closes)

        # Value Area
        lb    = self.cfg.get("va_lookback_bars", 20)
        vp    = volume_profile(closes[-lb:], volumes[-lb:]) if len(volumes) >= lb else {}
        va_h  = vp.get("vah", price * 1.01)
        va_l  = vp.get("val", price * 0.99)
        poc   = vp.get("poc", price)

        # Anchored VWAP (anchor = start of week = ~5 bars ago)
        anchor = max(0, len(closes) - 5)
        avwap_v = anchored_vwap(highs, lows, closes, volumes, anchor) if len(volumes) > 5 else price

        direction = "HOLD"

        # Mean reversion: price at or below VAL → buy back to POC
        if price <= va_l and price > avwap_v * 0.995 and cot_net > 0:
            direction = "BUY"
        # Price at or above VAH → sell back to POC
        elif price >= va_h and price < avwap_v * 1.005 and cot_net < 0:
            direction = "SELL"
        # AVWAP reclaim (bullish): price crosses above AVWAP with volume
        elif price > avwap_v and adx_v > self.cfg.get("adx_trending", 25) and seasonal > 0.3:
            direction = "BUY"
        # AVWAP loss (bearish)
        elif price < avwap_v and adx_v > self.cfg.get("adx_trending", 25) and seasonal < -0.3:
            direction = "SELL"

        return TechnicalSignals(symbol=symbol, direction=direction, rsi=rsi_v, atr=atr_v)


# ══════════════════════════════════════════════════════════════════════════════
# Combined Sector Engine
# ══════════════════════════════════════════════════════════════════════════════

class SectorStrategyEngine:
    """
    Runs all four sector strategies and pushes TechnicalSignals to
    shared state.signal_queue.  Runs as an asyncio task.
    """

    def __init__(self, state: SharedState, config: dict[str, Any]):
        self.state    = state
        self.config   = config
        self.interval = config.get("intervals", {}).get("strategy_seconds", 15)
        self.crypto   = CryptoStrategy(config)
        self.stocks   = StocksStrategy(config)
        self.forex    = ForexStrategy(config)
        self.futures  = FuturesStrategy(config)

    async def run(self) -> None:
        log.info("SectorStrategyEngine started.")
        while True:
            try:
                await self._evaluate_all()
            except Exception as e:
                log.warning(f"SectorStrategyEngine error: {e}")
            await asyncio.sleep(self.interval)

    async def _evaluate_all(self) -> None:
        if self.state.circuit_open or self.state.news_veto:
            return

        await asyncio.gather(
            self._run_crypto(),
            self._run_stocks(),
            self._run_forex(),
            self._run_futures(),
        )

    async def _run_crypto(self) -> None:
        for sym, bars in self.state.market.crypto.items():
            sig = self.crypto.evaluate(sym, bars)
            if sig.direction != "HOLD":
                await self.state.signal_queue.put(sig)

    async def _run_stocks(self) -> None:
        for sym, bars in self.state.market.stocks.items():
            qsig  = self.state.quant_signals.get(sym, {})
            score = qsig.get("composite_score", 0.50)
            sig   = self.stocks.evaluate(sym, bars, composite_score=score)
            if sig.direction != "HOLD":
                await self.state.signal_queue.put(sig)

    async def _run_forex(self) -> None:
        for pair, bars in self.state.market.forex.items():
            sig = self.forex.evaluate(pair, bars)
            if sig.direction != "HOLD":
                await self.state.signal_queue.put(sig)

    async def _run_futures(self) -> None:
        for sym, bars in self.state.market.futures.items():
            sig = self.futures.evaluate(sym, bars)
            if sig.direction != "HOLD":
                await self.state.signal_queue.put(sig)
