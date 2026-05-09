"""
chimera_v12/utils/ta.py
Technical analysis helpers — all vectorized with numpy for speed.
No TA-Lib dependency required.
"""

from __future__ import annotations
import numpy as np


def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2.0 / (period + 1)
    out   = np.zeros_like(prices, dtype=float)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(prices: np.ndarray, period: int = 14) -> float:
    """RSI — returns current RSI value."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = np.mean(gains[-period:])
    avg_l  = np.mean(losses[-period:])
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return round(100.0 - 100.0 / (1 + rs), 4)


def adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Average Directional Index."""
    if len(closes) < period * 2:
        return 0.0
    tr_arr, pdm_arr, ndm_arr = [], [], []
    for i in range(1, len(closes)):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        tr    = max(h - l, abs(h - pc), abs(l - pc))
        pdm_v = max(highs[i] - highs[i - 1], 0.0)
        ndm_v = max(lows[i - 1] - lows[i],   0.0)
        if pdm_v > ndm_v:
            ndm_v = 0.0
        else:
            pdm_v = 0.0
        tr_arr.append(tr); pdm_arr.append(pdm_v); ndm_arr.append(ndm_v)
    tr_a  = np.array(tr_arr[-period * 2:])
    pdm_a = np.array(pdm_arr[-period * 2:])
    ndm_a = np.array(ndm_arr[-period * 2:])
    atr_v = np.mean(tr_a[-period:])
    if atr_v == 0:
        return 0.0
    pdi = 100 * np.mean(pdm_a[-period:]) / atr_v
    ndi = 100 * np.mean(ndm_a[-period:]) / atr_v
    dx  = 100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) > 0 else 0.0
    return round(dx, 2)


def atr_value(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Average True Range — returns single float."""
    if len(closes) < period + 1:
        return 0.0
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
           for i in range(1, len(closes))]
    return round(float(np.mean(trs[-period:])), 4)


def bollinger_squeeze(closes: np.ndarray, period: int = 20, mult: float = 2.0) -> bool:
    """True if Bollinger Bands are inside Keltner Channels (squeeze condition)."""
    if len(closes) < period:
        return False
    c  = closes[-period:]
    bb = mult * np.std(c)
    kc = mult * np.mean(np.abs(np.diff(c)))
    return bool(bb < kc)


def detect_rsi_divergence(closes: np.ndarray, rsi_arr: np.ndarray, lookback: int = 14) -> str:
    """Detect bullish/bearish RSI divergence. Returns 'bull', 'bear', or 'none'."""
    if len(closes) < lookback or len(rsi_arr) < lookback:
        return "none"
    price_trend = closes[-1] - closes[-lookback]
    rsi_trend   = rsi_arr[-1] - rsi_arr[-lookback]
    if price_trend < 0 and rsi_trend > 0:
        return "bull"
    if price_trend > 0 and rsi_trend < 0:
        return "bear"
    return "none"


def vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
    """Volume-Weighted Average Price."""
    typical = (highs + lows + closes) / 3.0
    total_v = np.sum(volumes)
    if total_v == 0:
        return float(closes[-1])
    return float(np.sum(typical * volumes) / total_v)


def anchored_vwap(
    highs: np.ndarray, lows: np.ndarray,
    closes: np.ndarray, volumes: np.ndarray,
    anchor: int,
) -> float:
    """Anchored VWAP from a specific bar index."""
    return vwap(highs[anchor:], lows[anchor:], closes[anchor:], volumes[anchor:])


def volume_profile(closes: np.ndarray, volumes: np.ndarray, bins: int = 20) -> dict:
    """
    Compute volume profile and identify Value Area High/Low (70% of volume).
    Returns: {poc, vah, val, va_pct}
    """
    if len(closes) < 2:
        return {"poc": closes[-1] if len(closes) else 0, "vah": 0, "val": 0}
    counts, edges = np.histogram(closes, bins=bins, weights=volumes)
    poc_idx = int(np.argmax(counts))
    poc     = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)

    total   = np.sum(counts)
    target  = total * 0.70
    cumvol  = 0.0
    included = [poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx
    while cumvol < target and (lo_idx > 0 or hi_idx < len(counts) - 1):
        lo_add = counts[lo_idx - 1] if lo_idx > 0 else 0
        hi_add = counts[hi_idx + 1] if hi_idx < len(counts) - 1 else 0
        if lo_add >= hi_add and lo_idx > 0:
            lo_idx -= 1; cumvol += lo_add
        elif hi_idx < len(counts) - 1:
            hi_idx += 1; cumvol += hi_add
        else:
            break

    return {
        "poc": poc,
        "vah": float((edges[hi_idx] + edges[hi_idx + 1]) / 2),
        "val": float((edges[lo_idx] + edges[lo_idx + 1]) / 2),
    }
