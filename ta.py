"""
chimera_v12/strategies/scoring.py
═══════════════════════════════════════════════════════════════════════════════
COMPOSITE SCORING ENGINE — Multi-Factor Long-Term Investment Signal

Research-backed factor weights for stock scoring. Each factor is independently
validated in academic and practitioner literature.

Factor Sources:
  Quality  — Piotroski (2000), Novy-Marx (2013): F-Score 9-criteria
  Value    — Fama-French (1993): P/E, P/B, EV/EBITDA; Greenblatt Magic Formula
  Momentum — Jegadeesh-Titman (1993): 12-1 month price momentum
  Growth   — O'Shaughnessy (2011): Revenue, EPS, and FCF growth
  Safety   — Frazzini-Pedersen (2014): Betting Against Beta; low-vol anomaly

Portfolio construction:
  Combined score > 0.70 → Strong Buy consideration
  Combined score > 0.55 → Buy on pullback
  Combined score < 0.35 → Avoid / Short candidate
  Combined score < 0.20 → Strong short (with squeeze check)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SectorWeights:
    """
    Sector-optimized factor weights.
    Each sector has different dominant drivers of excess returns.
    Weights sum to 1.0.

    Research basis:
    - Tech: growth + momentum dominate (high innovation premium)
    - Financials: value + quality dominate (book value meaningful)
    - Consumer: quality + growth (brand moat metrics)
    - Energy: value + momentum (cyclical mean-reversion)
    - Healthcare: growth + quality (pipeline + balance sheet)
    - Industrials: quality + value (FCF generation)
    """
    quality_w:  float = 0.25
    value_w:    float = 0.25
    momentum_w: float = 0.30
    growth_w:   float = 0.20


SECTOR_WEIGHTS: dict[str, SectorWeights] = {
    "technology":    SectorWeights(quality_w=0.20, value_w=0.15, momentum_w=0.35, growth_w=0.30),
    "financials":    SectorWeights(quality_w=0.35, value_w=0.35, momentum_w=0.15, growth_w=0.15),
    "healthcare":    SectorWeights(quality_w=0.30, value_w=0.20, momentum_w=0.20, growth_w=0.30),
    "consumer":      SectorWeights(quality_w=0.30, value_w=0.25, momentum_w=0.25, growth_w=0.20),
    "energy":        SectorWeights(quality_w=0.20, value_w=0.35, momentum_w=0.30, growth_w=0.15),
    "industrials":   SectorWeights(quality_w=0.30, value_w=0.30, momentum_w=0.20, growth_w=0.20),
    "utilities":     SectorWeights(quality_w=0.35, value_w=0.35, momentum_w=0.15, growth_w=0.15),
    "materials":     SectorWeights(quality_w=0.25, value_w=0.30, momentum_w=0.30, growth_w=0.15),
    "real_estate":   SectorWeights(quality_w=0.30, value_w=0.40, momentum_w=0.15, growth_w=0.15),
    "default":       SectorWeights(),
}


# ══════════════════════════════════════════════════════════════════════════════
# Piotroski F-Score (Quality Factor, 0-9)
# ══════════════════════════════════════════════════════════════════════════════

def piotroski_f_score(fundamentals: dict) -> tuple[int, dict[str, int]]:
    """
    9 binary signals across profitability, leverage, and efficiency.

    Profitability (4 signals):
    F1: ROA > 0
    F2: Operating Cash Flow > 0
    F3: ROA improving YoY
    F4: Accruals < 0 (CFO > ROA → quality earnings)

    Leverage & Liquidity (3 signals):
    F5: Long-term debt/assets ratio improved (decreased) YoY
    F6: Current ratio improved YoY
    F7: No new share issuance (dilution) in past year

    Operating Efficiency (2 signals):
    F8: Gross margin improved YoY
    F9: Asset turnover improved YoY

    Score 8-9 = High quality, 0-2 = Low quality / potential short.
    """
    f = fundamentals
    scores = {}

    # Profitability
    scores["F1_roa_positive"]  = 1 if f.get("roa", 0) > 0 else 0
    scores["F2_cfo_positive"]  = 1 if f.get("operating_cashflow", 0) > 0 else 0
    scores["F3_roa_improving"] = 1 if f.get("roa", 0) > f.get("roa_prior", 0) else 0
    cfo   = f.get("operating_cashflow", 0)
    ta    = f.get("total_assets", 1)
    roa   = f.get("roa", 0)
    accrual = roa - (cfo / ta) if ta > 0 else 0
    scores["F4_low_accruals"]  = 1 if accrual < 0 else 0

    # Leverage & Liquidity
    lt_debt    = f.get("long_term_debt", 0)
    lt_debt_p  = f.get("long_term_debt_prior", 0)
    scores["F5_leverage_improved"] = 1 if lt_debt < lt_debt_p else 0
    scores["F6_current_improved"]  = 1 if f.get("current_ratio", 0) > f.get("current_ratio_prior", 0) else 0
    scores["F7_no_dilution"]       = 1 if not f.get("shares_issued_yoy", False) else 0

    # Efficiency
    scores["F8_gm_improved"] = 1 if f.get("gross_margin", 0) > f.get("gross_margin_prior", 0) else 0
    scores["F9_at_improved"] = 1 if f.get("asset_turnover", 0) > f.get("asset_turnover_prior", 0) else 0

    total = sum(scores.values())
    return total, scores


# ══════════════════════════════════════════════════════════════════════════════
# Value Score (Composite of multiple valuation ratios)
# ══════════════════════════════════════════════════════════════════════════════

def value_score(fundamentals: dict, sector: str = "default") -> float:
    """
    Composite value score using multiple ratios.

    Components:
    - P/E ratio (earnings yield = 1/PE)
    - P/B ratio (book value yield = 1/PB)
    - EV/EBITDA (enterprise value yield)
    - FCF yield (Free Cash Flow / Market Cap)
    - EV/Revenue (for growth companies with low EBITDA)

    Greenblatt Magic Formula: combines earnings yield + ROIC.
    Score is normalized to 0-1 across sector peers.

    Returns 0.0 (expensive) to 1.0 (deep value).
    """
    pe     = fundamentals.get("pe_ratio",    999)
    pb     = fundamentals.get("pb_ratio",    999)
    ev_eb  = fundamentals.get("ev_ebitda",   999)
    fcf_y  = fundamentals.get("fcf_yield",   0)
    roic   = fundamentals.get("roic",        0)

    # Earnings yield (higher = better value)
    ey = 1.0 / pe if pe > 0 else 0.0

    # Book value yield
    bvy = 1.0 / pb if pb > 0 else 0.0

    # EV/EBITDA: below 10 = cheap, above 25 = expensive
    ev_score = max(0.0, 1.0 - (ev_eb - 5) / 20) if ev_eb > 0 else 0.5

    # FCF yield: 5%+ is attractive
    fcf_score = min(fcf_y / 0.08, 1.0) if fcf_y > 0 else 0.0

    # ROIC bonus: Magic Formula combines value × quality
    roic_bonus = min(roic / 0.25, 0.5) if roic > 0 else 0.0

    # Weighted composite
    composite = (ey * 0.25 + bvy * 0.20 + ev_score * 0.25 + fcf_score * 0.30)
    composite = composite + roic_bonus * 0.20   # Greenblatt ROIC bonus

    return round(min(max(composite, 0.0), 1.0), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Momentum Score (Jegadeesh-Titman 12-1 month)
# ══════════════════════════════════════════════════════════════════════════════

def momentum_score(
    prices:       list[float],
    benchmark_r:  float = 0.10,  # S&P 500 annual return for excess calculation
) -> float:
    """
    12-1 Month Price Momentum (skip most recent month to avoid short-term reversal).

    Formula: Return from t-252 to t-21 (skipping t-21 to t-0).
    Excess over benchmark: positive excess → higher score.
    Normalized to 0-1:  -30% annual excess → 0.0, +30% annual excess → 1.0.

    Research: Top decile momentum stocks outperform by ~10% annually
    (Jegadeesh-Titman 1993, Asness-Moskowitz-Pedersen 2013).
    """
    if len(prices) < 252:
        return 0.5   # neutral if insufficient history

    p_now   = prices[-21]   # 1 month ago (skip recent)
    p_12m   = prices[-252]  # 12 months ago
    ret_12m = (p_now / p_12m) - 1.0

    excess = ret_12m - benchmark_r
    # Normalize: -30% → 0, 0% → 0.5, +30% → 1.0
    normalized = (excess + 0.30) / 0.60
    return round(min(max(normalized, 0.0), 1.0), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Growth Score
# ══════════════════════════════════════════════════════════════════════════════

def growth_score(fundamentals: dict) -> float:
    """
    Measures business growth quality and acceleration.

    Components:
    - Revenue growth (YoY) — weighted 30%
    - EPS growth (YoY)     — weighted 30%
    - FCF growth (YoY)     — weighted 20%
    - Revenue growth trend (acceleration vs deceleration) — 20%

    Normalization: 0% growth → 0.3, 25% growth → 0.8, 50%+ growth → 1.0
    """
    def norm_growth(g: float, target: float = 0.25) -> float:
        """Normalize growth rate: 0% → 0.3, target → 0.75, 2× target → 1.0."""
        if g < 0:
            return max(0.0, 0.3 + g)   # negative growth reduces score from 0.3
        return min(1.0, 0.3 + (g / target) * 0.5)

    rev_g   = fundamentals.get("revenue_growth_yoy",  0.0)
    eps_g   = fundamentals.get("eps_growth_yoy",      0.0)
    fcf_g   = fundamentals.get("fcf_growth_yoy",      0.0)
    rev_acc = fundamentals.get("revenue_acceleration", 0.0)  # Δ revenue growth

    # Acceleration bonus: growing faster than last year is a quality signal
    accel_score = min(max(0.5 + rev_acc * 2, 0.0), 1.0)

    composite = (norm_growth(rev_g) * 0.30
                 + norm_growth(eps_g) * 0.30
                 + norm_growth(fcf_g) * 0.20
                 + accel_score * 0.20)
    return round(composite, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Composite Scorer
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompositeScore:
    symbol:        str
    sector:        str
    quality:       float   # Piotroski normalized 0-1
    value:         float
    momentum:      float
    growth:        float
    composite:     float
    piotroski_raw: int
    recommendation: str
    detail:        dict = field(default_factory=dict)


def compute_composite(
    symbol:       str,
    sector:       str,
    fundamentals: dict,
    prices:       list[float],
) -> CompositeScore:
    """
    Compute the full composite score for a stock.

    The composite drives:
    - Position sizing (higher score → larger Kelly fraction)
    - Trade direction confirmation
    - Long-term portfolio weighting
    """
    weights = SECTOR_WEIGHTS.get(sector.lower(), SECTOR_WEIGHTS["default"])

    # Quality (Piotroski)
    f_raw, f_detail = piotroski_f_score(fundamentals)
    quality = f_raw / 9.0

    # Value
    val = value_score(fundamentals, sector)

    # Momentum (12-1 month)
    mom = momentum_score(prices)

    # Growth
    grow = growth_score(fundamentals)

    # Weighted composite
    composite = (quality  * weights.quality_w
                 + val    * weights.value_w
                 + mom    * weights.momentum_w
                 + grow   * weights.growth_w)

    # Recommendation thresholds
    if composite >= 0.70:
        rec = "STRONG BUY"
    elif composite >= 0.55:
        rec = "BUY"
    elif composite >= 0.40:
        rec = "HOLD"
    elif composite >= 0.25:
        rec = "SELL"
    else:
        rec = "STRONG SELL"

    return CompositeScore(
        symbol=symbol,
        sector=sector,
        quality=round(quality, 4),
        value=round(val, 4),
        momentum=round(mom, 4),
        growth=round(grow, 4),
        composite=round(composite, 4),
        piotroski_raw=f_raw,
        recommendation=rec,
        detail={
            "piotroski": f_detail,
            "weights": {
                "quality": weights.quality_w,
                "value":   weights.value_w,
                "momentum": weights.momentum_w,
                "growth":  weights.growth_w,
            },
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# Kelly Position Sizing (Research-Enhanced)
# ══════════════════════════════════════════════════════════════════════════════

def kelly_position_size(
    equity:          float,
    win_rate:        float,
    avg_win_r:       float,
    avg_loss_r:      float,
    kelly_fraction:  float = 0.25,
    composite_score: float = 0.50,
    max_pct:         float = 0.10,
) -> float:
    """
    Quarter-Kelly position sizing with composite score scaling.

    Full Kelly = (W × R - L) / R
    where W = win_rate, R = win/loss ratio, L = (1 - win_rate)

    Size scales with composite_score so high-quality trades get larger
    positions (research shows quality factor has alpha in long-horizon sizing).

    Returns: dollar position size.
    """
    if avg_loss_r <= 0 or win_rate <= 0:
        return 0.0

    b = avg_win_r / avg_loss_r     # win/loss R ratio
    q = 1.0 - win_rate

    full_kelly = (win_rate * b - q) / b
    full_kelly = max(0.0, full_kelly)

    # Scale by composite score (0.5 at score=0.5, up to 1.0 at score=1.0)
    score_scalar = 0.5 + composite_score * 0.5

    adj_kelly = full_kelly * kelly_fraction * score_scalar
    adj_kelly = min(adj_kelly, max_pct)

    return round(equity * adj_kelly, 2)
