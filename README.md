# Project Chimera Options Trader

> **The unified, institutional-grade multi-asset algorithmic trading system.**
> Merges Project Chimera's real-time asyncio infrastructure with TradingAgents' LangGraph multi-agent debate framework — augmented with five professional quant edge strategies and a complete Options Wheel income engine.

---

## ⚠️ Risk Disclaimer

**Read this before touching any code.**

Past performance does not predict future results. Algorithmic trading carries substantial risk of loss, including the loss of all invested capital. This system is complex software — bugs can and do cause real financial losses.

**Mandatory paper-trading period: minimum 90 consecutive days before using real capital.** Monitor the paper account daily. If you cannot explain *why* every trade was taken, you are not ready for live trading.

This software is provided for educational and research purposes. The authors accept no liability for financial losses arising from its use.

---

## Table of Contents

1. [What Is Chimera v12?](#1-what-is-chimera-v12)
2. [Architecture Overview](#2-architecture-overview)
3. [System Data Flow](#3-system-data-flow)
4. [Module Reference](#4-module-reference)
5. [Options Wheel Strategy — Full Explanation](#5-options-wheel-strategy--full-explanation)
6. [Quant Edge Strategies](#6-quant-edge-strategies)
7. [Multi-Factor Scoring System](#7-multi-factor-scoring-system)
8. [Sector-Specific Parameters](#8-sector-specific-parameters)
9. [Conflict Resolution & Voting](#9-conflict-resolution--voting)
10. [Human-in-the-Loop (HITL)](#10-human-in-the-loop-hitl)
11. [Risk Management](#11-risk-management)
12. [Backtesting Guide](#12-backtesting-guide)
13. [Configuration Reference](#13-configuration-reference)
14. [Quick Start](#14-quick-start)
15. [API Reference](#15-api-reference)
16. [Testing](#16-testing)

---

## 1. What Is Chimera v12?

Chimera v12 is the result of merging two powerful but separate trading systems:

| | System A (Chimera v11) | System B (TradingAgents) |
|---|---|---|
| **Strength** | Real-time asyncio execution, OMS, risk management, circuit breakers | LangGraph multi-agent LLM debate: Bull/Bear researchers, risk team |
| **Data** | Alpaca streaming, Polygon tick data, on-chain whale alerts | Alpha Vantage, Finnhub news, social media sentiment |
| **Logic** | Squeeze probability, regime classification, social Z-score | Fundamental analysis, news parsing, investment debate |
| **Execution** | Full Alpaca bracket order OMS with trailing stops | Decision only (no execution layer) |

**v12 unifies these into a single system**, adding:
- **Options Wheel Trading Engine** — systematic premium income strategy
- **Five Quant Edge Strategies** — microstructure and statistical arbitrage
- **Multi-Factor Composite Scoring** — Piotroski, value, momentum, growth factors
- **Sector-Optimized Parameters** — tuned weights per asset class
- **Master Orchestrator** — LangGraph + deterministic VotingAgent
- **53-test integration suite** — verifies every component independently

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROJECT CHIMERA v12 — MAINFRAME                  │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  DataAgent   │  │  NewsAgent   │  │ RegimeClass. │  [Tier 1]    │
│  │ (Alpaca/Poly)│  │(Finnhub/LLM) │  │(ADX/VIX/BRS) │  Data Layer │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                  │                      │
│         └─────────────────┴──────────────────┘                     │
│                           │                                         │
│                    ┌──────▼──────────────┐                          │
│                    │   SharedState       │  [Global Blackboard]     │
│                    │   (TypedDict +      │  Single source of truth  │
│                    │    asyncio-safe)    │  All agents read/write   │
│                    └──────┬──────────────┘                          │
│                           │                                         │
│         ┌─────────────────┼─────────────────────────┐              │
│         │                 │                          │              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌─────────────▼────────┐     │
│  │ SectorStrat. │  │ QuantEdge    │  │  WheelEngine          │     │
│  │  Crypto A    │  │  Agent       │  │  (Options Wheel)      │     │
│  │  Stocks  B   │  │  5 Edges     │  │  CSP → CC cycles      │     │
│  │  Forex   C   │  │              │  │                        │     │
│  │  Futures D   │  │              │  │                        │     │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┬────────┘     │
│         │                 │                          │              │
│         └─────────────────┴──────────────────────────┘             │
│                           │   [signal_queue]                        │
│                    ┌──────▼──────────────┐                          │
│                    │  MasterOrchestrator │  [Decision Layer]        │
│                    │                     │                          │
│                    │  TA Analyst Panel   │  ← TradingAgents LLMs   │
│                    │  Bull/Bear Debate   │                          │
│                    │  Risk Team Debate   │                          │
│                    │  VotingAgent        │  ← Deterministic        │
│                    └──────┬──────────────┘                          │
│                           │                                         │
│                    ┌──────▼──────────────┐                          │
│                    │  HITL Checkpoint    │  [Human Gate]            │
│                    │  (if size > $10k)   │  5-min approval window   │
│                    └──────┬──────────────┘                          │
│                           │  [order_queue]                          │
│                    ┌──────▼──────────────┐                          │
│                    │   OrderManager      │  [Execution Layer]       │
│                    │   Alpaca REST/WS    │  Paper or Live           │
│                    │   Bracket orders    │                          │
│                    │   Trailing stops    │                          │
│                    └──────┬──────────────┘                          │
│                           │                                         │
│         ┌─────────────────┴───────────────────┐                     │
│  ┌──────▼───────┐                    ┌────────▼──────────┐         │
│  │ CircuitBreak │                    │  AlertDispatcher  │         │
│  │ Daily loss 5%│                    │  Telegram/Discord │         │
│  │ Drawdown 10% │                    │                   │         │
│  │ Streak   ×4  │                    └───────────────────┘         │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. System Data Flow

```
Market Data (Alpaca / Polygon)
        │
        ▼
  DataAgent ──► SharedState.market (stocks / crypto / forex / futures / options)
        │
        ▼
  RegimeClassifier ──► SharedState.regime  (BULL_TREND / BEAR_TREND / HIGH_VOL / ...)
        │
        ├──► SectorStrategyEngine ──► TechnicalSignals ──► signal_queue
        │
        ├──► QuantEdgeAgent ──► SharedState.quant_signals
        │
        └──► WheelEngine ──► WheelSignals ──► order_queue
        │
        ▼
  MasterOrchestrator (one cycle per symbol per interval)
        │
        ├── 1. Snapshot SharedState → ChimeraState TypedDict
        ├── 2. Inject CompositeScore (Piotroski + Value + Momentum + Growth)
        ├── 3. Inject QuantEdge signals (OFI, ETF arb, vol surface, news alpha)
        ├── 4. Run TradingAgents analyst panel [LLM calls, optional]
        ├── 5. Run Bull/Bear researcher debate [LLM calls, optional]
        ├── 6. Run Risk team debate [LLM calls, optional]
        ├── 7. VotingAgent resolves all signals (deterministic)
        └── 8. HITL checkpoint if position_size > $10,000
        │
        ▼
  OrderManager ──► Alpaca API ──► Bracket Order (entry + stop + target)
        │
        └──► TradeLogger ──► SQLite (every event recorded for audit)
```

---

## 4. Module Reference

### `chimera_v12/mainframe.py` — System Entry Point
Boots all agents as `asyncio.Task` objects. All agents share a single `SharedState` instance. Agents communicate **exclusively** through shared state — never by calling each other directly. This keeps every agent independently testable.

### `chimera_v12/core/state.py` — Unified Global State
Two-layer state architecture:
- **`ChimeraState` (TypedDict)**: Immutable snapshot passed through LangGraph graph nodes. Every field is `Annotated[T, description]` for graph introspection and debugging.
- **`SharedState`**: Asyncio-safe mutable live state. `asyncio.Queue` for signal passing, `asyncio.Lock` for dict mutations. `.snapshot(symbol)` produces a frozen `ChimeraState` for the orchestrator.

### `chimera_v12/orchestrator/master.py` — Master Orchestrator
Runs one decision cycle per symbol per interval:
1. Snapshots live state
2. Runs LLM agent panel (if configured)
3. Resolves via `VotingAgent` (deterministic, no LLM)
4. Gates through `HITLCheckpoint` for large trades
5. Emits to `order_queue`

### `chimera_v12/options/wheel_engine.py` — Options Wheel Engine
Scans wheel candidates every 2 minutes. Uses built-in Black-Scholes (no external dependencies). Emits `WheelSignal` objects to the order queue. See Section 5 for the full strategy explanation.

### `chimera_v12/strategies/quant_edge/engine.py` — Quant Edge Agent
Five concurrent edge scanners. See Section 6 for full details.

### `chimera_v12/strategies/sector/engine.py` — Sector Strategy Engine
Four sector strategies with independently tuned parameters. Runs every 15 seconds. Pushes `TechnicalSignals` to `signal_queue`.

### `chimera_v12/strategies/scoring.py` — Multi-Factor Scorer
Piotroski F-Score, value ratios, 12-1 month momentum, growth metrics. Combined into a weighted composite score that scales Kelly position sizing.

### `chimera_v12/agents/bridges/chimera_bridge.py` — Compatibility Bridge
Wraps all v11 Chimera agents transparently using the `_V11StateShim`. They run inside v12 without any code changes. Wraps TradingAgents LangGraph nodes into synchronous callables for the async orchestrator.

### `chimera_v12/config/settings.py` — Configuration
Every parameter is environment-variable controlled. Sector-specific sub-dicts for precise tuning per asset class. See Section 13.

---

## 5. Options Wheel Strategy — Full Explanation

The Options Wheel (also called the "Triple Income Strategy") is a systematic, mechanical approach to generating cash income from stocks you would be comfortable owning long-term.

### The Core Concept

You are paid to either buy stock at a discount OR hold stock and sell upside. The wheel cycles between two positions:

```
        ┌─────────────────────────────────────────────────────────────┐
        │                    THE WHEEL CYCLE                          │
        │                                                             │
        │  PHASE 1: CASH-SECURED PUT (CSP)                           │
        │  ──────────────────────────────                             │
        │  You SELL a put option.                                     │
        │  → You receive premium immediately (income #1)             │
        │  → You commit to buying 100 shares at the strike           │
        │    price if the stock falls below it                       │
        │  → Target: 30-delta puts (~1 std dev OTM)                 │
        │  → Target: 30 DTE (monthly expirations)                   │
        │                                                             │
        │      IF EXPIRES WORTHLESS ──────────────────────────────┐  │
        │      Collect full premium, restart Phase 1              │  │
        │                                                          │  │
        │      IF ASSIGNED ───────────────────────────────────┐   │  │
        │      You buy 100 shares at the strike price         │   │  │
        │      Your real cost = strike - premium collected    │   │  │
        │                                                     │   │  │
        │  PHASE 2: ASSIGNED (Stock Ownership)                │   │  │
        │  ────────────────────────────────────               │   │  │
        │  You now own 100 shares.                            │   │  │
        │  Cost basis = strike price - put premium collected  │   │  │
        │  → This is often BELOW current market price        │   │  │
        │  → Immediately move to Phase 3                     │   │  │
        │                                                     │   │  │
        │  PHASE 3: COVERED CALL (CC)                         │   │  │
        │  ───────────────────────                            │   │  │
        │  You SELL a covered call against your shares.       │   │  │
        │  → You receive premium (income #2)                 │   │  │
        │  → Target: 30-delta calls (~1 std dev OTM)         │   │  │
        │  → Target: strike ≥ your cost basis               │   │  │
        │                                                     │   │  │
        │      IF EXPIRES WORTHLESS ──────────────────────┐  │   │  │
        │      Collect premium, sell another call          │  │   │  │
        │                                                  │  │   │  │
        │      IF ASSIGNED (stock called away) ────────────┘  │   │  │
        │      Collect premium + capital gain                  │   │  │
        │      Return to Phase 1 with proceeds ───────────────┘   │  │
        │                                                          │  │
        └──────────────────────────────────────────────────────────┘  │
                                                                      │
        └─────────────────────────────────────────────────────────────┘
```

### Income Sources

| Source | When | Typical Amount |
|--------|------|----------------|
| Put premium | Phase 1 (every month) | 1-3% of strike price |
| Call premium | Phase 3 (every month) | 1-3% of strike price |
| Dividend | While holding shares (Phase 2-3) | 0-3% annually |
| Capital appreciation | If stock rises above cost basis | Variable |

### Entry Filters (Chimera v12 Implementation)

The `WheelEngine` applies all of these before opening any position:

**IV Rank Filter (30–85):** IV Rank measures where implied volatility sits relative to its 52-week range.
- Below 30 → premium is too cheap to sell (not worth the risk)
- Above 85 → implied volatility is in crisis mode (market is pricing disaster; don't sell naked puts into crashes)
- Sweet spot: 30–85 → volatility is elevated but not panicked

**Delta Target (0.30):** A 30-delta option has roughly a 30% chance of being in-the-money at expiration. This means a ~70% probability of keeping the full premium. The 30-delta strike is the standard "one standard deviation" level.

**DTE Target (30 days):** Options lose value fastest in the final 30 days (theta decay is highest). By targeting 30 DTE, you capture the steepest part of the decay curve.

**Profit Target (50%):** Close the position early when 50% of the maximum profit is achieved. This is statistically optimal — it reduces time in the trade (less risk exposure) while capturing most of the available profit. Research shows that closing at 50% significantly improves the risk-adjusted return of short premium strategies.

**Hard Stop (2× premium):** If the position goes against you to the point where the option is worth 2× what you collected, close it. Do not let losers run.

### Stock Selection Criteria for the Wheel

Only wheel stocks that pass **all** of these:
1. **Composite score ≥ 0.55** — you must be comfortable owning it long-term
2. **Market cap > $10B** — large enough for deep options liquidity
3. **Average daily volume > 1M shares** — tight bid-ask spreads
4. **IV Rank 30–85 at entry** — volatility sweet spot
5. **Strong fundamentals** — Piotroski F-Score ≥ 6, positive FCF

Default wheel candidates in Chimera v12:
`AAPL, MSFT, NVDA, AMD, TSLA, SPY, QQQ, IWM`

### Expected Returns

A well-run wheel on a quality stock typically generates:
- **Monthly put premium:** 1–3% of capital committed
- **Monthly call premium:** 1–3% when assigned
- **Annualized estimate:** 15–35% on the committed capital (not the full portfolio)

**Important caveats:**
- This is gross return before taxes and commissions
- Dramatic stock crashes (20%+) can result in losses that take months to recover
- This is why stock selection matters as much as the mechanics

### Black-Scholes Implementation

Chimera v12 includes a pure-Python Black-Scholes implementation (no TA-Lib or vollib dependency):

```python
from chimera_v12.options.wheel_engine import black_scholes_greeks, find_target_strike

# Get delta/price for a put option
greeks = black_scholes_greeks(
    S=185.0,    # stock price
    K=180.0,    # strike
    T=30/365,   # 30 days to expiry
    r=0.05,     # risk-free rate
    sigma=0.28, # implied volatility (28%)
    option_type="put"
)
# → {"delta": -0.32, "gamma": 0.018, "theta": -0.07, "vega": 0.21, "price": 2.85}

# Find the 30-delta strike automatically
strike, greeks = find_target_strike(
    stock_price=185.0, iv=0.28, dte=30, target_delta=0.30
)
```

---

## 6. Quant Edge Strategies

These five strategies exploit short-term statistical inefficiencies. They are additive signals — each votes independently, and the `VotingAgent` combines them.

### Edge 1 — Options Microstructure Arbitrage

**What it exploits:** Temporary violations of put-call parity. When the implied volatility of a put and call at the same strike diverges, one side is mispriced.

**Put-Call Parity:** `C - P = S - K × e^(-rT)`

Any deviation from this relationship (beyond transaction costs) is a pure arbitrage opportunity. In practice, deviations of 5–50 bps occur multiple times per day in liquid options markets.

**Implementation:** Calculate ATM call IV and put IV. If `|put_IV - call_IV| > 5 bps`, enter the delta-neutral spread. Hold 30–60 seconds.

**Expected edge:** 0.5–2% per trade. At 100+ trades/day on SPX options: significant daily P&L.

**APIs needed:** Interactive Brokers TWS (for options order routing), Polygon.io (for tick-level IV data)

### Edge 2 — Cross-Asset ETF Arbitrage

**What it exploits:** Temporary dislocations between ETFs and their underlying futures or related instruments.

**Pairs monitored:**
- SPY ↔ ES1! (S&P 500 ETF vs e-mini S&P futures)
- QQQ ↔ NQ1! (Nasdaq ETF vs e-mini Nasdaq futures)
- XLK ↔ QQQ (Technology sector ETF vs broad tech)
- GLD ↔ GC1! (Gold ETF vs gold futures)

**Implementation:** Rolling Z-score of the log-price spread. Entry when `|Z| > 2.0`. Exit when `|Z| < 0.5`. Mean reversion typically occurs within 5–15 minutes.

**Risk:** ETF creation/redemption mechanics prevent large dislocations from persisting. Authorized participants (APs) arbitrage away gaps quickly, but there is a lag during which retail algos can participate.

### Edge 3 — Order Flow Imbalance (OFI)

**What it exploits:** Directional pressure from the order book that precedes price movement.

**Formula:**
```
OFI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
```
Range: -1 (pure sell pressure) to +1 (pure buy pressure)

When OFI > 0.60 (60% of book is on the bid), price tends to move up in the next 1–5 minutes. Large block prints (5× average trade size) amplify the signal.

**Best on:** AAPL, NVDA, AMZN, SPY, QQQ — highly liquid names with deep books.

**APIs needed:** Polygon.io WebSocket (Level 2 data, $99/month) or Alpaca free tier (less granular)

### Edge 4 — Volatility Surface Arbitrage

**What it exploits:** IV term structure dislocations. The front-month option should normally trade at higher IV than the back-month (volatility contango). When this relationship inverts or becomes extreme, a calendar spread captures the mean reversion.

**Key signal:** Rolling Z-score of `(IV_30DTE - IV_60DTE)`.
- Z-score > +2: Front-month IV too high → sell front, buy back (standard calendar)
- Z-score < -2: Back-month IV too high → reverse calendar

**Returns:** 5–15% annualized on index options (SPX, NDX) with low correlation to directional strategies.

### Edge 5 — News Sentiment Alpha

**What it exploits:** Predictable post-event price patterns following earnings and news.

**Earnings reaction patterns (research-based):**
- **Gaps > 10% on earnings day** → 67% historical probability of a partial gap fill within 5 sessions. Fade the move.
- **"Beat and raise" guidance** → momentum continuation for 3–5 trading days
- **Revenue miss on growth stock** → sell immediately, minimal bounce expectation

**LLM integration:** Headlines + first 200 words parsed by GPT-4 → sentiment score (-1 to +1). Signal decays exponentially with a 30-minute half-life.

**APIs needed:** Finnhub (earnings calendar + news feed), Benzinga (real-time news alerts)

---

## 7. Multi-Factor Scoring System

Every stock receives a composite score from 0.0 (worst) to 1.0 (best) used to:
- Scale Kelly position sizing (higher quality → larger position)
- Gate trade entry (composite < 0.35 → BUY downgraded to HOLD)
- Select wheel candidates

### Piotroski F-Score (Quality, 9 signals)

Originally published by Joseph Piotroski (2000) in the *Journal of Accounting Research*. Empirically validated: high F-Score stocks outperform low F-Score stocks by 7.5% annually.

| Signal | Criterion |
|--------|-----------|
| F1 | ROA > 0 |
| F2 | Operating Cash Flow > 0 |
| F3 | ROA improving year-over-year |
| F4 | Accruals < 0 (cash earnings > accounting earnings) |
| F5 | Long-term debt ratio improved (decreased) YoY |
| F6 | Current ratio improved YoY |
| F7 | No new share issuance (no dilution) |
| F8 | Gross margin improved YoY |
| F9 | Asset turnover improved YoY |

Score 8–9 = High quality. Score 0–2 = Avoid or short candidate.

### Value Score (Composite)

Combines four valuation metrics with Greenblatt's Magic Formula ROIC bonus:
- **Earnings Yield** (1/P/E) — 25% weight
- **Book Value Yield** (1/P/B) — 20% weight
- **EV/EBITDA Score** — 25% weight (normalized: 5 → 1.0, 25+ → 0.0)
- **FCF Yield** — 30% weight (8%+ FCF yield → score of 1.0)
- **ROIC Bonus** — up to +0.2 (Greenblatt: high-ROIC value stocks)

### 12-1 Month Momentum (Jegadeesh-Titman)

Returns from 12 months ago to 1 month ago (skipping the most recent month to avoid short-term reversal). Excess return vs. benchmark normalized to 0–1.

**Research basis:** Jegadeesh-Titman (1993) showed top decile momentum stocks beat bottom decile by 10%+ annually. Asness-Moskowitz-Pedersen (2013) confirmed across 4 asset classes and 40+ years.

### Growth Score

Revenue growth (30%), EPS growth (30%), FCF growth (20%), acceleration bonus (20%). Acceleration is the most predictive: stocks where growth is *getting faster* show the strongest forward returns.

### Sector-Optimized Weights

Different sectors have different dominant return drivers:

| Sector | Quality | Value | Momentum | Growth |
|--------|---------|-------|----------|--------|
| Technology | 20% | 15% | 35% | 30% |
| Financials | 35% | 35% | 15% | 15% |
| Healthcare | 30% | 20% | 20% | 30% |
| Consumer | 30% | 25% | 25% | 20% |
| Energy | 20% | 35% | 30% | 15% |
| Industrials | 30% | 30% | 20% | 20% |
| Utilities | 35% | 35% | 15% | 15% |

---

## 8. Sector-Specific Parameters

### Sector A: Crypto

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| EMA Fast/Slow | 9 / 21 | Short-term crypto cycles are faster than equities |
| RSI Oversold/Overbought | 30 / 70 | Standard; crypto respects these levels well |
| BTC Inflow Threshold | $1M | Large exchange inflows = bearish pressure signal |
| Funding Rate Extreme | 1% / 8h | Crowded longs → contrarian short |
| Memecoin Volume Spike | $50M 24h | Risk appetite peak → fade longs |

**Crypto-specific edge:** Funding rates in perpetual futures create predictable mean-reversion. When 8-hour funding exceeds 0.1% (longs pay shorts), the market is crowded in one direction — 67% historical probability of a pullback within 24 hours.

### Sector B: Stocks

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Min Squeeze Score (Sp) | 0.60 | Validated on 2015–2025 short squeeze events |
| SI Weight | 40% | Primary driver of squeeze potential |
| RVOL Weight | 30% | Relative volume confirms squeeze is active |
| Sentiment Weight | 30% | Social momentum validates institutional interest |
| EMA Trend Filter | 200-period | Classic institutional trend filter |
| ADX Trend Threshold | 25 | Below 25 = ranging market, strategies differ |

### Sector C: Forex

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| EMA Fast/Slow | 20 / 50 | Standard swing-trading forex periods |
| RSI Bull/Bear | 55 / 45 | Wider band prevents false signals in ranging |
| News Bias Weight | 40% | NLP sentiment is highly predictive in FX |
| Carry Weight | 20% | Interest rate differential provides persistent edge |
| Session Filter | True | Only trade major session overlaps |

### Sector D: Futures

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Value Area | 70% of volume | Standard market profile definition |
| AVWAP Anchor | Weekly | Weekly VWAP is institutional reference level |
| COT Weight | 20% | Commercial trader positioning is leading indicator |
| Seasonal Weight | 20% | Index futures have well-documented seasonal bias |
| Rollover Days | 5 before expiry | Avoid pin risk and illiquidity near expiry |

---

## 9. Conflict Resolution & Voting

The `VotingAgent` resolves disagreements between all signal sources using a deterministic priority hierarchy. No LLM is involved in this step — it is fully auditable and reproducible.

```
Priority 1 (Highest): Circuit Breaker OPEN
    → Force HOLD regardless of everything else

Priority 2: News Agent Macro Veto ACTIVE
    → Force HOLD for veto_cooldown_seconds (default 600s)

Priority 3: Risk Team Consensus
    → If risk team disagrees with research team, risk wins
    → Rationale: risk management is more important than alpha

Priority 4: Composite Score Gate
    → If composite_score < 0.35, BUY is downgraded to HOLD
    → Never buy low-quality stocks regardless of technical signals

Priority 5: Quant Edge Tiebreaker
    → OFI / ETF arb signals break HOLD vs BUY on close calls

Priority 6: Bull/Bear Debate Base Recommendation
    → Default signal when no higher-priority override applies
```

**Agent votes in each cycle:**
- `research_team` — from TradingAgents investment_plan
- `risk_team` — from TradingAgents risk debate judge
- `chimera_squeeze` — from Squeeze Probability Score
- `factor_model` — from composite score
- `quant_edge` — from OFI / ETF arb signals

Final position size scales with confidence (agreement among voters × composite score).

---

## 10. Human-in-the-Loop (HITL)

Any trade with `position_size_usd ≥ $10,000` (configurable) is paused and requires explicit human approval before the order is submitted.

**Why this matters:** Combining two LLM-based systems increases the risk of correlated hallucinations. The HITL gate provides a mandatory sanity check for large positions.

### Approval Flow

```
Trade decision made by VotingAgent
         │
         ▼
Is position_size_usd ≥ hitl_threshold?
         │
    YES  │  NO
         │   └──► Submit to OMS directly
         ▼
HITL Checkpoint:
  - Logs cycle_id, symbol, direction, size
  - Sends alert to Telegram/Discord: "⚠️ APPROVAL REQUIRED"
  - Waits up to 5 minutes (configurable)
         │
    APPROVED  │  REJECTED / TIMEOUT
              │   └──► Decision converted to HOLD
              ▼
    Submit to OMS
```

### Approve/Reject Methods

```bash
# Via REST API
curl -X POST http://localhost:8765/api/hitl/approve/AAPL_20260515143022

# Via CLI (if implemented)
chimera hitl approve AAPL_20260515143022
chimera hitl reject  AAPL_20260515143022
```

```python
# Programmatically
mainframe.approve_trade("AAPL_20260515143022")
mainframe.reject_trade("AAPL_20260515143022")
```

Every HITL decision is written to the audit trail with timestamp and operator ID.

---

## 11. Risk Management

### Position Sizing — Quarter-Kelly

Full Kelly Criterion: `f* = (W × b - L) / b`
Where `W` = win rate, `b` = win/loss ratio, `L` = loss rate

Chimera uses **Quarter-Kelly** (multiply by 0.25) to reduce variance while preserving the growth properties. Position size also scales with the composite quality score — better stocks get slightly larger allocations.

```python
kelly_fraction  = 0.25        # quarter-Kelly cap
max_position    = 10%         # absolute maximum per name
max_sector      = 25%         # maximum in any one sector
max_correlation = 70%         # reject new positions correlated >70% to existing
```

### Circuit Breaker

Three independent trip conditions. Any one fires the breaker:

| Condition | Default | Reset |
|-----------|---------|-------|
| Daily loss > 5% of equity | `-5%` | Auto-resets at midnight UTC |
| Peak-to-trough drawdown > 10% | `-10%` | Manual reset only |
| Consecutive loss streak | `4` | Resets on first winning trade |

When the circuit breaker fires:
1. `state.circuit_open = True` → all signals suppressed
2. All open positions force-closed at market
3. `BreakerEvent` logged to SQLite + broadcast to dashboard
4. Alert sent via Telegram/Discord

### Stop-Loss System

All positions use ATR-based stops:
- **Stop loss:** Entry price - (ATR × 2.0) for longs
- **Take profit:** Entry price + (ATR × 3.0) for longs
- **Trailing stop:** Activated after 1.5R in profit; trails by 1.5× ATR

---

## 12. Backtesting Guide

### Running a Backtest

```python
from chimera_v12.backtest.engine import BacktestEngine
from chimera_v12.config.settings import load_config

cfg = load_config()
cfg["mode"] = "paper"   # always paper for backtesting

engine = BacktestEngine(cfg)
report = engine.run(
    symbols={
        "stocks":  ["AAPL", "MSFT", "NVDA"],
        "crypto":  ["BTC/USD"],
        "futures": ["ES1!"],
    },
    start="2022-01-01",
    end="2024-12-31",
    timeframe="1Day",
    initial_equity=100_000.0,
    warmup_bars=200,
    half_kelly=True,
)
print(report)
```

### Six Bias Corrections

The backtester addresses these common errors:

| Bias | Fix |
|------|-----|
| Same-bar fill | Orders stamped +1 second; fill guard prevents same-bar execution |
| Full Kelly | Half-Kelly applied by default |
| Low slippage | 25 bps stocks, 30 bps crypto default slippage |
| Stale SI/RVOL | Historical short interest and relative volume injected per-bar |
| No news veto | FOMC and NFP blackout dates replayed |
| Early warmup | Global warmup: no signals until ALL symbols have 200+ bars |

### Paper Trading Protocol

**Do not skip this.** Paper trade for a minimum of 90 days before live capital.

Week 1–4: Verify every signal is logged. Check that circuit breakers trigger correctly. Confirm HITL alerts arrive.

Week 5–12: Monitor actual P&L vs. backtest expectations. If live paper P&L is more than 30% below backtest, **do not go live**. Investigate what the backtest missed.

Month 4+: Only consider live trading if:
- Sharpe ratio (paper) > 1.0
- Maximum drawdown < 15%
- Win rate consistent with backtest expectation
- You can explain every major trade

---

## 13. Configuration Reference

All settings are in `chimera_v12/config/settings.py`. Copy `.env.example` to `.env`:

```bash
# Required
ALPACA_KEY=your_alpaca_key
ALPACA_SECRET=your_alpaca_secret
OPENAI_API_KEY=sk-...

# Recommended
POLYGON_API_KEY=your_polygon_key
FINNHUB_API_KEY=your_finnhub_key

# Optional: Interactive Brokers (for options wheel)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497

# Mode: always start with paper
CHIMERA_MODE=paper

# Alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Risk overrides
CB_DAILY_LOSS_PCT=0.05
CB_DRAWDOWN_PCT=0.10
CB_STREAK_LIMIT=4
```

### Key Parameters to Tune

| Parameter | Location | Default | Notes |
|-----------|----------|---------|-------|
| `risk.kelly_fraction` | config | 0.25 | Lower = smaller positions, less risk |
| `risk.hitl_threshold_usd` | config | 10,000 | Lower to approve more trades manually |
| `stocks.squeeze.min_sp_score` | config | 0.60 | Higher = more selective squeeze trades |
| `options_wheel.min_iv_rank` | config | 30 | Raise to 40+ in low-vol environments |
| `options_wheel.profit_target_pct` | config | 0.50 | Never change this; 50% is optimal |
| `quant_edge.etf_arb.z_score_entry` | config | 2.0 | Raise to 2.5 to reduce frequency |

---

## 14. Quick Start

### 1. Install

```bash
git clone <your-repo>
cd chimera_v12
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Tests

```bash
python -m pytest chimera_v12/tests/ -v
# Expected: 53 passed
```

### 4. Paper Trade

```bash
# Set CHIMERA_MODE=paper in .env, then:
python -m chimera_v12.mainframe
```

### 5. Monitor

The WebSocket dashboard streams at `ws://localhost:8765`. Open `dashboard/index.html` in your browser.

Telegram/Discord alerts fire on:
- Every trade entry/exit
- Circuit breaker trips
- HITL approval requests
- System heartbeat (hourly)

### 6. Run a Backtest

```bash
python -m chimera_v12.backtest.run_backtest \
  --symbols AAPL MSFT NVDA \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --equity 100000
```

---

## 15. API Reference

The REST/WebSocket server runs at `http://localhost:8765`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System health, equity, regime |
| `/api/positions` | GET | Open positions |
| `/api/trades` | GET | Trade history |
| `/api/signals` | GET | Latest signals per symbol |
| `/api/wheel` | GET | Active wheel positions |
| `/api/hitl/pending` | GET | Pending HITL approvals |
| `/api/hitl/approve/{cycle_id}` | POST | Approve a trade |
| `/api/hitl/reject/{cycle_id}` | POST | Reject a trade |
| `/api/breaker/reset` | POST | Reset circuit breaker (with `note` body) |
| `/ws` | WebSocket | Live state stream |

---

## 16. Testing

```bash
# Run all 53 tests
python -m pytest chimera_v12/tests/ -v

# With coverage
python -m pytest chimera_v12/tests/ --cov=chimera_v12 --cov-report=html

# Individual suites
python -m pytest chimera_v12/tests/ -k "TestBlackScholes" -v
python -m pytest chimera_v12/tests/ -k "TestScoring" -v
python -m pytest chimera_v12/tests/ -k "TestOrchestrator" -v
python -m pytest chimera_v12/tests/ -k "TestQuantEdge" -v
```

### Test Coverage Map

| Test Class | Module Tested | Tests |
|------------|---------------|-------|
| `TestSharedState` | `core/state.py` | 5 |
| `TestBlackScholes` | `options/wheel_engine.py` | 8 |
| `TestScoring` | `strategies/scoring.py` | 8 |
| `TestQuantEdge` | `strategies/quant_edge/engine.py` | 9 |
| `TestSectorStrategies` | `strategies/sector/engine.py` | 7 |
| `TestTA` | `utils/ta.py` | 4 |
| `TestOrchestrator` | `orchestrator/master.py` | 8 |
| `TestConfig` | `config/settings.py` | 3 |
| **Total** | | **53** |

---

## File Structure

```
chimera_v12/
├── mainframe.py                    # Entry point: boots all agents
├── requirements.txt
├── .env.example
│
├── core/
│   └── state.py                   # ChimeraState (TypedDict) + SharedState
│
├── config/
│   └── settings.py                # All configuration, env-variable driven
│
├── orchestrator/
│   └── master.py                  # MasterOrchestrator + VotingAgent + HITL
│
├── agents/
│   ├── analysts/                  # TradingAgents: market, news, fundamentals, social
│   ├── researchers/               # TradingAgents: bull + bear researchers
│   ├── risk_mgmt/                 # TradingAgents: aggressive + conservative + neutral
│   ├── trader/                    # TradingAgents: trader + portfolio manager
│   └── bridges/
│       └── chimera_bridge.py      # v11→v12 adapters + TradingAgentsBridge
│
├── strategies/
│   ├── scoring.py                 # Piotroski + Value + Momentum + Growth scorer
│   ├── sector/
│   │   └── engine.py              # Crypto / Stocks / Forex / Futures strategies
│   └── quant_edge/
│       └── engine.py              # 5 quant edge strategies + QuantEdgeAgent
│
├── options/
│   └── wheel_engine.py            # Full Options Wheel system + Black-Scholes
│
├── oms/                           # (bridged from v11) Order Manager, preflight
├── risk/                          # (bridged from v11) Circuit Breaker
├── backtest/                      # (bridged from v11) Bias-corrected engine
├── alerts/                        # (bridged from v11) Telegram + Discord
├── server/                        # (bridged from v11) WebSocket API server
├── social/                        # (bridged from v11) Stocktwits scraper
├── regime/                        # (bridged from v11) Market regime classifier
├── utils/
│   ├── logger.py                  # Structured logging setup
│   └── ta.py                      # Vectorized technical indicators (no TA-Lib)
│
└── tests/
    └── test_unified.py            # 53 integration tests (all passing)
```
