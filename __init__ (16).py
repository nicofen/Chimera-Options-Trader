"""
chimera_v12/mainframe.py
═══════════════════════════════════════════════════════════════════════════════
PROJECT CHIMERA v12 — MAINFRAME

The top-level orchestrator that boots all agents as asyncio tasks.
All agents share a single SharedState instance.
Communication is EXCLUSIVELY through state — agents never call each other.

Agent startup order (dependency-aware):
  1. DataAgent        — populates market data
  2. RegimeClassifier — classifies market regime
  3. NewsAgent        — monitors macro events, can issue veto
  4. SocialScraper    — Stocktwits sentiment Z-score
  5. QuantEdgeAgent   — microstructure + arb signals
  6. WheelEngine      — options wheel scanning
  7. StrategyAgent    — technical signal generation
  8. RiskAgent        — position sizing + Kelly
  9. MasterOrchestrator — LLM debate pipeline + voting
 10. OrderManager     — execution
 11. CircuitBreaker   — risk kill-switch
 12. AlertDispatcher  — Telegram + Discord notifications
 13. APIServer        — WebSocket dashboard
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from chimera_v12.core.state import SharedState
from chimera_v12.config.settings import load_config
from chimera_v12.orchestrator.master import MasterOrchestrator
from chimera_v12.options.wheel_engine import WheelEngine
from chimera_v12.strategies.quant_edge.engine import QuantEdgeAgent
from chimera_v12.utils.logger import setup_logger

log = setup_logger("mainframe")

# Lazy imports from chimera_v11 bridge modules
def _import_chimera_agents(state, config):
    from chimera_v12.agents.bridges.chimera_bridge import (
        DataAgentBridge, NewsAgentBridge, StrategyAgentBridge,
        RiskAgentBridge, OrderManagerBridge, SocialScraperBridge,
        CircuitBreakerBridge, AlertDispatcherBridge, RegimeClassifierBridge,
        APIServerBridge,
    )
    return {
        "data":         DataAgentBridge(state, config),
        "news":         NewsAgentBridge(state, config),
        "strategy":     StrategyAgentBridge(state, config),
        "risk":         RiskAgentBridge(state, config),
        "order":        OrderManagerBridge(state, config),
        "social":       SocialScraperBridge(state, config),
        "circuit":      CircuitBreakerBridge(state, config),
        "alerts":       AlertDispatcherBridge(state, config),
        "regime":       RegimeClassifierBridge(state, config),
        "api":          APIServerBridge(state, config),
    }


class Mainframe:
    """
    Chimera v12 Mainframe.
    Creates one SharedState and starts all agents as concurrent asyncio tasks.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.state  = SharedState()
        self._tasks: list[asyncio.Task] = []

        # Core v12 agents
        self.quant_edge   = QuantEdgeAgent(self.state, self.config)
        self.wheel_engine = WheelEngine(self.state, self.config)
        self.orchestrator = MasterOrchestrator(self.state, self.config)

        # v11 bridged agents (lazy-imported to allow running without full deps)
        try:
            self._legacy = _import_chimera_agents(self.state, self.config)
        except ImportError as e:
            log.warning(f"Legacy chimera agents unavailable (run pip install -r requirements.txt): {e}")
            self._legacy = {}

    async def run(self) -> None:
        log.info("╔══════════════════════════════════════════════╗")
        log.info("║   PROJECT CHIMERA v12 — MAINFRAME BOOT       ║")
        log.info("╚══════════════════════════════════════════════╝")
        log.info(f"  Boot time : {datetime.utcnow().isoformat()}Z")
        log.info(f"  Mode      : {self.config.get('mode', 'paper').upper()}")
        log.info(f"  LLM       : {self.config.get('llm_model', 'gpt-4o')}")
        log.info(f"  Symbols   : {len(self.config.get('stock_symbols_tier1', []))} tier1 + "
                 f"{len(self.config.get('stock_symbols_tier2', []))} tier2")
        log.info(f"  Wheel     : {len(self.config.get('wheel_candidates', []))} candidates")

        # ── Build task list ────────────────────────────────────────────────────
        task_map = {
            "QuantEdgeAgent":       self.quant_edge.run(),
            "WheelEngine":          self.wheel_engine.run(),
            "MasterOrchestrator":   self.orchestrator.run(),
        }

        # Add legacy agents if available
        for name, agent in self._legacy.items():
            if hasattr(agent, "run"):
                task_map[name] = agent.run()

        self._tasks = [
            asyncio.create_task(coro, name=name)
            for name, coro in task_map.items()
        ]

        log.info(f"  Tasks     : {len(self._tasks)} agents running")
        log.info("=" * 50)

        try:
            await asyncio.gather(*self._tasks, return_exceptions=False)
        except asyncio.CancelledError:
            log.warning("Mainframe received shutdown signal.")
        except Exception as e:
            log.exception(f"Fatal mainframe error: {e}")
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        log.info("Shutting down all agents...")
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("All agents halted. Goodbye.")

    def approve_trade(self, cycle_id: str) -> bool:
        """Approve a pending HITL trade from the CLI or API."""
        return self.orchestrator.approve_hitl(cycle_id)

    def reject_trade(self, cycle_id: str) -> bool:
        """Reject a pending HITL trade."""
        return self.orchestrator.reject_hitl(cycle_id)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    mf  = Mainframe(cfg)
    asyncio.run(mf.run())
