# ══════════════════════════════════════════════════════════════════════════════
# Project Chimera v12 — Requirements
# ══════════════════════════════════════════════════════════════════════════════

# ── Core async & typing ────────────────────────────────────────────────────────
asyncio-mqtt>=0.16.2
typing-extensions>=4.9.0
python-dotenv>=1.0.0

# ── Numerical / quant ─────────────────────────────────────────────────────────
numpy>=1.26.0
pandas>=2.1.0
scipy>=1.11.0

# ── LLM / LangGraph (TradingAgents) ──────────────────────────────────────────
openai>=1.30.0
langchain>=0.2.0
langchain-openai>=0.1.0
langgraph>=0.1.0

# ── Broker APIs ───────────────────────────────────────────────────────────────
alpaca-py>=0.19.0
ibapi>=9.81.1            # Interactive Brokers (options / wheel)

# ── Market data ───────────────────────────────────────────────────────────────
polygon-api-client>=1.13.0
finnhub-python>=2.4.19
alpha_vantage>=2.3.1
yfinance>=0.2.38

# ── Options / volatility ──────────────────────────────────────────────────────
# vollib>=1.0.2           # optional: alternative Black-Scholes / Greeks

# ── Social & scraping ─────────────────────────────────────────────────────────
aiohttp>=3.9.0
requests>=2.31.0
beautifulsoup4>=4.12.0

# ── Database / persistence ────────────────────────────────────────────────────
aiosqlite>=0.19.0

# ── Alerts ────────────────────────────────────────────────────────────────────
python-telegram-bot>=21.0
discord-webhook>=1.3.0

# ── API server ────────────────────────────────────────────────────────────────
websockets>=12.0
fastapi>=0.111.0
uvicorn>=0.29.0

# ── Backtesting ───────────────────────────────────────────────────────────────
backtrader>=1.9.78
vectorbt>=0.26.0

# ── Testing ───────────────────────────────────────────────────────────────────
pytest>=8.2.0
pytest-asyncio>=0.23.0
pytest-cov>=5.0.0
