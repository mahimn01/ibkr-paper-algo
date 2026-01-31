# Orchestrator Trading System

A multi-edge ensemble day trading system that connects to Interactive Brokers (IBKR) for automated trading.

## The Orchestrator Approach

Unlike traditional retail algorithms that use single indicators, the Orchestrator combines **6 independent edge sources** into a unified decision framework:

1. **Market Regime Engine** - Detects trend/range/reversal days
2. **Relative Strength** - Compares stock vs sector vs market
3. **Statistical Extremes** - Identifies z-score extremes
4. **Volume Profile** - Auction theory (value area, POC)
5. **Cross-Asset Confirmation** - Checks related assets
6. **Time-of-Day Patterns** - Opening drive, lunch, power hour

**Trade only when 4+ edges agree. Any edge can veto.**

## Quick Start

```bash
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure (copy .env.example to .env)
cp .env.example .env

# Dry run (always do this first!)
python3 run.py --dry-run

# Live trading with auto-selected stocks
python3 run.py
```

## Usage

```bash
# Auto-select top movers and trade
python3 run.py

# Specify number of stocks
python3 run.py --top 5

# Trade specific symbols
python3 run.py --symbols INTC AMD NVDA

# Dry run mode (paper trade simulation)
python3 run.py --dry-run
```

## Project Structure

```
├── run.py                         # Main entry point
├── trading_algo/
│   ├── orchestrator/              # The Orchestrator strategy
│   │   ├── __init__.py
│   │   ├── strategy.py            # Main Orchestrator class
│   │   ├── types.py               # Enums and dataclasses
│   │   └── edges/                 # Independent edge engines
│   │       ├── market_regime.py
│   │       ├── relative_strength.py
│   │       ├── statistics.py
│   │       ├── volume_profile.py
│   │       ├── cross_asset.py
│   │       └── time_of_day.py
│   ├── broker/                    # Broker implementations
│   │   ├── ibkr.py                # Interactive Brokers
│   │   ├── sim.py                 # Simulation broker
│   │   └── base.py                # Base abstractions
│   ├── stock_selector/            # Stock scanning/selection
│   └── strategies/                # Backward compatibility imports
├── archive/                       # Deprecated code (RAT, Chameleon)
├── tests/                         # Test suite
└── docs/                          # Documentation
```

## Prerequisites

1. Install **Trader Workstation (TWS)** or **IB Gateway**
2. Log into **Paper Trading**
3. Enable API access:
   - TWS: `File -> Global Configuration -> API -> Settings -> Enable ActiveX and Socket Clients`
   - Port: TWS paper `7497`, Gateway paper `4002`

## Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key settings:
- `TRADING_DRY_RUN=true` - Stages orders only (no orders sent)
- `TRADING_LIVE_ENABLED=false` - Blocks sending orders
- `TRADING_ORDER_TOKEN` - Required token for live orders
- `TRADING_DB_PATH` - SQLite audit log path

## CLI Commands

```bash
# Place orders
python3 -m trading_algo.cli place-order --broker ibkr --symbol AAPL --qty 1 --side BUY --type MKT

# Market data snapshot
python3 -m trading_algo.cli snapshot --broker ibkr --kind STK --symbol AAPL

# Historical bars
python3 -m trading_algo.cli history --broker ibkr --kind STK --symbol AAPL --duration "1 D" --bar-size "5 mins"

# Paper smoke test
python3 -m trading_algo.cli --ibkr-port 7497 paper-smoke
```

## Tests

```bash
python3 -m unittest discover -s tests
```

## Documentation

- `docs/ARCHITECTURE.md` - System architecture
- `docs/STRATEGY.md` - Orchestrator strategy details
- `docs/SAFETY.md` - Safety mechanisms
- `docs/WORKFLOWS.md` - Common workflows
- `docs/DB_SCHEMA.md` - Audit database schema

## License

MIT
