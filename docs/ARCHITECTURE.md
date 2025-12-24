# Architecture

This repo is a **paper-trading-first** IBKR (TWS/IB Gateway) skeleton with:

- A broker abstraction (`trading_algo/broker/base.py`)
- An IBKR broker adapter (`trading_algo/broker/ibkr.py`)
- A deterministic sim broker for tests (`trading_algo/broker/sim.py`)
- An order manager (OMS) for submit/modify/cancel + persistence + recovery (`trading_algo/oms.py`)
- A runner that stitches strategy + risk + OMS together (`trading_algo/autorun.py`)
- A CLI for common workflows (`trading_algo/cli.py`)

## Core flow

1. **Strategy** emits `TradeIntent` objects (`trading_algo/orders.py`)
2. **RiskManager** validates the intent using broker account/position + market data (`trading_algo/risk.py`)
3. **OMS** enforces send-gates (dry-run/live/token) and submits through Broker; persists everything in SQLite if enabled (`trading_algo/oms.py`, `trading_algo/persistence.py`)
4. **AutoRunner** loops, reconciles on startup, and tracks order lifecycle transitions (`trading_algo/autorun.py`)

## Where to add your algo

- Implement a `Strategy` (see `trading_algo/strategy/base.py`) and plug it into `trading_algo/autorun.py` or your own runner.

