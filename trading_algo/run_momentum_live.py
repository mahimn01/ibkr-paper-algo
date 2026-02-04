#!/usr/bin/env python3
"""
Start Live Momentum Trading

Production-ready script to run Pure Momentum strategy live with IBKR.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import logging
from datetime import datetime

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.quant_core.strategies.momentum_live_trader import (
    MomentumLiveTrader
)
from trading_algo.quant_core.strategies.pure_momentum import MomentumConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run live momentum trading."""

    print("="*70)
    print(" "*20 + "PURE MOMENTUM LIVE TRADER")
    print("="*70)

    # Select configuration
    print("\nSelect strategy configuration:")
    print("  1. Conservative (10-15% returns, 10% max DD)")
    print("  2. Moderate     (15-25% returns, 15% max DD) [RECOMMENDED]")
    print("  3. Aggressive   (30-50% returns, 25% max DD)")

    choice = input("\nEnter choice (1-3): ").strip()

    configs = {
        "1": ("Conservative", MomentumConfig(
            fast_ma=20, slow_ma=50, trend_ma=200,
            momentum_lookback=60,
            max_position=0.20,
            target_exposure=1.2,
            vol_scale=True,
            target_vol=0.15,
        )),
        "2": ("Moderate", MomentumConfig(
            fast_ma=10, slow_ma=30, trend_ma=100,
            momentum_lookback=40,
            max_position=0.30,
            target_exposure=1.5,
            vol_scale=True,
            target_vol=0.20,
        )),
        "3": ("Aggressive", MomentumConfig(
            fast_ma=5, slow_ma=20, trend_ma=50,
            momentum_lookback=20,
            max_position=0.40,
            target_exposure=2.0,
            vol_scale=True,
            target_vol=0.30,
        )),
    }

    if choice not in configs:
        print("Invalid choice, using Moderate")
        choice = "2"

    config_name, config = configs[choice]

    print(f"\nSelected: {config_name}")
    print(f"  Target Exposure: {config.target_exposure:.1f}x")
    print(f"  Max Position: {config.max_position:.0%}")
    print(f"  Target Vol: {config.target_vol:.0%}")

    # Select universe
    print("\nSelect trading universe:")
    print("  1. Growth Tech (QQQ, AAPL, MSFT, NVDA, META, GOOGL, AMZN, TSLA)")
    print("  2. Commodities (GLD, SLV, USO, DBA, UNG, PPLT)")
    print("  3. Sector ETFs (SPY, QQQ, XLF, XLE, XLV, XLK, XLI, XLP)")

    univ_choice = input("\nEnter choice (1-3): ").strip()

    universes = {
        "1": ["QQQ", "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA"],
        "2": ["GLD", "SLV", "USO", "DBA", "UNG", "PPLT"],
        "3": ["SPY", "QQQ", "XLF", "XLE", "XLV", "XLK", "XLI", "XLP"],
    }

    universe = universes.get(univ_choice, universes["1"])
    print(f"\nUniverse: {universe}")

    # Connect to IBKR
    print("\nConnecting to IBKR TWS (Paper Trading)...")

    ibkr_config = IBKRConfig(
        host="127.0.0.1",
        port=7497,  # Paper trading
        client_id=20,
    )

    broker = IBKRBroker(config=ibkr_config, require_paper=True)

    try:
        broker.connect()
        print("✓ Connected to IBKR")

        # Get account value
        snapshot = broker.get_account_snapshot()
        equity = snapshot.values.get('NetLiquidation', 100000.0)
        print(f"✓ Account Equity: ${equity:,.2f}")

        # Create trader
        trader = MomentumLiveTrader(
            config=config,
            ibkr_broker=broker,
            universe=universe,
            initial_capital=float(equity),
        )

        print("\n" + "="*70)
        print("STARTING LIVE TRADING")
        print("="*70)
        print("Press Ctrl+C to stop\n")

        # Run one iteration for now (in production, would run continuously)
        iterations = int(input("Number of iterations (1 for test): ").strip() or "1")

        trader.start(iterations=iterations)

        print("\n✓ Trading session complete")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nDisconnecting from IBKR...")
        broker.disconnect()
        print("✓ Disconnected")

    print("\n" + "="*70)
    print("Session ended")
    print("="*70)


if __name__ == "__main__":
    main()
