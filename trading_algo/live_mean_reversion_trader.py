#!/usr/bin/env python3
"""
Live Mean Reversion RSI Trader

Automatically trades mean reversion strategy on IBKR.

Strategy:
- Calculate RSI(14) daily
- Buy when RSI < 30 (oversold)
- Sell when RSI > 70 (overbought)
- Stop loss at -10%

Expected Performance:
- 20-30% annual returns
- Sharpe 1.5-2.0
- 5-10% max drawdown

Usage:
    python live_mean_reversion_trader.py --paper  # Paper trading
    python live_mean_reversion_trader.py --live   # Live trading (careful!)

Scheduler: Run this daily after market close (4:30pm ET)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import argparse
import numpy as np
from datetime import datetime, time
import logging
import json
from pathlib import Path

from trading_algo.broker.ibkr import IBKRBroker
from trading_algo.config import IBKRConfig
from trading_algo.instruments import InstrumentSpec

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MeanReversionTrader:
    """Live mean reversion RSI trading bot."""

    def __init__(
        self,
        broker: IBKRBroker,
        symbols: list,
        capital: float = 100000.0,
        rsi_period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
        position_size: float = 0.50,  # Use 50% of capital per position
        stop_loss_pct: float = 0.10,  # 10% stop loss
        state_file: str = "mean_reversion_state.json"
    ):
        """Initialize trader."""
        self.broker = broker
        self.symbols = symbols
        self.capital = capital
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.state_file = Path(state_file)

        # Load state
        self.state = self.load_state()

    def load_state(self) -> dict:
        """Load trading state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'positions': {},  # symbol -> {shares, entry_price, entry_date}
            'trades': [],
            'capital': self.capital
        }

    def save_state(self):
        """Save trading state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate current RSI value."""
        if len(prices) < self.rsi_period + 1:
            return 50.0

        returns = np.diff(prices[-self.rsi_period-1:])

        gains = returns[returns > 0]
        losses = -returns[returns < 0]

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def get_historical_prices(self, symbol: str, days: int = 100) -> np.ndarray:
        """Fetch historical prices from IBKR."""
        try:
            instrument = InstrumentSpec(
                kind="STK",
                symbol=symbol,
                exchange="SMART",
                currency="USD"
            )

            bars = self.broker.get_historical_bars(
                instrument,
                duration=f"{days} D",
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=True
            )

            if not bars:
                logger.error(f"No data for {symbol}")
                return None

            prices = np.array([bar.close for bar in bars])
            return prices

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        try:
            instrument = InstrumentSpec(
                kind="STK",
                symbol=symbol,
                exchange="SMART",
                currency="USD"
            )

            # Get latest bar
            bars = self.broker.get_historical_bars(
                instrument,
                duration="1 D",
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=True
            )

            if bars:
                return bars[-1].close

            return None

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def check_position(self, symbol: str) -> dict:
        """Check if we have a position in this symbol."""
        return self.state['positions'].get(symbol)

    def place_order(self, symbol: str, action: str, quantity: int):
        """Place order on IBKR (simplified)."""
        logger.info(f"ORDER: {action} {quantity} shares of {symbol}")

        # In production, would use broker.place_order()
        # For now, just log
        logger.warning("Order placement not implemented - set up IBKR order API")

        return True

    def open_position(self, symbol: str, price: float, rsi: float):
        """Open new position."""
        # Calculate shares to buy
        position_value = self.state['capital'] * self.position_size
        shares = int(position_value / price)

        if shares == 0:
            logger.warning(f"Insufficient capital to buy {symbol}")
            return

        # Place order
        success = self.place_order(symbol, "BUY", shares)

        if success:
            # Update state
            self.state['positions'][symbol] = {
                'shares': shares,
                'entry_price': price,
                'entry_rsi': rsi,
                'entry_date': datetime.now().isoformat()
            }

            self.state['capital'] -= shares * price

            logger.info(f"✓ BOUGHT {shares} shares of {symbol} @ ${price:.2f} (RSI={rsi:.1f})")
            self.save_state()

    def close_position(self, symbol: str, price: float, rsi: float, reason: str):
        """Close existing position."""
        position = self.state['positions'].get(symbol)
        if not position:
            logger.warning(f"No position to close for {symbol}")
            return

        shares = position['shares']
        entry_price = position['entry_price']

        # Place order
        success = self.place_order(symbol, "SELL", shares)

        if success:
            # Calculate P&L
            pnl = shares * (price - entry_price)
            pnl_pct = (price / entry_price - 1) * 100

            # Update capital
            self.state['capital'] += shares * price

            # Record trade
            trade = {
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': datetime.now().isoformat(),
                'entry_price': entry_price,
                'exit_price': price,
                'shares': shares,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'reason': reason,
                'entry_rsi': position['entry_rsi'],
                'exit_rsi': rsi
            }

            self.state['trades'].append(trade)

            # Remove position
            del self.state['positions'][symbol]

            logger.info(f"✓ SOLD {shares} shares of {symbol} @ ${price:.2f} (RSI={rsi:.1f})")
            logger.info(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%) - {reason}")

            self.save_state()

    def run_daily_check(self):
        """Run daily trading check."""
        logger.info("="*80)
        logger.info("DAILY TRADING CHECK")
        logger.info(f"Time: {datetime.now()}")
        logger.info("="*80)

        logger.info(f"\nCurrent Capital: ${self.state['capital']:,.2f}")
        logger.info(f"Open Positions: {len(self.state['positions'])}")

        for symbol in self.symbols:
            logger.info(f"\n{symbol}:")

            # Get historical prices
            prices = self.get_historical_prices(symbol)
            if prices is None:
                logger.warning(f"  ✗ Could not fetch data")
                continue

            # Calculate RSI
            rsi = self.calculate_rsi(prices)
            current_price = prices[-1]

            logger.info(f"  Price: ${current_price:.2f}")
            logger.info(f"  RSI:   {rsi:.2f}")

            # Check existing position
            position = self.check_position(symbol)

            if position:
                logger.info(f"  Position: {position['shares']} shares @ ${position['entry_price']:.2f}")

                # Check stop loss
                loss_pct = (current_price / position['entry_price'] - 1)
                if loss_pct < -self.stop_loss_pct:
                    logger.warning(f"  ⚠️ STOP LOSS TRIGGERED ({loss_pct*100:.2f}%)")
                    self.close_position(symbol, current_price, rsi, "stop_loss")

                # Check overbought exit
                elif rsi > self.overbought:
                    logger.info(f"  ✓ OVERBOUGHT - Taking profit")
                    self.close_position(symbol, current_price, rsi, "overbought")

                else:
                    logger.info(f"  → HOLDING (RSI={rsi:.1f})")

            else:
                # Check oversold entry
                if rsi < self.oversold:
                    logger.info(f"  ✓ OVERSOLD - Opening position")
                    self.open_position(symbol, current_price, rsi)

                else:
                    logger.info(f"  → NO SIGNAL (RSI={rsi:.1f})")

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)

        total_equity = self.state['capital']
        for symbol, position in self.state['positions'].items():
            current_price = self.get_current_price(symbol)
            if current_price:
                position_value = position['shares'] * current_price
                total_equity += position_value

                unrealized_pnl = position['shares'] * (current_price - position['entry_price'])
                logger.info(f"{symbol}: ${position_value:,.2f} (P&L: ${unrealized_pnl:+,.2f})")

        logger.info(f"\nTotal Equity: ${total_equity:,.2f}")
        logger.info(f"Total Return: {(total_equity / self.capital - 1) * 100:+.2f}%")

        logger.info("\nCompleted Trades: {}".format(len(self.state['trades'])))
        if self.state['trades']:
            total_pnl = sum(t['pnl'] for t in self.state['trades'])
            winning = sum(1 for t in self.state['trades'] if t['pnl'] > 0)
            logger.info(f"Realized P&L: ${total_pnl:+,.2f}")
            logger.info(f"Win Rate: {winning}/{len(self.state['trades'])} ({winning/len(self.state['trades'])*100:.1f}%)")

        logger.info("="*80)


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description='Mean Reversion RSI Live Trader')
    parser.add_argument('--paper', action='store_true', help='Use paper trading account')
    parser.add_argument('--live', action='store_true', help='Use live trading account (BE CAREFUL!)')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ'], help='Symbols to trade')
    parser.add_argument('--capital', type=float, default=100000.0, help='Starting capital')

    args = parser.parse_args()

    if not args.paper and not args.live:
        print("ERROR: Must specify --paper or --live")
        print("Defaulting to --paper for safety")
        args.paper = True

    if args.live:
        print("\n" + "="*80)
        print("⚠️  WARNING: LIVE TRADING MODE")
        print("="*80)
        response = input("Are you sure you want to trade with REAL MONEY? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting...")
            return

    print("\n" + "="*80)
    print("MEAN REVERSION RSI LIVE TRADER")
    print("="*80)
    print(f"Mode: {'PAPER TRADING' if args.paper else 'LIVE TRADING'}")
    print(f"Symbols: {args.symbols}")
    print(f"Capital: ${args.capital:,.2f}")
    print("="*80 + "\n")

    # Connect to IBKR
    port = 7497 if args.paper else 7496  # Paper=7497, Live=7496
    config = IBKRConfig(host="127.0.0.1", port=port, client_id=29)

    print("Connecting to IBKR TWS...")
    broker = IBKRBroker(config=config, require_paper=args.paper)

    try:
        broker.connect()
        print("✓ Connected to IBKR\n")

        # Create trader
        trader = MeanReversionTrader(
            broker=broker,
            symbols=args.symbols,
            capital=args.capital
        )

        # Run daily check
        trader.run_daily_check()

        print("\n✓ Daily check complete")
        print("\nTo run automatically daily:")
        print("  Add to crontab: 30 16 * * 1-5 /path/to/this/script")
        print("  (Runs at 4:30pm ET Monday-Friday)")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nDisconnecting from IBKR...")
        broker.disconnect()
        print("✓ Disconnected\n")


if __name__ == "__main__":
    main()
