"""
Pairs Trading Strategy (Statistical Arbitrage)

Academic Foundation:
- Renaissance Technologies core strategy
- Sharpe ratio: 1.9-2.4 (academic studies)
- Win rate: 55-65%

Strategy:
1. Find cointegrated pairs (stocks that move together)
2. Calculate spread: Spread = Stock_A - β * Stock_B
3. Calculate Z-score: Z = (Spread - Mean) / StdDev
4. Entry: |Z| > 2.0 (spread diverges)
5. Exit: Z returns to 0 or crosses

Mathematical Basis:
- Ornstein-Uhlenbeck mean reversion process
- Cointegration tests (Engle-Granger)
- Error correction models

Reference:
- Vidyamurthy (2004) "Pairs Trading: Quantitative Methods and Analysis"
- Gatev, Goetzmann, Rouwenhorst (2006)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy import stats
from collections import deque


@dataclass
class PairConfig:
    """Configuration for pairs trading."""
    lookback_period: int = 60  # Days for calculating mean/std
    entry_threshold: float = 2.0  # Z-score for entry
    exit_threshold: float = 0.5  # Z-score for exit
    stop_loss_threshold: float = 4.0  # Z-score stop loss
    position_size: float = 0.10  # 10% of capital per pair
    min_correlation: float = 0.75  # Minimum correlation to trade


@dataclass
class PairPosition:
    """Active pair position."""
    pair_name: str
    stock_a: str
    stock_b: str
    beta: float
    entry_spread: float
    entry_zscore: float
    entry_date: datetime
    quantity_a: float
    quantity_b: float
    entry_price_a: float
    entry_price_b: float
    side: str  # 'long_spread' or 'short_spread'


class PairsTradingStrategy:
    """
    Statistical arbitrage through pairs trading.

    Identifies cointegrated pairs and trades mean reversion of the spread.
    """

    def __init__(
        self,
        config: PairConfig = None,
        initial_capital: float = 100000.0
    ):
        """Initialize pairs trading strategy."""
        self.config = config or PairConfig()
        self.initial_capital = initial_capital
        self.capital = initial_capital

        # Positions
        self.positions: List[PairPosition] = []

        # Performance tracking
        self.equity_curve = [initial_capital]
        self.dates = []
        self.trades = []

        # Spread history for each pair
        self.spread_history: Dict[str, deque] = {}

    def calculate_correlation(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray
    ) -> float:
        """Calculate correlation between two price series."""
        if len(prices_a) < 20 or len(prices_b) < 20:
            return 0.0

        returns_a = np.diff(prices_a) / prices_a[:-1]
        returns_b = np.diff(prices_b) / prices_b[:-1]

        # Align lengths (symbols may have different bar counts)
        min_len = min(len(returns_a), len(returns_b))
        returns_a = returns_a[-min_len:]
        returns_b = returns_b[-min_len:]

        correlation = np.corrcoef(returns_a, returns_b)[0, 1]
        return float(correlation)

    def calculate_beta(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray
    ) -> float:
        """
        Calculate hedge ratio (beta) between two stocks.

        Uses linear regression: Price_A = α + β * Price_B
        """
        if len(prices_a) < 2 or len(prices_b) < 2:
            return 1.0

        # Align lengths and apply lookback
        min_len = min(len(prices_a), len(prices_b), self.config.lookback_period)
        pa = prices_a[-min_len:]
        pb = prices_b[-min_len:]

        # Guard: constant prices cause singular regression
        if np.ptp(pb) == 0:
            return 1.0

        slope, intercept, r_value, p_value, std_err = stats.linregress(pb, pa)

        return float(slope)

    def calculate_spread(
        self,
        price_a: float,
        price_b: float,
        beta: float
    ) -> float:
        """Calculate current spread."""
        return price_a - beta * price_b

    def calculate_zscore(
        self,
        current_spread: float,
        spread_history: np.ndarray
    ) -> float:
        """Calculate Z-score of current spread."""
        if len(spread_history) < 10:
            return 0.0

        mean_spread = np.mean(spread_history)
        std_spread = np.std(spread_history)

        if std_spread < 1e-6:
            return 0.0

        zscore = (current_spread - mean_spread) / std_spread
        return float(zscore)

    def find_tradeable_pairs(
        self,
        symbols: List[str],
        price_data: Dict[str, np.ndarray],
        current_date: datetime
    ) -> List[Tuple[str, str, float, float]]:
        """
        Find pairs suitable for trading.

        Returns:
            List of (stock_a, stock_b, correlation, beta)
        """
        tradeable_pairs = []

        # Check all possible pairs
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i+1:]:
                if sym_a == sym_b:
                    continue

                prices_a = price_data.get(sym_a)
                prices_b = price_data.get(sym_b)

                if prices_a is None or prices_b is None:
                    continue

                if len(prices_a) < self.config.lookback_period or len(prices_b) < self.config.lookback_period:
                    continue

                # Calculate correlation
                correlation = self.calculate_correlation(prices_a, prices_b)

                if correlation < self.config.min_correlation:
                    continue

                # Calculate beta (hedge ratio)
                beta = self.calculate_beta(prices_a, prices_b)

                tradeable_pairs.append((sym_a, sym_b, correlation, beta))

        return tradeable_pairs

    def generate_signals(
        self,
        symbols: List[str],
        price_data: Dict[str, np.ndarray],
        current_prices: Dict[str, float],
        current_date: datetime
    ) -> List[Dict]:
        """
        Generate pair trading signals.

        Returns:
            List of signal dictionaries
        """
        signals = []

        # Find tradeable pairs
        pairs = self.find_tradeable_pairs(symbols, price_data, current_date)

        for stock_a, stock_b, correlation, beta in pairs:
            pair_name = f"{stock_a}_{stock_b}"

            # Calculate current spread
            price_a = current_prices.get(stock_a)
            price_b = current_prices.get(stock_b)

            if price_a is None or price_b is None:
                continue

            current_spread = self.calculate_spread(price_a, price_b, beta)

            # Update spread history
            if pair_name not in self.spread_history:
                self.spread_history[pair_name] = deque(maxlen=self.config.lookback_period)

            # Calculate historical spreads
            prices_a = price_data[stock_a]
            prices_b = price_data[stock_b]
            n = min(len(prices_a), len(prices_b))

            historical_spreads = []
            for i in range(max(0, n - self.config.lookback_period), n):
                spread = self.calculate_spread(prices_a[i], prices_b[i], beta)
                historical_spreads.append(spread)

            if len(historical_spreads) < 10:
                continue

            # Calculate Z-score
            zscore = self.calculate_zscore(current_spread, np.array(historical_spreads))

            # Update spread history
            self.spread_history[pair_name].append(current_spread)

            # Check if we already have position in this pair
            existing = any(p.pair_name == pair_name for p in self.positions)

            if not existing:
                # Entry signals
                if zscore > self.config.entry_threshold:
                    # Spread too high - SHORT spread (short A, long B)
                    signals.append({
                        'action': 'short_spread',
                        'pair_name': pair_name,
                        'stock_a': stock_a,
                        'stock_b': stock_b,
                        'beta': beta,
                        'zscore': zscore,
                        'spread': current_spread,
                        'price_a': price_a,
                        'price_b': price_b
                    })

                elif zscore < -self.config.entry_threshold:
                    # Spread too low - LONG spread (long A, short B)
                    signals.append({
                        'action': 'long_spread',
                        'pair_name': pair_name,
                        'stock_a': stock_a,
                        'stock_b': stock_b,
                        'beta': beta,
                        'zscore': zscore,
                        'spread': current_spread,
                        'price_a': price_a,
                        'price_b': price_b
                    })

        return signals

    def open_position(
        self,
        signal: Dict,
        current_date: datetime
    ):
        """Open a new pair position."""
        # Calculate position sizes
        position_value = self.capital * self.config.position_size

        price_a = signal['price_a']
        price_b = signal['price_b']
        beta = signal['beta']

        # For long spread: long A, short B
        # For short spread: short A, long B

        if signal['action'] == 'long_spread':
            # Long stock A
            quantity_a = position_value / price_a
            # Short stock B (hedge)
            quantity_b = -beta * quantity_a
        else:  # short_spread
            # Short stock A
            quantity_a = -position_value / price_a
            # Long stock B (hedge)
            quantity_b = -beta * quantity_a

        position = PairPosition(
            pair_name=signal['pair_name'],
            stock_a=signal['stock_a'],
            stock_b=signal['stock_b'],
            beta=beta,
            entry_spread=signal['spread'],
            entry_zscore=signal['zscore'],
            entry_date=current_date,
            quantity_a=quantity_a,
            quantity_b=quantity_b,
            entry_price_a=price_a,
            entry_price_b=price_b,
            side=signal['action']
        )

        # Update capital (simplified - assumes can short without restrictions)
        cost_a = abs(quantity_a) * price_a
        cost_b = abs(quantity_b) * price_b

        self.positions.append(position)

    def close_position(
        self,
        position: PairPosition,
        current_price_a: float,
        current_price_b: float,
        current_date: datetime,
        reason: str = 'exit'
    ):
        """Close a pair position."""
        # Calculate P&L
        if position.side == 'long_spread':
            # Long A, short B
            pnl_a = position.quantity_a * (current_price_a - position.entry_price_a)
            pnl_b = position.quantity_b * (current_price_b - position.entry_price_b)
        else:
            # Short A, long B
            pnl_a = position.quantity_a * (current_price_a - position.entry_price_a)
            pnl_b = position.quantity_b * (current_price_b - position.entry_price_b)

        total_pnl = pnl_a + pnl_b

        # Update capital
        self.capital += total_pnl

        # Record trade
        self.trades.append({
            'pair': position.pair_name,
            'entry_date': position.entry_date,
            'exit_date': current_date,
            'entry_zscore': position.entry_zscore,
            'pnl': total_pnl,
            'reason': reason,
            'holding_days': (current_date - position.entry_date).days
        })

        # Remove position
        self.positions.remove(position)

    def update_positions(
        self,
        current_prices: Dict[str, float],
        price_data: Dict[str, np.ndarray],
        current_date: datetime
    ):
        """Update all positions and check for exits."""
        positions_to_close = []

        for position in self.positions:
            price_a = current_prices.get(position.stock_a)
            price_b = current_prices.get(position.stock_b)

            if price_a is None or price_b is None:
                continue

            # Calculate current spread and Z-score
            current_spread = self.calculate_spread(price_a, price_b, position.beta)

            # Get historical spreads
            pair_name = position.pair_name
            if pair_name in self.spread_history and len(self.spread_history[pair_name]) > 0:
                historical = np.array(self.spread_history[pair_name])
                current_zscore = self.calculate_zscore(current_spread, historical)

                # Exit conditions
                if abs(current_zscore) < self.config.exit_threshold:
                    # Spread has reverted to mean
                    positions_to_close.append((position, price_a, price_b, 'mean_reversion'))

                elif abs(current_zscore) > self.config.stop_loss_threshold:
                    # Stop loss - spread diverged too much
                    positions_to_close.append((position, price_a, price_b, 'stop_loss'))

                elif (current_date - position.entry_date).days > 30:
                    # Max holding period
                    positions_to_close.append((position, price_a, price_b, 'max_holding'))

        # Close positions
        for position, price_a, price_b, reason in positions_to_close:
            self.close_position(position, price_a, price_b, current_date, reason)

        # Update equity curve
        unrealized_pnl = 0
        for position in self.positions:
            price_a = current_prices.get(position.stock_a, position.entry_price_a)
            price_b = current_prices.get(position.stock_b, position.entry_price_b)

            pnl_a = position.quantity_a * (price_a - position.entry_price_a)
            pnl_b = position.quantity_b * (price_b - position.entry_price_b)
            unrealized_pnl += pnl_a + pnl_b

        equity = self.capital + unrealized_pnl
        self.equity_curve.append(equity)
        self.dates.append(current_date)

    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics."""
        if len(self.equity_curve) < 2:
            return {}

        equity = np.array(self.equity_curve)
        total_return = (equity[-1] / self.initial_capital - 1) * 100

        n_years = len(self.dates) / 252 if len(self.dates) > 0 else 1
        ann_return = ((equity[-1] / self.initial_capital) ** (1 / n_years) - 1) * 100

        # Sharpe ratio
        returns = np.diff(equity) / equity[:-1]
        sharpe = (np.mean(returns) * 252 - 0.02) / (np.std(returns) * np.sqrt(252)) if len(returns) > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_dd = np.max(drawdown)

        # Trade statistics
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        avg_pnl = np.mean([t['pnl'] for t in self.trades]) if self.trades else 0

        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'final_equity': equity[-1]
        }
