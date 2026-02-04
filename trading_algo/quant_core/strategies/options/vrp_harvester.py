"""
Variance Risk Premium (VRP) Harvesting Strategy

Systematically sells options when implied volatility > expected realized volatility.

Academic Basis:
    VRP = E[IV²] - E[RV²]

    Options are systematically overpriced due to:
    1. Insurance premium demanded by hedgers
    2. Risk aversion of option buyers
    3. Jump/crash risk premium

Historical Performance:
    - Sharpe Ratio: 0.6-1.5 across asset classes
    - Best on commodities (Sharpe 1.5)
    - Average daily returns: 0.5-1.5%
    - Tail risk: Can lose -800% in extreme events

Strategy:
    1. Forecast realized volatility using GARCH
    2. Compare to implied volatility from options
    3. When IV > forecast RV: SELL options (collect premium)
    4. Delta hedge to maintain market neutrality
    5. Close positions before expiration or when VRP narrows

Risk Management:
    - Stop loss at 200% of premium collected
    - Maximum position size limits
    - Diversification across underlyings
    - Reduce exposure when VIX > 25

Reference:
    - Carr & Wu (2009) - "Variance Risk Premiums"
    - AQR Capital - "Understanding the Volatility Risk Premium"
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .base_option_strategy import BaseOptionStrategy
from trading_algo.quant_core.models.garch import estimate_garch_volatility, calculate_variance_premium


class VRPHarvester(BaseOptionStrategy):
    """
    Variance Risk Premium Harvesting Strategy.

    Sells options systematically when IV > forecasted RV.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_size: float = 0.05,  # 5% per underlying
        min_vrp_threshold: float = 2.0,  # Minimum 2% VRP to trade
        target_delta: float = 0.30,  # Sell 30-delta options
        days_to_expiry: int = 30,  # Sell 30-45 DTE
        stop_loss_multiplier: float = 2.0,  # Stop at 2x premium
        vix_threshold: float = 25.0,  # Reduce exposure when VIX > 25
        **kwargs
    ):
        """
        Initialize VRP Harvester.

        Args:
            initial_capital: Starting capital
            max_position_size: Max % of capital per position
            min_vrp_threshold: Minimum VRP to enter trade (percentage points)
            target_delta: Target delta for sold options (0.30 = 30%)
            days_to_expiry: Target days to expiration
            stop_loss_multiplier: Stop loss as multiple of premium collected
            vix_threshold: VIX level above which to reduce exposure
        """
        super().__init__(initial_capital, max_position_size, **kwargs)

        self.min_vrp_threshold = min_vrp_threshold
        self.target_delta = target_delta
        self.days_to_expiry = days_to_expiry
        self.stop_loss_multiplier = stop_loss_multiplier
        self.vix_threshold = vix_threshold

        # Track VRP calculations
        self.vrp_history = {}

    def generate_signals(
        self,
        symbol: str,
        underlying_price: float,
        historical_data: Dict,
        current_date: datetime
    ) -> List[Dict]:
        """
        Generate VRP-based trading signals.

        Args:
            symbol: Underlying symbol
            underlying_price: Current price
            historical_data: Dict with 'returns', 'prices', 'implied_vol', 'vix'
            current_date: Current date

        Returns:
            List of signal dicts
        """
        signals = []

        # Get historical returns for GARCH
        returns = historical_data.get('returns')
        if returns is None or len(returns) < 100:
            return signals  # Need at least 100 days for GARCH

        # Get current implied volatility
        implied_vol = historical_data.get('implied_vol', 0.25)

        # Get VIX if available
        vix = historical_data.get('vix', 15.0)

        # Forecast realized volatility using GARCH
        forecast_rv = estimate_garch_volatility(
            returns,
            forecast_horizon=self.days_to_expiry
        )

        # Calculate Variance Risk Premium
        vrp = calculate_variance_premium(
            implied_vol,
            returns,
            forecast_horizon=self.days_to_expiry
        )

        # Store VRP
        if symbol not in self.vrp_history:
            self.vrp_history[symbol] = []
        self.vrp_history[symbol].append({
            'date': current_date,
            'vrp': vrp,
            'iv': implied_vol,
            'rv_forecast': forecast_rv,
            'vix': vix
        })

        # TRADING LOGIC
        # Sell options when VRP is positive and significant
        if vrp >= self.min_vrp_threshold:

            # Adjust position size based on VIX
            size_multiplier = 1.0
            if vix > self.vix_threshold:
                # Reduce size by 50% when VIX is elevated
                size_multiplier = 0.5

            # Calculate number of contracts
            position_value = self.capital * self.max_position_size * size_multiplier
            contracts_to_sell = int(position_value / (underlying_price * 100))

            if contracts_to_sell > 0:
                # Sell both puts and calls (strangle) for max premium
                # Or just puts if market is bullish

                # Calculate strikes based on target delta
                # For 30-delta put: strike is roughly 30% OTM
                put_strike = underlying_price * (1 - self.target_delta)
                call_strike = underlying_price * (1 + self.target_delta)

                # Sell put signal
                signals.append({
                    'action': 'sell',
                    'option_type': 'put',
                    'strike': put_strike,
                    'quantity': contracts_to_sell,
                    'expiry_days': self.days_to_expiry,
                    'reason': f'VRP={vrp:.2f}%, IV={implied_vol:.2%}, RV_forecast={forecast_rv:.2%}',
                    'vrp': vrp
                })

                # Optionally sell call as well (short strangle)
                # Comment out if you only want to sell puts
                signals.append({
                    'action': 'sell',
                    'option_type': 'call',
                    'strike': call_strike,
                    'quantity': contracts_to_sell,
                    'expiry_days': self.days_to_expiry,
                    'reason': f'VRP={vrp:.2f}%, IV={implied_vol:.2%}, RV_forecast={forecast_rv:.2%}',
                    'vrp': vrp
                })

        return signals

    def manage_risk(self, current_date: datetime):
        """
        Risk management for VRP strategy.

        Checks:
        - Stop losses (2x premium collected)
        - Time decay targets (close at 50% profit)
        - Expiration management (roll or close)
        """
        positions_to_close = []

        for position in self.positions:
            # Check stop loss
            pnl = position.pnl()
            premium_collected = abs(position.entry_price * position.quantity * 100)

            if pnl < -premium_collected * self.stop_loss_multiplier:
                # Hit stop loss
                positions_to_close.append((position, 'stop_loss'))
                continue

            # Check profit target (50% of max profit)
            if pnl > premium_collected * 0.50:
                # Take profits early
                positions_to_close.append((position, 'profit_target'))
                continue

            # Check days to expiration
            days_left = (position.expiry - current_date).days

            if days_left <= 7:
                # Close positions with < 7 days to avoid pin risk
                positions_to_close.append((position, 'expiration'))
                continue

        return positions_to_close

    def calculate_delta_hedge(self, symbol: str) -> float:
        """
        Calculate required stock position to delta hedge.

        Returns:
            Number of shares to buy (positive) or sell (negative)
        """
        # Sum portfolio delta for this symbol
        symbol_delta = sum(
            p.delta for p in self.positions if p.symbol == symbol
        )

        # Delta hedge: buy shares to offset negative delta
        # (Negative delta from short options needs long stock)
        shares_needed = -symbol_delta * 100  # Each option = 100 shares

        return shares_needed


def backtest_vrp_strategy(
    symbol: str,
    historical_prices: np.ndarray,
    historical_returns: np.ndarray,
    implied_vols: np.ndarray,
    dates: List[datetime],
    initial_capital: float = 100000.0,
    **strategy_params
) -> Dict:
    """
    Backtest VRP Harvesting strategy.

    Args:
        symbol: Underlying symbol
        historical_prices: Price series
        historical_returns: Return series
        implied_vols: Implied volatility series
        dates: Date series
        initial_capital: Starting capital
        **strategy_params: Parameters for VRPHarvester

    Returns:
        Performance dictionary
    """
    strategy = VRPHarvester(initial_capital=initial_capital, **strategy_params)

    # Simulation loop
    for t in range(60, len(dates)):  # Start after 60 days for GARCH
        current_date = dates[t]
        current_price = historical_prices[t]
        current_iv = implied_vols[t]

        # Prepare historical data
        hist_data = {
            'returns': historical_returns[:t],
            'prices': historical_prices[:t],
            'implied_vol': current_iv,
            'vix': 15.0  # Simplified - would use actual VIX
        }

        # Generate signals
        signals = strategy.generate_signals(symbol, current_price, hist_data, current_date)

        # Execute signals
        for signal in signals:
            # Calculate option price (simplified - would use actual market prices)
            time_to_expiry = signal['expiry_days'] / 365.0

            from trading_algo.quant_core.models.greeks import OptionSpec, BlackScholesCalculator

            strike = signal['strike']
            option_spec = OptionSpec(
                spot=current_price,
                strike=strike,
                time_to_expiry=time_to_expiry,
                volatility=current_iv,
                risk_free_rate=strategy.risk_free_rate,
                option_type=signal['option_type']
            )

            option_price = BlackScholesCalculator.price(option_spec)

            # Open position
            expiry_date = current_date + timedelta(days=signal['expiry_days'])

            strategy.open_position(
                symbol=symbol,
                option_type=signal['option_type'],
                strike=strike,
                expiry=expiry_date,
                quantity=-signal['quantity'],  # Negative = short
                price=option_price,
                underlying_price=current_price,
                current_date=current_date,
                implied_vol=current_iv
            )

        # Update existing positions
        strategy.update_positions(symbol, current_price, current_iv, current_date)

        # Risk management
        to_close = strategy.manage_risk(current_date)
        for position, reason in to_close:
            # Recalculate current option price
            time_left = (position.expiry - current_date).days / 365.0
            close_spec = OptionSpec(
                spot=current_price,
                strike=position.strike,
                time_to_expiry=time_left,
                volatility=current_iv,
                risk_free_rate=strategy.risk_free_rate,
                option_type=position.option_type
            )
            close_price = BlackScholesCalculator.price(close_spec)

            strategy.close_position(position, close_price, current_date)

    # Return performance stats
    stats = strategy.get_portfolio_stats()
    stats['equity_curve'] = strategy.performance.equity_curve
    stats['dates'] = strategy.performance.dates
    stats['vrp_history'] = strategy.vrp_history

    return stats
