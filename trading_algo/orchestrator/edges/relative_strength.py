"""
Edge 2: Relative Strength Engine

Compares a stock's performance to its sector and the market.

Key insight: Stocks that outperform their sector tend to continue outperforming.
Stocks that underperform their sector tend to continue underperforming.

This is NOT about absolute momentum - it's about RELATIVE performance.
"""

from typing import Dict, Optional, Tuple

from ..types import AssetState, EdgeSignal, EdgeVote


class RelativeStrengthEngine:
    """
    Compares a stock's performance to its sector and the market.

    Key insight: Stocks that outperform their sector tend to continue outperforming.
    Stocks that underperform their sector tend to continue underperforming.

    Example:
    - INTC up 11%, SMH up 6%, SPY up 1% → INTC is a leader, trade WITH it
    - INTC up 2%, SMH up 6%, SPY up 2% → INTC is a laggard, avoid or fade
    """

    # Sector ETF mappings
    SECTOR_MAP = {
        # Tech / Semiconductors
        "INTC": "SMH", "AMD": "SMH", "NVDA": "SMH", "MU": "SMH",
        "AVGO": "SMH", "QCOM": "SMH", "TSM": "SMH", "MRVL": "SMH",
        "SMCI": "SMH", "ARM": "SMH",
        # Tech general
        "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "META": "XLK",
        "CRM": "XLK", "ORCL": "XLK", "ADBE": "XLK",
        # Consumer tech
        "AMZN": "XLY", "TSLA": "XLY", "NFLX": "XLY",
        # Financials
        "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF",
        "C": "XLF", "WFC": "XLF", "SCHW": "XLF",
        # Energy
        "XOM": "XLE", "CVX": "XLE", "OXY": "XLE", "SLB": "XLE",
        # EV / Clean energy
        "RIVN": "QCLN", "LCID": "QCLN", "NIO": "QCLN", "PLUG": "QCLN",
        # Consumer
        "SBUX": "XLY", "MCD": "XLY", "NKE": "XLY", "DIS": "XLY",
        # Healthcare
        "UNH": "XLV", "JNJ": "XLV", "PFE": "XLV", "ABBV": "XLV",
        # Leveraged ETFs - compare to underlying
        "SOXL": "SMH", "TQQQ": "QQQ",
        # Precious metals
        "GLD": "SPY", "SIVR": "SPY", "SLV": "SPY",
    }

    def __init__(self):
        self.asset_states: Dict[str, AssetState] = {}
        self.spy_state: Optional[AssetState] = None

    def update(self, symbol: str, state: AssetState):
        """Update with latest asset state."""
        self.asset_states[symbol] = state
        if symbol == "SPY":
            self.spy_state = state

    def calculate_relative_strength(
        self,
        symbol: str,
        lookback_bars: int = 20
    ) -> Tuple[float, float, float, str]:
        """
        Calculate relative strength vs sector and market.

        Returns: (rs_vs_sector, rs_vs_market, percentile_rank, reason)

        rs > 0 means outperforming
        rs < 0 means underperforming
        percentile_rank is 0-100 (100 = strongest)
        """
        if symbol not in self.asset_states:
            return 0.0, 0.0, 50.0, "No data"

        state = self.asset_states[symbol]
        if len(state.prices) < lookback_bars:
            return 0.0, 0.0, 50.0, "Insufficient history"

        prices = list(state.prices)
        stock_return = (prices[-1] - prices[-lookback_bars]) / prices[-lookback_bars]

        # Get sector ETF return
        sector_etf = self.SECTOR_MAP.get(symbol, "SPY")
        sector_return = 0.0
        if sector_etf in self.asset_states:
            sector_prices = list(self.asset_states[sector_etf].prices)
            if len(sector_prices) >= lookback_bars:
                sector_return = (sector_prices[-1] - sector_prices[-lookback_bars]) / sector_prices[-lookback_bars]

        # Get SPY return
        market_return = 0.0
        if self.spy_state and len(self.spy_state.prices) >= lookback_bars:
            spy_prices = list(self.spy_state.prices)
            market_return = (spy_prices[-1] - spy_prices[-lookback_bars]) / spy_prices[-lookback_bars]

        # Relative strength
        rs_vs_sector = stock_return - sector_return
        rs_vs_market = stock_return - market_return

        # Calculate percentile rank among all tracked stocks
        all_returns = []
        for sym, st in self.asset_states.items():
            if sym not in ["SPY", "QQQ", "IWM"] and len(st.prices) >= lookback_bars:
                ret = (st.prices[-1] - st.prices[-lookback_bars]) / st.prices[-lookback_bars]
                all_returns.append((sym, ret))

        if all_returns:
            sorted_returns = sorted(all_returns, key=lambda x: x[1])
            rank = next((i for i, (s, r) in enumerate(sorted_returns) if s == symbol), len(sorted_returns)//2)
            percentile = (rank / len(sorted_returns)) * 100
        else:
            percentile = 50.0

        # Build reason
        reason_parts = []
        if rs_vs_sector > 0.005:
            reason_parts.append(f"Outperforming {sector_etf} by {rs_vs_sector*100:.1f}%")
        elif rs_vs_sector < -0.005:
            reason_parts.append(f"Underperforming {sector_etf} by {abs(rs_vs_sector)*100:.1f}%")

        if rs_vs_market > 0.01:
            reason_parts.append(f"Leading SPY by {rs_vs_market*100:.1f}%")
        elif rs_vs_market < -0.01:
            reason_parts.append(f"Lagging SPY by {abs(rs_vs_market)*100:.1f}%")

        reason_parts.append(f"Rank: {percentile:.0f}th percentile")

        return rs_vs_sector, rs_vs_market, percentile, " | ".join(reason_parts)

    def get_vote(self, symbol: str) -> EdgeSignal:
        """Get voting signal based on relative strength."""
        rs_sector, rs_market, percentile, reason = self.calculate_relative_strength(symbol)

        # Strong signals
        if percentile > 90 and rs_sector > 0.01:
            return EdgeSignal("RelativeStrength", EdgeVote.STRONG_LONG, 0.9, reason,
                            {"rs_sector": rs_sector, "rs_market": rs_market, "percentile": percentile})

        if percentile < 10 and rs_sector < -0.01:
            return EdgeSignal("RelativeStrength", EdgeVote.STRONG_SHORT, 0.9, reason,
                            {"rs_sector": rs_sector, "rs_market": rs_market, "percentile": percentile})

        # Moderate signals
        if percentile > 70 and rs_sector > 0.005:
            return EdgeSignal("RelativeStrength", EdgeVote.LONG, 0.6, reason,
                            {"rs_sector": rs_sector, "rs_market": rs_market, "percentile": percentile})

        if percentile < 30 and rs_sector < -0.005:
            return EdgeSignal("RelativeStrength", EdgeVote.SHORT, 0.6, reason,
                            {"rs_sector": rs_sector, "rs_market": rs_market, "percentile": percentile})

        # Conflict detection (stock vs sector divergence can be a trap)
        if rs_sector > 0.02 and rs_market < -0.01:
            # Stock up, sector down, market down - potential trap
            return EdgeSignal("RelativeStrength", EdgeVote.VETO_LONG, 0.7,
                            f"Divergence warning: stock up but sector/market down",
                            {"rs_sector": rs_sector, "rs_market": rs_market, "percentile": percentile})

        return EdgeSignal("RelativeStrength", EdgeVote.NEUTRAL, 0.5, reason,
                        {"rs_sector": rs_sector, "rs_market": rs_market, "percentile": percentile})
