"""
Aggressive Configuration Presets for Higher Returns

These configurations target 25-50%+ annual returns with corresponding
higher risk. Choose based on your risk tolerance and market conditions.

WARNING: Higher returns = higher risk. These configurations can have
40-60% drawdowns in adverse conditions.
"""

from trading_algo.quant_core.engine.orchestrator import (
    EngineConfig, EngineMode
)
from trading_algo.quant_core.engine.signal_aggregator import AggregatorConfig
from trading_algo.quant_core.engine.risk_controller import RiskConfig
from trading_algo.quant_core.engine.portfolio_manager import PortfolioConfig
from trading_algo.quant_core.engine.execution_manager import ExecutionConfig


def get_aggressive_equity_config() -> EngineConfig:
    """
    Aggressive equity configuration.

    Target: 25-40% annual returns
    Expected Drawdown: 30-40%
    Sharpe Target: 0.8-1.2

    Key differences from conservative:
    - Higher leverage (2x gross exposure)
    - Full Kelly sizing
    - Shorter momentum lookbacks (more responsive)
    - Smaller position limits for more diversification
    """
    return EngineConfig(
        mode=EngineMode.BACKTEST,
        universe=[
            # High-beta tech/growth (momentum works well)
            'QQQ', 'TQQQ',  # Tech/leveraged tech
            'ARKK',         # Innovation
            # Sector momentum plays
            'XLF', 'XLE', 'XLK', 'XLV',
            # Volatility products
            'VXX', 'UVXY',
        ],
        benchmark_symbol='QQQ',
        bar_frequency='1D',

        signal_config=AggregatorConfig(
            # Shorter lookbacks = more responsive
            ou_lookback=30,          # Was 60 - faster mean reversion
            momentum_lookback=60,    # Was 120 - faster trend capture
            vol_lookback=20,         # Was 30

            # Favor momentum in trending markets
            ou_weight=0.15,          # Reduced
            tsmom_weight=0.40,       # Increased - primary signal
            vol_mom_weight=0.25,     # Increased
            hmm_weight=0.20,         # Regime-dependent

            vol_target=0.25,         # Higher vol target
            min_signal_threshold=0.05,  # Lower threshold = more trades
        ),

        risk_config=RiskConfig(
            max_drawdown=0.35,       # Accept larger drawdowns
            max_daily_loss=0.06,     # 6% daily loss limit
            max_position_size=0.25,  # Max 25% in single position
            max_gross_exposure=2.0,  # 2x leverage
            max_sector_exposure=0.5, # Allow concentrated sectors
            var_limit=0.08,          # Higher VaR
            trailing_stop_pct=0.12,  # Wider stops
        ),

        portfolio_config=PortfolioConfig(
            target_gross_exposure=1.5,   # 150% invested
            max_position_size=0.20,      # 20% max per position
            min_position_size=0.02,      # 2% min
            kelly_fraction=0.75,         # Three-quarter Kelly
            max_kelly_leverage=2.5,      # Allow more leverage
            vol_target=0.25,             # 25% annual vol target
            min_trade_value=500,         # Smaller min trade
        ),

        execution_config=ExecutionConfig(),
        warmup_bars=30,
        rebalance_frequency='daily',
    )


def get_leveraged_etf_config() -> EngineConfig:
    """
    Leveraged ETF momentum configuration.

    Target: 40-80% annual returns (with 50%+ drawdowns possible)

    Uses 3x leveraged ETFs with momentum signals.
    ONLY for accounts you can afford to lose.
    """
    return EngineConfig(
        mode=EngineMode.BACKTEST,
        universe=[
            # 3x leveraged long
            'TQQQ', 'UPRO', 'TNA', 'SOXL', 'LABU',
            # 3x leveraged short (for hedging)
            'SQQQ', 'SPXU', 'TZA',
            # Cash proxy
            'SHY',
        ],
        benchmark_symbol='TQQQ',
        bar_frequency='1D',

        signal_config=AggregatorConfig(
            # Very short lookbacks for leveraged products
            ou_lookback=10,          # Fast mean reversion
            momentum_lookback=20,    # Fast momentum
            vol_lookback=10,

            # Heavy momentum bias
            ou_weight=0.10,
            tsmom_weight=0.50,       # Dominant
            vol_mom_weight=0.30,
            hmm_weight=0.10,

            vol_target=0.40,         # Very high vol target
            min_signal_threshold=0.10,  # Require strong signals
        ),

        risk_config=RiskConfig(
            max_drawdown=0.50,       # 50% max drawdown
            max_daily_loss=0.15,     # 15% daily (leveraged can move fast)
            max_position_size=0.30,
            max_gross_exposure=1.0,  # No additional leverage on leveraged ETFs!
            max_sector_exposure=0.6,
            var_limit=0.15,
            trailing_stop_pct=0.20,  # Wide stops for vol
        ),

        portfolio_config=PortfolioConfig(
            target_gross_exposure=0.8,   # Actually conservative on gross
            max_position_size=0.25,
            min_position_size=0.05,
            kelly_fraction=0.50,         # Half Kelly (already levered)
            max_kelly_leverage=1.0,      # No extra leverage
            vol_target=0.40,
            min_trade_value=1000,
        ),

        execution_config=ExecutionConfig(),
        warmup_bars=15,
        rebalance_frequency='daily',
    )


def get_volatility_trading_config() -> EngineConfig:
    """
    Volatility trading configuration.

    Target: 20-40% annual returns

    Exploits volatility mean reversion and term structure.
    Best in high-vol environments.
    """
    return EngineConfig(
        mode=EngineMode.BACKTEST,
        universe=[
            'VXX', 'UVXY', 'SVXY',  # Vol products
            'SPY', 'QQQ',           # For hedging
            'TLT',                   # Bonds (vol correlation)
        ],
        benchmark_symbol='SPY',
        bar_frequency='1D',

        signal_config=AggregatorConfig(
            # OU is key for vol mean reversion
            ou_lookback=15,          # Short for vol
            momentum_lookback=10,
            vol_lookback=5,

            # Heavy OU weight for mean reversion
            ou_weight=0.50,          # Dominant
            tsmom_weight=0.20,
            vol_mom_weight=0.20,
            hmm_weight=0.10,

            vol_target=0.30,
            min_signal_threshold=0.15,  # Strong signals only
        ),

        risk_config=RiskConfig(
            max_drawdown=0.30,
            max_daily_loss=0.10,
            max_position_size=0.20,
            max_gross_exposure=1.5,
            max_sector_exposure=0.40,
            var_limit=0.10,
            trailing_stop_pct=0.15,
        ),

        portfolio_config=PortfolioConfig(
            target_gross_exposure=1.0,
            max_position_size=0.20,
            min_position_size=0.03,
            kelly_fraction=0.50,
            max_kelly_leverage=1.5,
            vol_target=0.30,
            min_trade_value=500,
        ),

        execution_config=ExecutionConfig(),
        warmup_bars=20,
        rebalance_frequency='daily',
    )


def get_fx_momentum_config() -> EngineConfig:
    """
    FX momentum configuration (requires FX data feed).

    Target: 15-30% annual returns
    Expected Drawdown: 15-25%

    Classic G10 currency momentum with carry overlay.
    This is where institutional CTAs make money.
    """
    return EngineConfig(
        mode=EngineMode.BACKTEST,
        universe=[
            # G10 currency ETFs
            'FXE',   # Euro
            'FXY',   # Yen
            'FXB',   # Pound
            'FXA',   # Aussie
            'FXC',   # CAD
            'FXF',   # Swiss Franc
            # EM exposure
            'CEW',   # EM currencies
            # Dollar index
            'UUP',   # Long dollar
        ],
        benchmark_symbol='UUP',
        bar_frequency='1D',

        signal_config=AggregatorConfig(
            # Medium lookbacks for FX
            ou_lookback=40,
            momentum_lookback=90,    # FX momentum works at 3-12 months
            vol_lookback=20,

            # TSMOM dominant (research-backed for FX)
            ou_weight=0.15,
            tsmom_weight=0.45,       # Dominant
            vol_mom_weight=0.25,
            hmm_weight=0.15,

            vol_target=0.15,         # FX is lower vol
            min_signal_threshold=0.08,
        ),

        risk_config=RiskConfig(
            max_drawdown=0.20,
            max_daily_loss=0.04,
            max_position_size=0.20,
            max_gross_exposure=3.0,  # FX allows high leverage
            max_sector_exposure=0.30,
            var_limit=0.05,
            trailing_stop_pct=0.08,
        ),

        portfolio_config=PortfolioConfig(
            target_gross_exposure=2.0,
            max_position_size=0.15,
            min_position_size=0.03,
            kelly_fraction=0.50,
            max_kelly_leverage=4.0,  # FX leverage
            vol_target=0.15,
            min_trade_value=500,
        ),

        execution_config=ExecutionConfig(),
        warmup_bars=60,
        rebalance_frequency='daily',
    )


def get_commodity_momentum_config() -> EngineConfig:
    """
    Commodity momentum configuration.

    Target: 20-35% annual returns
    Expected Drawdown: 20-30%

    Trend-following across commodity sectors.
    Works best in inflationary/deflationary regimes.
    """
    return EngineConfig(
        mode=EngineMode.BACKTEST,
        universe=[
            # Energy
            'USO', 'UNG', 'XLE',
            # Metals
            'GLD', 'SLV', 'PPLT',
            # Agriculture
            'DBA', 'CORN', 'WEAT',
            # Broad commodity
            'DBC', 'GSG',
        ],
        benchmark_symbol='DBC',
        bar_frequency='1D',

        signal_config=AggregatorConfig(
            # Longer lookbacks for commodities
            ou_lookback=60,
            momentum_lookback=180,   # 6-12 month momentum works
            vol_lookback=30,

            # Heavy momentum bias
            ou_weight=0.10,
            tsmom_weight=0.50,       # Dominant
            vol_mom_weight=0.25,
            hmm_weight=0.15,

            vol_target=0.20,
            min_signal_threshold=0.08,
        ),

        risk_config=RiskConfig(
            max_drawdown=0.25,
            max_daily_loss=0.05,
            max_position_size=0.15,
            max_gross_exposure=2.0,
            max_sector_exposure=0.35,
            var_limit=0.06,
            trailing_stop_pct=0.10,
        ),

        portfolio_config=PortfolioConfig(
            target_gross_exposure=1.5,
            max_position_size=0.12,
            min_position_size=0.03,
            kelly_fraction=0.50,
            max_kelly_leverage=2.0,
            vol_target=0.20,
            min_trade_value=500,
        ),

        execution_config=ExecutionConfig(),
        warmup_bars=90,
        rebalance_frequency='weekly',  # Less frequent for commodities
    )


# Summary of expected returns by strategy
STRATEGY_EXPECTATIONS = {
    'aggressive_equity': {
        'target_return': '25-40%',
        'expected_drawdown': '30-40%',
        'sharpe_range': '0.8-1.2',
        'best_regime': 'Trending markets',
        'worst_regime': 'Choppy/ranging',
    },
    'leveraged_etf': {
        'target_return': '40-80%',
        'expected_drawdown': '50-70%',
        'sharpe_range': '0.5-1.0',
        'best_regime': 'Strong trends',
        'worst_regime': 'Volatility spikes',
    },
    'volatility': {
        'target_return': '20-40%',
        'expected_drawdown': '25-35%',
        'sharpe_range': '0.8-1.5',
        'best_regime': 'High vol -> low vol',
        'worst_regime': 'Sustained low vol',
    },
    'fx_momentum': {
        'target_return': '15-30%',
        'expected_drawdown': '15-25%',
        'sharpe_range': '0.8-1.3',
        'best_regime': 'Divergent monetary policy',
        'worst_regime': 'Central bank interventions',
    },
    'commodity_momentum': {
        'target_return': '20-35%',
        'expected_drawdown': '20-30%',
        'sharpe_range': '0.7-1.2',
        'best_regime': 'Inflation/deflation trends',
        'worst_regime': 'Supply shock reversals',
    },
}
