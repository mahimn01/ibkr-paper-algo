import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from trading_algo.orchestrator.edges.time_of_day import TimeOfDayEngine
from trading_algo.orchestrator.types import EdgeVote, TradeType
from trading_algo.quant_core.engine.orchestrator import EngineConfig, QuantOrchestrator
from trading_algo.quant_core.engine.signal_aggregator import (
    AggregatorConfig,
    ModelSignal,
    SignalAggregator,
    SignalType,
)
from trading_algo.quant_core.engine.trading_context import BacktestContext, OrderSide
from trading_algo.quant_core.models.hmm_regime import MarketRegime


def test_time_of_day_respects_short_direction() -> None:
    engine = TimeOfDayEngine()
    timestamp = datetime(2025, 1, 2, 9, 35)  # opening drive window

    signal = engine.get_vote(
        timestamp,
        TradeType.MOMENTUM_CONTINUATION,
        intended_direction="short",
    )

    assert signal.vote == EdgeVote.SHORT


def test_regime_weight_tilt_favors_momentum_in_bull() -> None:
    cfg = AggregatorConfig(regime_weight_tilt=0.5, min_signal_threshold=0.0)
    agg = SignalAggregator(cfg)
    agg._current_regime = MarketRegime.BULL

    adjusted = agg._get_regime_adjusted_weights()

    assert adjusted[SignalType.MOMENTUM] > agg._weights[SignalType.MOMENTUM]


def test_signal_disagreement_penalty_reduces_blended_signal() -> None:
    cfg = AggregatorConfig(min_signal_threshold=0.0, disagreement_penalty=0.5)
    agg = SignalAggregator(cfg)

    opposing_signals = {
        SignalType.MOMENTUM: ModelSignal(
            signal_type=SignalType.MOMENTUM,
            value=1.0,
            confidence=1.0,
            raw_value=1.0,
        ),
        SignalType.MEAN_REVERSION: ModelSignal(
            signal_type=SignalType.MEAN_REVERSION,
            value=-1.0,
            confidence=1.0,
            raw_value=-1.0,
        ),
    }

    combined, confidence = agg._combine_signals(opposing_signals)

    assert abs(combined) < 0.1
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize(
    ("bar_frequency", "expected"),
    [
        ("1D", 252.0),
        ("1H", 6.5 * 252.0),
        ("5m", (390.0 / 5.0) * 252.0),
    ],
)
def test_periods_per_year_parsing(bar_frequency: str, expected: float) -> None:
    orchestrator = QuantOrchestrator(EngineConfig(universe=[]))
    assert orchestrator._periods_per_year(bar_frequency) == pytest.approx(expected)


def test_backtest_context_reports_closed_trades_not_fill_events() -> None:
    timestamps = np.array([
        datetime(2026, 1, 2, 9, 30),
        datetime(2026, 1, 2, 9, 35),
        datetime(2026, 1, 2, 9, 40),
    ], dtype=object)
    historical_data = {
        "AAA": np.array([
            [100.0, 100.0, 100.0, 100.0, 1000.0],
            [105.0, 105.0, 105.0, 105.0, 1000.0],
            [104.0, 104.0, 104.0, 104.0, 1000.0],
        ])
    }
    ctx = BacktestContext(
        historical_data=historical_data,
        timestamps=timestamps,
        commission_rate=0.0,
        slippage_rate=0.0,
    )

    ctx.submit_order("AAA", OrderSide.BUY, 10)
    assert ctx.advance()
    ctx.submit_order("AAA", OrderSide.SELL, 10)

    results = ctx.get_results()
    assert results["n_fills"] == 2
    assert results["n_trades"] == 1
    assert len(results["trades"]) == 1
    assert results["trades"][0]["pnl"] == pytest.approx(50.0)


def test_backtest_context_handles_flip_with_fifo_pnl() -> None:
    timestamps = np.array([
        datetime(2026, 1, 2, 9, 30),
        datetime(2026, 1, 2, 9, 35),
        datetime(2026, 1, 2, 9, 40),
        datetime(2026, 1, 2, 9, 45),
    ], dtype=object)
    historical_data = {
        "AAA": np.array([
            [100.0, 100.0, 100.0, 100.0, 1000.0],
            [101.0, 101.0, 101.0, 101.0, 1000.0],
            [99.0, 99.0, 99.0, 99.0, 1000.0],
            [99.0, 99.0, 99.0, 99.0, 1000.0],
        ])
    }
    ctx = BacktestContext(
        historical_data=historical_data,
        timestamps=timestamps,
        commission_rate=0.0,
        slippage_rate=0.0,
    )

    ctx.submit_order("AAA", OrderSide.BUY, 10)   # +10 @100
    assert ctx.advance()
    ctx.submit_order("AAA", OrderSide.SELL, 15)  # close +10 @101, open -5 @101
    assert ctx.advance()
    ctx.submit_order("AAA", OrderSide.BUY, 5)    # close -5 @99

    results = ctx.get_results()
    assert results["n_fills"] == 3
    assert results["n_trades"] == 2
    assert sum(t["pnl"] for t in results["trades"]) == pytest.approx(20.0)
    assert results["trades"][0]["direction"] == "LONG"
    assert results["trades"][1]["direction"] == "SHORT"


def test_save_results_exports_reconciled_trade_and_fill_logs(tmp_path: Path) -> None:
    timestamps = np.array([
        datetime(2026, 1, 2, 9, 30),
        datetime(2026, 1, 2, 9, 35),
        datetime(2026, 1, 2, 9, 40),
    ], dtype=object)
    historical_data = {
        "AAA": np.array([
            [100.0, 100.0, 100.0, 100.0, 1000.0],
            [105.0, 105.0, 105.0, 105.0, 1000.0],
            [104.0, 104.0, 104.0, 104.0, 1000.0],
        ])
    }
    ctx = BacktestContext(
        historical_data=historical_data,
        timestamps=timestamps,
        commission_rate=0.0,
        slippage_rate=0.0,
    )
    ctx.submit_order("AAA", OrderSide.BUY, 10)
    assert ctx.advance()
    ctx.submit_order("AAA", OrderSide.SELL, 10)

    orchestrator = QuantOrchestrator(EngineConfig(universe=[]))
    orchestrator.context = ctx
    orchestrator._trade_log = [{"symbol": "AAA", "shares": 10, "price": 100.0}]

    out = tmp_path / "exports"
    orchestrator.save_results(str(out))

    assert (out / "execution_events.json").exists()
    assert (out / "trades.json").exists()
    assert (out / "fills.json").exists()

    trades = json.loads((out / "trades.json").read_text())
    fills = json.loads((out / "fills.json").read_text())
    summary = json.loads((out / "summary.json").read_text())

    assert len(trades) == 1
    assert len(fills) == 2
    assert summary["n_trades"] == 1
    assert summary["n_fills"] == 2
