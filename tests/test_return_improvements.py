from datetime import datetime

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
