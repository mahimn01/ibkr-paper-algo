# Codex Algorithm Changes — Review & Next Steps

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-07
**Branch reviewed:** `codex/algorithm/return-improvements-core`
**Commits analyzed:** 4 (25f58c5 → 8910505)

---

## Summary of What Codex Changed

Codex delivered 4 commits adding ~25,800 lines across 75 new files and modifying 7 existing files:

### Commit 1: Add comprehensive quantitative trading framework (`25f58c5`)
The largest commit — a full `quant_core` module with:

| Component | Files | Purpose |
|-----------|-------|---------|
| **Strategies** | 8 | Mean Reversion RSI, Opening Range Breakout, Pairs Trading, Options (Iron Condor, Theta, VRP), Pure Momentum, Volatility Maximizer |
| **Engine** | 7 | QuantOrchestrator, SignalAggregator, PortfolioManager, RiskController, ExecutionManager, TradingContext |
| **Models** | 6 | GARCH, HMM Regime Detection, Ornstein-Uhlenbeck, TSMOM, Vol-Managed Momentum, Greeks |
| **Portfolio** | 3 | Kelly Criterion, HRP, Mean-Variance Optimizer |
| **Risk** | 3 | Expected Shortfall, Tail Risk, Risk Metrics |
| **Execution** | 2 | TWAP/VWAP, Almgren-Chriss optimal execution |
| **ML** | 3 | Feature Engine, Signal Combiner, Cross-Validation |
| **Validation** | 2 | Backtest Validator, Probability of Backtest Overfitting (PBO) |
| **Examples** | 12 | Backtest scripts for each strategy |

### Commit 2: Improve return quality across orchestrator and quant engine (`b63a0a3`)
Targeted improvements to reduce overtrading and improve signal quality:

- **Time-of-Day Edge fix**: Previously always voted `LONG` regardless of intended trade direction. Now respects `intended_direction` parameter (critical bug fix).
- **Re-entry cooldown**: 12-bar minimum cooldown after exits to reduce churn.
- **ATR filters**: Skip trades when ATR% < 0.15% (chop) or > 3% (panic).
- **Regime confidence gate**: Reject trades when regime confidence < 0.5.
- **Opposition scoring**: Track edges voting *against* the trade direction and block when opposition > 0.35.
- **Directional quality metric**: Block when quality < 0.6 (support / (support + opposition)).
- **Position sizing overhaul**: Reduced base from 1% to 0.8%, added volatility scalar and quality boost.
- **Signal Aggregator**: Added disagreement penalty (0.35) to reduce blended signal when models conflict. Added regime-adaptive weight tilting (momentum favored in bull, mean-reversion in high-vol).

### Commit 3: Fix backtest trade accounting (`4dea07f`)
- Implemented FIFO lot matching (`_match_lots`) in `BacktestContext` — previously only tracked fills, not round-trip trades.
- Trade PnL now properly distinguishes entries vs. exits, long vs. short, and handles position flips.
- Added `closed_trades` list with proper per-trade PnL, gross/net, commission, slippage breakdown.
- Added `_infer_periods_per_year()` for automatic annualization factor detection.

### Commit 4: Reconcile trade PnL with equity (`8910505`)
- Fixed PnL reconciliation so `sum(trade.pnl)` matches `final_equity - initial_capital`.
- Slippage is now embedded in fill prices (not double-counted).
- Added `_mark_to_market_open_trades()` for open positions at backtest end.
- Added `save_results()` export with separate trades.json, fills.json, summary.json.

---

## Assessment of Code Quality

### Strengths

1. **The Time-of-Day direction bug fix is the most impactful change.** Before this, the TimeOfDay edge always voted LONG, meaning it systematically boosted long consensus and suppressed short trades. This is a genuine signal quality improvement.

2. **Lot-matching trade accounting is correct.** The FIFO implementation handles adds, reduces, closes, and flips. The test suite covers all these cases and verifies PnL reconciliation with slippage. This was a real gap in the backtest system.

3. **Signal disagreement penalty is well-designed.** When models disagree on direction, the blended signal shrinks toward zero. This reduces overtrading in noisy regimes.

4. **Test coverage is solid.** 10 new targeted tests covering direction-aware voting, regime weight tilting, disagreement penalties, period parsing, FIFO lot matching, flip handling, slippage reconciliation, and export integrity. All 10 pass. Full suite: 303 pass, 2 skip (expected — no `ib_insync` in this env).

### Concerns

1. **Massive surface area with limited integration.** The `quant_core` module is ~20,000 lines of new code that is essentially standalone. The `QuantOrchestrator` and the original `Orchestrator` are two separate systems with no bridge between them. The original `Orchestrator` (used by `run.py` and the autorunner) does NOT use any of the quant_core signal models, portfolio optimization, or risk controller.

2. **Many hardcoded magic numbers in the Orchestrator.** The new filters (`min_reentry_bars=12`, `min_atr_pct=0.0015`, `max_atr_pct=0.03`, `max_opposition_score=0.35`, directional quality threshold 0.6, base_size 0.008, quality_boost coefficients) are all hardcoded without backtest evidence that these are optimal. They look reasonable as starting points but could easily be over-fit to specific market conditions.

3. **Position sizing formula is fragile.** The volatility scalar `max(0.35, min(1.35, 0.01 / max(atr_pct, 0.0005)))` targets ~1% ATR as the "ideal" — but the constant 0.01 is unexplained and the clamps at 0.35/1.35 are arbitrary. Small changes in these could swing position sizes dramatically.

4. **No backtest validation of the orchestrator changes.** The `quant_core` strategies have backtest examples, but the actual `Orchestrator` (the one that drives live trading) has no backtest comparing before/after these filter changes. We can't verify that the cooldown, ATR filter, opposition scoring, etc. actually improve returns on historical data.

5. **Strategy count explosion without clear winner.** Codex added 8+ strategies (Mean Reversion, ORB, Pairs, Options, Momentum, Volatility Maximizer) but the PURE_MOMENTUM_RESULTS.md shows mixed results and the code doesn't converge on a recommended configuration.

---

## Recommended Next Steps

### Priority 1: Validate the Orchestrator Changes with Historical Data

The Orchestrator is what actually trades. Run a before/after backtest:
- Backtest the Orchestrator **without** the new filters (revert to pre-Codex parameters)
- Backtest **with** the new filters
- Compare: trade count, win rate, Sharpe, max drawdown, profit factor
- Use real IBKR historical data (the existing `backtest_v2` engine can drive this)

This determines whether the new filters are net positive or just add complexity.

### Priority 2: Bridge the Two Orchestrators

The `QuantOrchestrator` has genuine improvements (HMM regime detection, OU mean reversion, TSMOM, Kelly sizing) that the main `Orchestrator` doesn't use. Two paths:

**Option A — Feed quant_core signals into the existing Orchestrator as an additional edge:**
- Create a 7th edge that wraps `SignalAggregator.generate_signal()` and votes based on the blended quant signal
- This is lower-risk: the existing 6-edge voting system stays intact, the quant signal is one more input

**Option B — Replace the Orchestrator with the QuantOrchestrator:**
- More disruptive but potentially higher quality (Kelly sizing, HRP allocation, proper risk controller)
- Requires extensive validation since the QuantOrchestrator has no live-trading track record

Recommend **Option A** first, then evaluate Option B after backtesting.

### Priority 3: Parameter Sensitivity Analysis

Run sensitivity sweeps on the new hardcoded thresholds:
- `min_reentry_bars`: test 6, 12, 20, 30
- `min_atr_pct` / `max_atr_pct`: test tighter and wider bands
- `max_opposition_score`: test 0.2, 0.35, 0.5
- `min_regime_confidence`: test 0.3, 0.5, 0.7
- Position sizing base and coefficients

This will reveal which parameters are stable across market conditions vs. which are fragile.

### Priority 4: Clean Up Strategy Sprawl

Narrow the 8+ strategies down to 2-3 with the best risk-adjusted returns:
- Pure Momentum and Mean Reversion have the clearest theoretical backing
- Pairs Trading has the best reported Sharpe (1.9-2.4) but requires correlated pairs
- Options strategies (Iron Condor, Theta, VRP) are a different asset class and should be a separate track

### Priority 5: End-to-End Integration Test

Create a test that:
1. Spins up a `SimBroker` (already exists in `broker/sim.py`)
2. Runs the full `Orchestrator` + `AutoRunner` loop for a simulated day
3. Verifies trades are created, risk limits are respected, positions are closed by EOD
4. Runs deterministically from recorded market data

This is the most important missing test — it validates the real trading path, not just individual components.

---

## Files Changed (Quick Reference)

| File | Change Type | Impact |
|------|-------------|--------|
| `trading_algo/orchestrator/strategy.py` | Modified | Core trading logic — new filters, sizing, cooldown |
| `trading_algo/orchestrator/edges/time_of_day.py` | Modified | Bug fix — direction-aware voting |
| `trading_algo/quant_core/engine/orchestrator.py` | New | Standalone quant orchestrator |
| `trading_algo/quant_core/engine/trading_context.py` | New | Backtest context with FIFO lot matching |
| `trading_algo/quant_core/engine/signal_aggregator.py` | New | Multi-model signal combination |
| `trading_algo/quant_core/engine/portfolio_manager.py` | New | Kelly/HRP position sizing |
| `tests/test_return_improvements.py` | New | 10 tests, all passing |
| + 68 other new files | New | Full quant framework |
