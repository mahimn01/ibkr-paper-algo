# Archive

This directory contains deprecated code that has been superseded by the Orchestrator trading system.

## Contents

### run_scripts/
Old entry point scripts that have been replaced by the unified `run.py`:
- `run_chameleon_*.py` - Chameleon strategy runners
- `run_daytrader_*.py` - Day trader runners
- `run_*_backtest.py` - Various backtesting scripts
- `run_auto_daytrader.py` - Old auto day trader

### strategies/rat/
The RAT (Reflexive Attention Trader) system - our first-generation algorithmic trading framework:
- `chameleon_daytrader.py` - Chameleon v3 day trading strategy
- `chameleon_strategy.py` - Original Chameleon strategy
- Momentum signals, risk management, etc.

**Why archived:** Single-indicator approach without market context. Replaced by multi-edge Orchestrator.

### strategies/llm/
LLM-based trading experiments:
- Gemini integration for trade decisions
- Chat-based trading interface

**Why archived:** Experimental feature, not part of core Orchestrator system.

### strategies/base/
Original strategy base classes and examples.

**Why archived:** Replaced by Orchestrator's unified architecture.

### strategies/orchestrator.py
The original monolithic Orchestrator implementation (1455 lines).

**Why archived:** Refactored into modular `trading_algo/orchestrator/` package.

## Restoring Archived Code

If you need to use any archived code:

```python
# Import from archive (not recommended for production)
import sys
sys.path.insert(0, 'archive/strategies')
from rat.chameleon_daytrader import ChameleonDayTrader
```

## Migration Notes

The Orchestrator system replaces all archived strategies with a unified, multi-edge approach:

| Old System | Orchestrator Equivalent |
|------------|------------------------|
| Chameleon momentum | Edge 1: Market Regime + Edge 2: Relative Strength |
| RSI/SMA filters | Edge 3: Statistical Extremes |
| ATR stops | Built into Orchestrator with trailing stops |
| Single-stock analysis | Edge 5: Cross-Asset Confirmation |

See `/docs/STRATEGY.md` for full Orchestrator documentation.
