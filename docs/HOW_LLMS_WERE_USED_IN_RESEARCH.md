# How LLMs Were Actually Used in Trading Research

**Critical Finding**: The research showing LLMs failing used them **incorrectly** for trading.

---

## Two Fundamentally Different Patterns

### Pattern A: LLM as Direct Decision Maker (FAILS)

This is what FINSABER, StockBench, FinMem, and FinAgent tested:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PATTERN A: DIRECT TRADING                     │
│                                                                  │
│   Market Data ─────▶ LLM Prompt ─────▶ "BUY" ─────▶ Execute     │
│   (prices, news)    "Should I buy?"   (direct)     Trade        │
│                                                                  │
│   RESULT: Underperforms buy-and-hold in most cases              │
└─────────────────────────────────────────────────────────────────┘
```

**Exact Implementation in Studies:**

```python
# FinMem/FinAgent approach (simplified)
prompt = f"""
Based on:
- Recent news: {news}
- Price history: {prices}
- Your memory of past trades: {memory}

What action should you take?
Output: Buy, Hold, or Sell
"""

response = llm.generate(prompt)  # Returns "Buy" or "Sell" or "Hold"
execute_trade(response)  # Direct execution
```

**StockBench Protocol:**
- Agent receives daily: prices, fundamentals, news
- Agent outputs: buy/sell/hold decision
- Decision directly executed
- Repeated for 82 trading days

**This is EXACTLY what your codebase does in `llm/trader.py`:**

```python
# From trading_algo/llm/trader.py:170-181
_SYSTEM_INSTRUCTIONS = (
    "You are an execution assistant for a PAPER-trading only system. "
    "Return ONLY valid JSON (no markdown). "
    ...
    "Output schema: {\"decisions\":[ ... ]} where each decision is one of:\n"
    "  {\"action\":\"PLACE\",\"reason\":...,\"order\":{...}}"  # <-- DIRECT DECISION
)
```

---

### Pattern B: LLM as Alpha Generator (WORKS)

This is what [AlphaAgent](https://arxiv.org/html/2502.16789v2), [Chain-of-Alpha](https://www.alphaxiv.org/overview/2508.06312v2), and the [hybrid approach study](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-41061-5) tested:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PATTERN B: ALPHA GENERATION                   │
│                                                                  │
│   Market Data ─▶ LLM Prompt ─▶ "Formula" ─▶ Quant System ─▶ Trade│
│   (historical)  "Generate     (factor)     (evaluates,         │
│                  alpha factor"             executes)            │
│                                                                  │
│   RESULT: 75% IC improvement, stable multi-year performance     │
└─────────────────────────────────────────────────────────────────┘
```

**Exact Implementation:**

```python
# AlphaAgent/Chain-of-Alpha approach (simplified)
prompt = f"""
Based on market data patterns, generate a formulaic alpha factor.
Current alpha pool: {existing_factors}
Backtest feedback: {last_factor_ic, last_factor_sharpe}

Generate a mathematical formula like:
  (close - sma(close, 20)) / std(close, 20)

Focus on factors with high Information Coefficient (IC).
"""

response = llm.generate(prompt)  # Returns: "(volume * price_momentum) / volatility"

# LLM does NOT decide to trade - it generates a FACTOR
factor = parse_formula(response)

# Traditional quant system uses the factor
signal = evaluate_factor(factor, current_data)  # Returns: 0.73

# Traditional algorithm decides based on signal
if signal > threshold:
    execute_trade("BUY")  # Deterministic decision
```

**Key Difference:**
- Pattern A: LLM outputs "BUY AAPL" → executed
- Pattern B: LLM outputs `"(close - sma_20) / volatility"` → traditional system evaluates → decides

---

## Why Pattern A Fails

The [FINSABER study](https://arxiv.org/abs/2505.07078) identified specific failure modes:

| Market Condition | LLM Behavior | Result |
|------------------|--------------|--------|
| Bull market (SPY +15%) | Too conservative, holds cash | Underperforms 10-20% |
| Bear market (SPY -15%) | Too aggressive, over-trades | Loses 20-30% |
| Sideways market | Inconsistent, noise-driven | Break-even minus costs |

**Root Causes:**
1. LLMs trained on text, not market dynamics
2. No concept of position sizing mathematics
3. Hallucinate confidence in uncertain situations
4. Cannot process tick-level data fast enough
5. Each decision is independent (no state machine)

---

## Why Pattern B Works

The [AlphaAgent study](https://dl.acm.org/doi/10.1145/3711896.3736838) showed:

| Metric | Traditional Factors | LLM-Generated Factors |
|--------|--------------------|-----------------------|
| IC after 1 year | 0.015 | 0.022 |
| IC after 4 years | ~0 (decay) | 0.020-0.025 (stable) |
| RankIC | Degrading | Maintained |

**Why This Works:**
1. LLM generates ideas, not decisions
2. Mathematical formulas are backtestable
3. Traditional system handles execution
4. LLM creativity + algorithmic rigor
5. Human-interpretable factors

---

## What This Means for Your Codebase

### Current State (Pattern A - Problematic)

```python
# trading_algo/llm/trader.py:107-111
raw = self.client.generate(
    prompt=prompt,
    system=_SYSTEM_INSTRUCTIONS,  # "Return {action: PLACE/MODIFY/CANCEL}"
)
decisions = parse_llm_decisions(raw)  # LLM decides directly
```

### Recommended State (Pattern B - Better)

```python
# Proposed: trading_algo/llm/alpha_generator.py

class AlphaGenerator:
    def generate_factors(self, market_context: dict) -> list[AlphaFactor]:
        """LLM generates factors, NOT trading decisions."""

        prompt = f"""
        Current market state: {market_context}
        Existing factors in use: {self.factor_pool}
        Recent factor performance (IC): {self.backtest_results}

        Generate 3 new candidate alpha factors as mathematical formulas.
        Format: ["(close - ema_20) / atr_14", ...]

        Focus on factors that might capture:
        - Momentum reversals
        - Volatility breakouts
        - Mean reversion signals
        """

        response = self.client.generate(prompt)
        return [AlphaFactor.from_formula(f) for f in parse_formulas(response)]


class HybridTrader:
    def __init__(self):
        self.alpha_generator = AlphaGenerator()
        self.factor_evaluator = FactorEvaluator()  # Traditional quant

    def on_tick(self, ctx: StrategyContext) -> list[TradeIntent]:
        # Periodically refresh factors (e.g., daily)
        if self.should_refresh_factors():
            new_factors = self.alpha_generator.generate_factors(ctx)
            self.factor_pool.update(new_factors)

        # Traditional system evaluates factors and decides
        signals = self.factor_evaluator.evaluate(self.factor_pool, ctx)

        # Deterministic trading logic
        if signals.aggregate_score > self.threshold:
            return [TradeIntent(BUY, ctx.symbol, self.position_size)]
        return []
```

---

## Summary Table

| Aspect | Pattern A (Direct) | Pattern B (Alpha Gen) |
|--------|-------------------|----------------------|
| **LLM Output** | "BUY AAPL" | "(close-sma)/std" |
| **Execution** | LLM decides | Algorithm decides |
| **Backtestable** | No (non-deterministic) | Yes (formula-based) |
| **Speed** | 500-3000ms per decision | Once per day/week |
| **Research Result** | Fails to beat baseline | 75% IC improvement |
| **Your Current Code** | `llm/trader.py` | Not implemented |

---

## The Honest Conclusion

**The research that showed LLMs failing tested them doing something they're bad at** (real-time trading decisions).

**The research that showed LLMs succeeding tested them doing something they're good at** (generating mathematical hypotheses).

Your codebase currently implements Pattern A. To maximize profit potential, implement Pattern B.

---

## Sources

- [FINSABER: LLM Financial Investing](https://arxiv.org/abs/2505.07078) - Pattern A testing
- [StockBench Benchmark](https://stockbench.github.io/) - Pattern A testing
- [FinMem Architecture](https://github.com/pipiku915/FinMem-LLM-StockTrading) - Pattern A implementation
- [AlphaAgent: Alpha Mining](https://arxiv.org/html/2502.16789v2) - Pattern B success
- [Chain-of-Alpha](https://www.alphaxiv.org/overview/2508.06312v2) - Pattern B success
- [Hybrid Alpha Discovery](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-41061-5) - Pattern B success
