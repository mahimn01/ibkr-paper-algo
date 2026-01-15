"""
Alpha Mutator: Generate new factor variations using minimal LLM assistance.

This is the ONLY module that uses LLM, and only when:
1. Mathematical mutation methods have been exhausted
2. Factor pool health is critically low
3. User explicitly requests creative mutations

Mutation strategies (non-LLM, preferred):
1. Parameter perturbation
2. Timeframe shifting
3. Indicator combination
4. Regime conditioning
5. Decay function modification

LLM mutation (last resort):
- Generate novel factor formulas
- Suggest unconventional data combinations
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from trading_algo.rat.alpha.tracker import AlphaFactor, AlphaTracker, DecayStage


class MutationType(Enum):
    """Types of factor mutations."""

    # Mathematical mutations (no LLM)
    PARAMETER_SHIFT = auto()        # Adjust lookback, threshold
    TIMEFRAME_CHANGE = auto()       # Different aggregation period
    INDICATOR_COMBINE = auto()      # Combine with another indicator
    REGIME_CONDITION = auto()       # Add regime filter
    DECAY_ADJUST = auto()           # Modify signal decay
    INVERSE = auto()                # Flip signal direction
    NORMALIZE = auto()              # Change normalization method

    # LLM-assisted mutations (minimal use)
    LLM_FORMULA = auto()            # Novel formula generation
    LLM_COMBINATION = auto()        # Creative factor combination


@dataclass
class MutationResult:
    """Result of a factor mutation."""

    original_name: str
    new_name: str
    mutation_type: MutationType
    compute_fn: Callable[[Dict], float]
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    used_llm: bool = False


@dataclass
class FactorTemplate:
    """Template for generating factor variations."""

    name: str
    base_formula: str
    parameters: Dict[str, Tuple[float, float, float]]  # param: (min, max, default)
    compute_fn_template: str  # Python code template


class AlphaMutator:
    """
    Generate new alpha factors through mutation.

    Prioritizes mathematical mutations over LLM:
    1. Try all mathematical mutations first
    2. Only use LLM when factor pool is critically depleted
    3. LLM generates formulas, not trading decisions
    """

    # Standard factor templates for mathematical mutation
    FACTOR_TEMPLATES = {
        "momentum": FactorTemplate(
            name="momentum",
            base_formula="(price[t] - price[t-n]) / price[t-n]",
            parameters={
                "lookback": (5, 60, 20),
                "smoothing": (1, 10, 3),
            },
            compute_fn_template="""
def compute(data):
    prices = data.get('prices', [])
    n = {lookback}
    smooth = {smoothing}
    if len(prices) < n + smooth:
        return 0.0
    raw = (prices[-1] - prices[-n-1]) / prices[-n-1]
    # Apply smoothing
    if smooth > 1:
        recent = [(prices[-i] - prices[-i-n]) / prices[-i-n] for i in range(1, smooth+1)]
        return sum(recent) / len(recent)
    return raw
""",
        ),
        "mean_reversion": FactorTemplate(
            name="mean_reversion",
            base_formula="(ma[n] - price) / std[n]",
            parameters={
                "lookback": (10, 100, 20),
                "entry_zscore": (1.0, 3.0, 2.0),
            },
            compute_fn_template="""
def compute(data):
    prices = data.get('prices', [])
    n = {lookback}
    threshold = {entry_zscore}
    if len(prices) < n:
        return 0.0
    window = prices[-n:]
    ma = sum(window) / n
    std = (sum((p - ma)**2 for p in window) / n) ** 0.5
    if std == 0:
        return 0.0
    zscore = (ma - prices[-1]) / std
    if abs(zscore) < threshold:
        return 0.0
    return zscore / threshold  # Normalized signal
""",
        ),
        "rsi": FactorTemplate(
            name="rsi",
            base_formula="100 - 100/(1 + avg_gain/avg_loss)",
            parameters={
                "period": (7, 28, 14),
                "overbought": (65, 85, 70),
                "oversold": (15, 35, 30),
            },
            compute_fn_template="""
def compute(data):
    prices = data.get('prices', [])
    n = {period}
    ob = {overbought}
    os = {oversold}
    if len(prices) < n + 1:
        return 0.0
    changes = [prices[i] - prices[i-1] for i in range(-n, 0)]
    gains = [c for c in changes if c > 0]
    losses = [-c for c in changes if c < 0]
    avg_gain = sum(gains) / n if gains else 0.0001
    avg_loss = sum(losses) / n if losses else 0.0001
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    if rsi > ob:
        return -(rsi - ob) / (100 - ob)  # Overbought = short signal
    elif rsi < os:
        return (os - rsi) / os  # Oversold = long signal
    return 0.0
""",
        ),
        "volatility_breakout": FactorTemplate(
            name="volatility_breakout",
            base_formula="(price - ma) / atr if |price - ma| > k * atr else 0",
            parameters={
                "atr_period": (10, 30, 14),
                "breakout_mult": (1.5, 3.0, 2.0),
            },
            compute_fn_template="""
def compute(data):
    prices = data.get('prices', [])
    highs = data.get('highs', prices)
    lows = data.get('lows', prices)
    n = {atr_period}
    k = {breakout_mult}
    if len(prices) < n + 1:
        return 0.0
    # Compute ATR
    trs = []
    for i in range(-n, 0):
        tr = max(highs[i] - lows[i], abs(highs[i] - prices[i-1]), abs(lows[i] - prices[i-1]))
        trs.append(tr)
    atr = sum(trs) / n
    if atr == 0:
        return 0.0
    ma = sum(prices[-n:]) / n
    deviation = (prices[-1] - ma) / atr
    if abs(deviation) > k:
        return deviation / k  # Breakout signal
    return 0.0
""",
        ),
        "order_flow_imbalance": FactorTemplate(
            name="order_flow_imbalance",
            base_formula="(buy_vol - sell_vol) / total_vol",
            parameters={
                "lookback": (10, 100, 50),
                "threshold": (0.3, 0.7, 0.5),
            },
            compute_fn_template="""
def compute(data):
    buy_vol = data.get('buy_volume', [])
    sell_vol = data.get('sell_volume', [])
    n = {lookback}
    thresh = {threshold}
    if len(buy_vol) < n or len(sell_vol) < n:
        return 0.0
    total_buy = sum(buy_vol[-n:])
    total_sell = sum(sell_vol[-n:])
    total = total_buy + total_sell
    if total == 0:
        return 0.0
    imbalance = (total_buy - total_sell) / total
    if abs(imbalance) < thresh:
        return 0.0
    return imbalance
""",
        ),
    }

    def __init__(
        self,
        tracker: AlphaTracker,
        llm_client: Optional[Any] = None,
        enable_llm: bool = False,
        llm_cooldown_hours: float = 24.0,
    ):
        self.tracker = tracker
        self.llm_client = llm_client
        self.enable_llm = enable_llm and llm_client is not None
        self.llm_cooldown_hours = llm_cooldown_hours

        # Track mutations
        self._mutation_history: List[MutationResult] = []
        self._last_llm_use: Optional[datetime] = None

        # Random seed for reproducibility in backtests
        self._rng = random.Random(42)

    def mutate_factor(
        self,
        factor_name: str,
        mutation_type: Optional[MutationType] = None,
        force_llm: bool = False,
    ) -> Optional[MutationResult]:
        """
        Mutate an existing factor to create a variation.

        Args:
            factor_name: Name of factor to mutate
            mutation_type: Specific mutation to apply (random if None)
            force_llm: Force LLM mutation even if not needed
        """
        if factor_name not in self.tracker._factors:
            return None

        factor = self.tracker._factors[factor_name]

        # Select mutation type
        if mutation_type is None:
            # Prefer mathematical mutations
            math_mutations = [
                MutationType.PARAMETER_SHIFT,
                MutationType.TIMEFRAME_CHANGE,
                MutationType.DECAY_ADJUST,
                MutationType.INVERSE,
                MutationType.NORMALIZE,
            ]
            mutation_type = self._rng.choice(math_mutations)

        # Apply mutation
        if mutation_type == MutationType.PARAMETER_SHIFT:
            return self._mutate_parameter_shift(factor)
        elif mutation_type == MutationType.TIMEFRAME_CHANGE:
            return self._mutate_timeframe(factor)
        elif mutation_type == MutationType.INVERSE:
            return self._mutate_inverse(factor)
        elif mutation_type == MutationType.DECAY_ADJUST:
            return self._mutate_decay(factor)
        elif mutation_type == MutationType.NORMALIZE:
            return self._mutate_normalize(factor)
        elif mutation_type in (MutationType.LLM_FORMULA, MutationType.LLM_COMBINATION):
            if force_llm or self._should_use_llm():
                return self._mutate_with_llm(factor, mutation_type)

        return None

    def generate_new_factors(
        self,
        count: int = 3,
        template_name: Optional[str] = None,
    ) -> List[MutationResult]:
        """
        Generate new factors from templates with random parameters.

        Pure mathematical - no LLM required.
        """
        results = []

        templates = [template_name] if template_name else list(self.FACTOR_TEMPLATES.keys())

        for _ in range(count):
            template_key = self._rng.choice(templates)
            template = self.FACTOR_TEMPLATES[template_key]

            # Randomize parameters
            params = {}
            for param_name, (min_val, max_val, default) in template.parameters.items():
                # Random value within range
                params[param_name] = self._rng.uniform(min_val, max_val)

            # Generate compute function
            code = template.compute_fn_template.format(**params)

            # Create callable
            local_ns: Dict[str, Any] = {}
            exec(code, {"__builtins__": {"sum": sum, "abs": abs, "max": max, "min": min}}, local_ns)
            compute_fn = local_ns["compute"]

            # Generate unique name
            param_str = "_".join(f"{k}{v:.1f}" for k, v in params.items())
            new_name = f"{template.name}_{param_str}_{self._rng.randint(1000, 9999)}"

            result = MutationResult(
                original_name=template.name,
                new_name=new_name,
                mutation_type=MutationType.PARAMETER_SHIFT,
                compute_fn=compute_fn,
                description=f"Generated {template.name} with params: {params}",
                used_llm=False,
            )

            results.append(result)

            # Register with tracker
            self.tracker.register_factor(new_name, compute_fn)

        self._mutation_history.extend(results)
        return results

    def _mutate_parameter_shift(self, factor: AlphaFactor) -> Optional[MutationResult]:
        """Mutate by shifting parameters."""
        # Wrap original function with parameter perturbation
        original_fn = factor.compute_fn

        # Random scaling factor
        scale = self._rng.uniform(0.7, 1.3)

        def mutated_compute(data: Dict) -> float:
            result = original_fn(data)
            return result * scale

        new_name = f"{factor.name}_scaled_{scale:.2f}"

        result = MutationResult(
            original_name=factor.name,
            new_name=new_name,
            mutation_type=MutationType.PARAMETER_SHIFT,
            compute_fn=mutated_compute,
            description=f"Parameter scaled by {scale:.2f}",
            used_llm=False,
        )

        self.tracker.register_factor(new_name, mutated_compute)
        self._mutation_history.append(result)
        return result

    def _mutate_timeframe(self, factor: AlphaFactor) -> Optional[MutationResult]:
        """Mutate by changing effective timeframe."""
        original_fn = factor.compute_fn

        # Subsample factor - use every nth data point effect
        skip = self._rng.choice([2, 3, 5])

        def mutated_compute(data: Dict) -> float:
            # Thin out the data
            modified_data = {}
            for key, value in data.items():
                if isinstance(value, list) and len(value) > skip:
                    modified_data[key] = value[::skip]
                else:
                    modified_data[key] = value
            return original_fn(modified_data)

        new_name = f"{factor.name}_tf{skip}x"

        result = MutationResult(
            original_name=factor.name,
            new_name=new_name,
            mutation_type=MutationType.TIMEFRAME_CHANGE,
            compute_fn=mutated_compute,
            description=f"Timeframe extended {skip}x",
            used_llm=False,
        )

        self.tracker.register_factor(new_name, mutated_compute)
        self._mutation_history.append(result)
        return result

    def _mutate_inverse(self, factor: AlphaFactor) -> Optional[MutationResult]:
        """Create inverse signal."""
        original_fn = factor.compute_fn

        def mutated_compute(data: Dict) -> float:
            return -original_fn(data)

        new_name = f"{factor.name}_inv"

        result = MutationResult(
            original_name=factor.name,
            new_name=new_name,
            mutation_type=MutationType.INVERSE,
            compute_fn=mutated_compute,
            description="Inverted signal direction",
            used_llm=False,
        )

        self.tracker.register_factor(new_name, mutated_compute)
        self._mutation_history.append(result)
        return result

    def _mutate_decay(self, factor: AlphaFactor) -> Optional[MutationResult]:
        """Add exponential decay to signal."""
        original_fn = factor.compute_fn

        decay_rate = self._rng.uniform(0.8, 0.99)
        self._decay_memory: Dict[str, float] = {}

        def mutated_compute(data: Dict) -> float:
            raw = original_fn(data)
            key = factor.name
            prev = self._decay_memory.get(key, 0.0)
            # Exponential smoothing
            smoothed = decay_rate * prev + (1 - decay_rate) * raw
            self._decay_memory[key] = smoothed
            return smoothed

        new_name = f"{factor.name}_decay{decay_rate:.2f}"

        result = MutationResult(
            original_name=factor.name,
            new_name=new_name,
            mutation_type=MutationType.DECAY_ADJUST,
            compute_fn=mutated_compute,
            description=f"Added decay with rate {decay_rate:.2f}",
            used_llm=False,
        )

        self.tracker.register_factor(new_name, mutated_compute)
        self._mutation_history.append(result)
        return result

    def _mutate_normalize(self, factor: AlphaFactor) -> Optional[MutationResult]:
        """Change normalization method."""
        original_fn = factor.compute_fn

        # Rolling normalization window
        window = self._rng.randint(10, 50)
        self._norm_history: List[float] = []

        def mutated_compute(data: Dict) -> float:
            raw = original_fn(data)
            self._norm_history.append(raw)
            if len(self._norm_history) > window:
                self._norm_history.pop(0)

            if len(self._norm_history) < 3:
                return raw

            # Z-score normalization
            mean = sum(self._norm_history) / len(self._norm_history)
            var = sum((x - mean) ** 2 for x in self._norm_history) / len(self._norm_history)
            std = math.sqrt(var) if var > 0 else 1.0

            return (raw - mean) / std

        new_name = f"{factor.name}_znorm{window}"

        result = MutationResult(
            original_name=factor.name,
            new_name=new_name,
            mutation_type=MutationType.NORMALIZE,
            compute_fn=mutated_compute,
            description=f"Z-score normalized over {window} periods",
            used_llm=False,
        )

        self.tracker.register_factor(new_name, mutated_compute)
        self._mutation_history.append(result)
        return result

    def _should_use_llm(self) -> bool:
        """Determine if LLM should be used based on conditions."""
        if not self.enable_llm:
            return False

        # Check cooldown
        if self._last_llm_use:
            hours_since = (datetime.now() - self._last_llm_use).total_seconds() / 3600
            if hours_since < self.llm_cooldown_hours:
                return False

        # Check if factor pool is critically low
        state = self.tracker.analyze()
        return state.needs_mutation and len(state.active_factors) < 2

    def _mutate_with_llm(
        self,
        factor: AlphaFactor,
        mutation_type: MutationType,
    ) -> Optional[MutationResult]:
        """
        Use LLM to generate novel factor mutation.

        MINIMAL USE - only when mathematical methods exhausted.
        """
        if not self.enable_llm or self.llm_client is None:
            return None

        self._last_llm_use = datetime.now()

        # Prepare prompt for LLM
        prompt = f"""Generate a novel variation of this trading factor.

Original factor: {factor.name}
Current performance stage: {factor.current_stage.name}

Requirements:
1. Output ONLY a Python function named 'compute' that takes a 'data' dict parameter
2. The data dict may contain: prices (list), volumes (list), buy_volume (list), sell_volume (list), highs (list), lows (list)
3. Return a float signal between -1 and 1
4. Use only basic math operations (no imports)
5. Must be mathematically different from simple parameter changes

Example format:
def compute(data):
    prices = data.get('prices', [])
    # Your novel logic here
    return signal_value
"""

        try:
            # Call LLM (simplified - actual implementation would use proper client)
            # This is a placeholder - real implementation depends on LLM client interface
            response = self._call_llm(prompt)

            if not response:
                return None

            # Extract and validate code
            code = self._extract_code(response)
            if not code:
                return None

            # Safely compile the function
            local_ns: Dict[str, Any] = {}
            safe_builtins = {
                "sum": sum, "abs": abs, "max": max, "min": min,
                "len": len, "range": range, "zip": zip,
                "float": float, "int": int, "bool": bool,
            }
            exec(code, {"__builtins__": safe_builtins}, local_ns)

            if "compute" not in local_ns:
                return None

            compute_fn = local_ns["compute"]

            # Test the function
            test_data = {"prices": [100.0] * 50}
            test_result = compute_fn(test_data)
            if not isinstance(test_result, (int, float)):
                return None

            new_name = f"{factor.name}_llm_{datetime.now().strftime('%H%M%S')}"

            result = MutationResult(
                original_name=factor.name,
                new_name=new_name,
                mutation_type=mutation_type,
                compute_fn=compute_fn,
                description=f"LLM-generated mutation",
                used_llm=True,
            )

            self.tracker.register_factor(new_name, compute_fn)
            self._mutation_history.append(result)
            return result

        except Exception:
            return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM client. Override for specific implementation."""
        if self.llm_client is None:
            return None

        # Placeholder - actual implementation depends on client interface
        # Could be: self.llm_client.generate(prompt)
        # Or: self.llm_client.chat(prompt)
        try:
            if hasattr(self.llm_client, "generate"):
                return self.llm_client.generate(prompt)
            elif hasattr(self.llm_client, "chat"):
                return self.llm_client.chat(prompt)
            elif callable(self.llm_client):
                return self.llm_client(prompt)
        except Exception:
            pass
        return None

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Look for def compute
        if "def compute" in response:
            start = response.find("def compute")
            # Find the end of the function (next def or end of string)
            lines = response[start:].split("\n")
            code_lines = []
            in_function = False
            for line in lines:
                if line.strip().startswith("def compute"):
                    in_function = True
                    code_lines.append(line)
                elif in_function:
                    if line.strip() and not line.startswith((" ", "\t")):
                        break
                    code_lines.append(line)
            return "\n".join(code_lines)

        return None

    def get_mutation_stats(self) -> Dict[str, Any]:
        """Get statistics on mutations performed."""
        total = len(self._mutation_history)
        llm_count = sum(1 for m in self._mutation_history if m.used_llm)

        by_type: Dict[str, int] = {}
        for m in self._mutation_history:
            type_name = m.mutation_type.name
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total_mutations": total,
            "llm_mutations": llm_count,
            "math_mutations": total - llm_count,
            "by_type": by_type,
            "llm_enabled": self.enable_llm,
            "last_llm_use": self._last_llm_use,
        }
