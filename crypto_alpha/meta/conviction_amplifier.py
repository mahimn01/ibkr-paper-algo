"""
Conviction Amplifier — multi-edge agreement detector.

When multiple independent edges agree on direction for the same symbol,
the resulting position size is amplified.  High agreement is rare in
practice, so when it *does* occur it represents an exceptionally strong
signal.  Conversely, if edges strongly disagree (roughly half long, half
short), the position is actively reduced to avoid whipsaws.

Standalone — no dependencies on other meta-innovations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


class ConvictionAmplifier:
    """Compute per-symbol position multipliers from cross-edge agreement.

    Parameters
    ----------
    n_edges : int
        Total number of edge engines in the system (used for context
        but not strictly required — agreement is computed from signals
        actually provided).
    agreement_threshold : int
        Minimum confidence-weighted edge count for base amplification.
    strong_agreement_threshold : int
        Minimum confidence-weighted edge count for strong amplification.
    base_amplifier : float
        Multiplier applied when agreement exceeds ``agreement_threshold``.
    strong_amplifier : float
        Multiplier applied when agreement exceeds ``strong_agreement_threshold``.
    max_amplifier : float
        Hard ceiling on the amplifier (safety cap).
    disagreement_penalty : float
        Multiplier applied when edges are roughly split in direction,
        i.e. strong disagreement.  Should be < 1.0 to reduce sizing.
    """

    def __init__(
        self,
        n_edges: int = 9,
        agreement_threshold: int = 5,
        strong_agreement_threshold: int = 7,
        base_amplifier: float = 1.5,
        strong_amplifier: float = 2.5,
        max_amplifier: float = 3.0,
        disagreement_penalty: float = 0.3,
    ):
        self._n_edges = n_edges
        self._agreement_threshold = agreement_threshold
        self._strong_agreement_threshold = strong_agreement_threshold
        self._base_amplifier = base_amplifier
        self._strong_amplifier = strong_amplifier
        self._max_amplifier = max_amplifier
        self._disagreement_penalty = disagreement_penalty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_amplifier(
        self,
        signals: List[Tuple[str, str, int, float]],
    ) -> Dict[str, float]:
        """Compute a position-size amplifier for each symbol.

        Parameters
        ----------
        signals : List[Tuple[str, str, int, float]]
            Each element is ``(edge_name, symbol, direction, confidence)``
            where *direction* is ``+1`` (long), ``-1`` (short), or ``0``
            (neutral) and *confidence* is in ``[0, 1]``.

        Returns
        -------
        Dict[str, float]
            Mapping of symbol -> amplifier multiplier.  Values range from
            ``disagreement_penalty`` (strong disagreement) up to
            ``max_amplifier`` (very strong agreement).

        Logic
        -----
        1. Group signals by symbol.
        2. For each symbol, bucket edges by direction and sum confidence
           weights per bucket.
        3. The *dominant* direction is the bucket with the highest
           confidence-weighted count.
        4. ``weighted_agreement`` = confidence-weighted count of edges in
           the dominant direction.
        5. If the opposite direction has a similar weight (within 40 % of
           dominant), treat as **strong disagreement** and apply penalty.
        6. Otherwise, apply amplification based on thresholds.
        """
        # Group by symbol
        by_symbol: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
        for edge_name, symbol, direction, confidence in signals:
            if direction != 0:
                by_symbol[symbol].append((edge_name, direction, confidence))

        amplifiers: Dict[str, float] = {}

        for symbol, sym_signals in by_symbol.items():
            if not sym_signals:
                amplifiers[symbol] = 1.0
                continue

            # Bucket by direction
            long_weight = 0.0
            short_weight = 0.0
            long_count = 0
            short_count = 0

            for _edge, direction, confidence in sym_signals:
                if direction > 0:
                    long_weight += confidence
                    long_count += 1
                elif direction < 0:
                    short_weight += confidence
                    short_count += 1

            # Determine dominant direction and agreement metrics
            if long_weight >= short_weight:
                dominant_weight = long_weight
                opposite_weight = short_weight
            else:
                dominant_weight = short_weight
                opposite_weight = long_weight

            # Check for strong disagreement: opposite side has >= 40% of
            # dominant side's weight AND at least 2 edges disagree
            min_disagree_edges = min(long_count, short_count)
            if (
                dominant_weight > 0
                and opposite_weight >= 0.4 * dominant_weight
                and min_disagree_edges >= 2
            ):
                amplifiers[symbol] = self._disagreement_penalty
                continue

            # Apply amplification from agreement
            weighted_agreement = dominant_weight

            if weighted_agreement >= self._strong_agreement_threshold:
                amp = self._strong_amplifier
            elif weighted_agreement >= self._agreement_threshold:
                amp = self._base_amplifier
            else:
                amp = 1.0

            amplifiers[symbol] = min(amp, self._max_amplifier)

        return amplifiers

    def get_agreement_report(
        self,
        signals: List[Tuple[str, str, int, float]],
    ) -> Dict:
        """Return a detailed agreement analysis per symbol.

        Parameters
        ----------
        signals : List[Tuple[str, str, int, float]]
            Same format as :meth:`compute_amplifier`.

        Returns
        -------
        Dict with keys:
            - ``per_symbol``: Dict[str, Dict] with per-symbol breakdown
              containing ``long_edges``, ``short_edges``, ``long_weight``,
              ``short_weight``, ``dominant_direction``, ``agreement_level``,
              and ``amplifier``.
            - ``summary``: Dict with ``total_signals``, ``symbols_amplified``,
              ``symbols_penalized``.
        """
        # Group by symbol
        by_symbol: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
        for edge_name, symbol, direction, confidence in signals:
            if direction != 0:
                by_symbol[symbol].append((edge_name, direction, confidence))

        amplifiers = self.compute_amplifier(signals)

        per_symbol: Dict[str, Dict] = {}
        symbols_amplified = 0
        symbols_penalized = 0

        for symbol, sym_signals in by_symbol.items():
            long_edges: List[str] = []
            short_edges: List[str] = []
            long_weight = 0.0
            short_weight = 0.0

            for edge_name, direction, confidence in sym_signals:
                if direction > 0:
                    long_edges.append(edge_name)
                    long_weight += confidence
                else:
                    short_edges.append(edge_name)
                    short_weight += confidence

            if long_weight >= short_weight:
                dominant = "long"
            else:
                dominant = "short"

            amp = amplifiers.get(symbol, 1.0)

            if amp > 1.0:
                level = "strong" if amp >= self._strong_amplifier else "moderate"
                symbols_amplified += 1
            elif amp < 1.0:
                level = "disagreement"
                symbols_penalized += 1
            else:
                level = "neutral"

            per_symbol[symbol] = {
                "long_edges": long_edges,
                "short_edges": short_edges,
                "long_weight": round(long_weight, 3),
                "short_weight": round(short_weight, 3),
                "dominant_direction": dominant,
                "agreement_level": level,
                "amplifier": amp,
            }

        return {
            "per_symbol": per_symbol,
            "summary": {
                "total_signals": len(signals),
                "symbols_amplified": symbols_amplified,
                "symbols_penalized": symbols_penalized,
            },
        }
