"""
Automated Pattern Discovery Engine

Scans millions of feature combinations to find statistically reliable alpha
signals, then validates each through 6 layers of rigorous testing. This is
the core signal factory: it generates candidates exhaustively, subjects them
to in-sample quality checks, out-of-sample walk-forward validation, multiple
testing correction, orthogonality screening, and alpha-decay analysis.

Methodology references:
    - Harvey, Liu & Zhu (2016): "...and the Cross-Section of Expected Returns"
      (multiple testing framework, t-stat > 3.0 threshold)
    - Benjamini & Hochberg (1995): "Controlling the False Discovery Rate"
    - Bailey, Borwein, Lopez de Prado & Zhu (2014): "Probability of Backtest
      Overfitting" (PBO framework)
    - Lopez de Prado (2018): "Advances in Financial Machine Learning"
      (purged walk-forward, combinatorial cross-validation)
    - Chordia, Goyal & Saretto (2020): "Anomalies and False Discoveries"

Key design principles:
    1. Exhaustive generation -- every plausible combination is tested.
    2. Conservative validation -- 6 independent gates reduce false positives.
    3. NaN-safe computation -- all correlations mask missing data.
    4. Numerical stability -- epsilon guards, z-score normalisation.
    5. Scalable -- vectorised NumPy wherever possible; candidate cap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from trading_algo.quant_core.utils.constants import (
    EPSILON,
    MAX_PBO_THRESHOLD,
    ML_PURGE_WINDOW,
    T_STAT_THRESHOLD,
)
from trading_algo.quant_core.utils.math_utils import zscore
from trading_algo.quant_core.validation.pbo import PBOCalculator

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SignalCandidate:
    """
    A candidate signal constructed from one or two features.

    Each candidate specifies how to combine raw features (single, multiply,
    ratio, difference), the lookback window used for rolling normalisation,
    and the expected holding period.  Validation results are accumulated
    in-place as the candidate passes through each gate.

    Attributes:
        name: Human-readable descriptor (auto-generated from components).
        feature_a: Name of the first (or only) input feature.
        feature_b: Name of the second feature for interaction signals.
        combination_type: One of ``"single"``, ``"multiply"``, ``"ratio"``,
            ``"difference"``.
        lookback_window: Window for rolling z-score normalisation.
        holding_period: Expected forward holding period in trading days.
        ic: Spearman rank correlation with forward returns (in-sample).
        ic_std: Standard deviation of rolling IC measurements.
        ic_ir: Information Coefficient Information Ratio (IC / IC_std).
        oos_ic: Average out-of-sample IC across walk-forward folds.
        t_stat: T-statistic for the in-sample IC.
        adjusted_pvalue: P-value after multiple-testing correction.
        max_correlation_to_existing: Highest absolute correlation with any
            previously accepted signal.
        decay_rate: Estimated alpha decay (1 - IC_last / IC_first).
        pbo: Probability of Backtest Overfitting.
        passed_all_tests: ``True`` only if all 6 validation stages pass.
        rejection_reason: Human-readable reason for failure, or ``None``.
    """

    name: str
    feature_a: str
    feature_b: Optional[str]
    combination_type: str
    lookback_window: int
    holding_period: int

    # Validation results (populated during scan)
    ic: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    oos_ic: float = 0.0
    t_stat: float = 0.0
    adjusted_pvalue: float = 1.0
    max_correlation_to_existing: float = 0.0
    decay_rate: float = 0.0
    pbo: float = 1.0

    # Status
    passed_all_tests: bool = False
    rejection_reason: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Complete result of validating a single ``SignalCandidate``.

    Attributes:
        candidate: The candidate that was evaluated.
        passed: ``True`` if the candidate passed every stage.
        stage_results: Mapping from stage name to pass/fail boolean.
        metrics: All numeric metrics collected during validation.
        signal_values: The constructed signal time series, retained only
            for candidates that pass all stages.
    """

    candidate: SignalCandidate
    passed: bool
    stage_results: Dict[str, bool]
    metrics: Dict[str, float]
    signal_values: Optional[NDArray[np.float64]] = None


@dataclass
class ScannerConfig:
    """
    Configuration for :class:`PatternScanner`.

    All thresholds are set conservatively by default so that the scanner
    produces a low false-positive rate at the cost of some true positives.

    Attributes:
        lookback_windows: Rolling z-score windows to try.
        holding_periods: Forward return horizons to evaluate.
        combination_types: Feature combination operators.
        max_interaction_features: Top-N features (by single-feature IC)
            admitted to pairwise interaction generation.
        min_ic: Minimum absolute in-sample IC to survive Stage 1.
        min_ic_ir: Minimum IC Information Ratio for Stage 2.
        min_oos_ic: Minimum average out-of-sample IC for Stage 3.
        max_adjusted_pvalue: Maximum adjusted p-value for Stage 4.
        max_correlation_to_existing: Orthogonality ceiling for Stage 5.
        max_decay_rate: Maximum alpha decay for Stage 6.
        max_pbo: Maximum Probability of Backtest Overfitting.
        n_folds: Walk-forward fold count.
        purge_days: Gap between train and test sets in walk-forward.
        correction_method: Multiple-testing correction method.
        max_candidates_per_run: Hard cap on total candidates evaluated.
        min_observations: Minimum time-series length required to scan.
    """

    # Signal construction
    lookback_windows: List[int] = field(
        default_factory=lambda: [5, 10, 20, 60, 120]
    )
    holding_periods: List[int] = field(
        default_factory=lambda: [1, 5, 10, 20]
    )
    combination_types: List[str] = field(
        default_factory=lambda: ["single", "multiply", "ratio", "difference"]
    )
    max_interaction_features: int = 20

    # Validation thresholds
    min_ic: float = 0.02
    min_ic_ir: float = 0.5
    min_oos_ic: float = 0.01
    max_adjusted_pvalue: float = 0.01
    max_correlation_to_existing: float = 0.30
    max_decay_rate: float = 0.50
    max_pbo: float = MAX_PBO_THRESHOLD  # 0.50

    # Walk-forward
    n_folds: int = 5
    purge_days: int = ML_PURGE_WINDOW  # 5

    # Multiple testing
    correction_method: str = "benjamini_hochberg"

    # Performance limits
    max_candidates_per_run: int = 50_000
    min_observations: int = 252


# =============================================================================
# PATTERN SCANNER
# =============================================================================


class PatternScanner:
    """
    Automated pattern discovery engine.

    Scans feature combinations exhaustively and validates through 6 stages:

    1. **In-sample IC** -- Spearman rank correlation between signal and
       forward returns must exceed ``min_ic``.
    2. **IC stability** -- Rolling IC must be positive in >60 % of
       non-overlapping windows and the IC Information Ratio (mean / std)
       must exceed ``min_ic_ir``.
    3. **Out-of-sample validation** -- Walk-forward average IC across
       ``n_folds`` folds must exceed ``min_oos_ic``.
    4. **Multiple testing correction** -- Benjamini-Hochberg adjusted
       p-value must be below ``max_adjusted_pvalue``.
    5. **Orthogonality** -- Maximum absolute correlation with any existing
       accepted signal must be below ``max_correlation_to_existing``.
    6. **Alpha decay** -- Decay rate across three equal sub-periods must
       not exceed ``max_decay_rate``.

    Usage::

        scanner = PatternScanner(config)
        results = scanner.scan(feature_matrix, feature_names, returns)
        accepted = [r for r in results if r.passed]

    Args:
        config: Scanner configuration.  Uses defaults if not supplied.
    """

    def __init__(self, config: Optional[ScannerConfig] = None) -> None:
        self.config = config or ScannerConfig()
        self._pbo_calculator = PBOCalculator(n_groups=10, metric="sharpe")

        # Bookkeeping populated during scan()
        self._total_candidates: int = 0
        self._stage_counts: Dict[str, int] = {
            "generated": 0,
            "stage1_ic": 0,
            "stage2_stability": 0,
            "stage3_oos": 0,
            "stage4_multiple_testing": 0,
            "stage5_orthogonality": 0,
            "stage6_decay": 0,
            "accepted": 0,
        }
        self._rejection_reasons: Dict[str, int] = {}

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def scan(
        self,
        features: NDArray[np.float64],
        feature_names: List[str],
        forward_returns: NDArray[np.float64],
        existing_signals: Optional[NDArray[np.float64]] = None,
    ) -> List[ValidationResult]:
        """
        Run a full discovery scan.

        Generates all candidate signal specifications from the feature
        matrix, constructs each signal, and passes it sequentially through
        the six validation stages.  Only candidates surviving all six
        gates are marked as accepted.

        Args:
            features: Feature matrix of shape ``(T, K)`` where *T* is the
                number of time steps and *K* is the number of features.
            feature_names: List of *K* feature name strings corresponding
                to columns of *features*.
            forward_returns: Array of shape ``(T,)`` containing next-period
                forward returns aligned with the feature matrix rows.
            existing_signals: Optional matrix of shape ``(T, M)`` with *M*
                previously accepted signals for orthogonality screening.
                Pass ``None`` to skip Stage 5.

        Returns:
            List of :class:`ValidationResult` for every candidate evaluated.
            Check ``result.passed`` or ``result.candidate.passed_all_tests``
            to filter accepted signals.

        Raises:
            ValueError: If input dimensions are inconsistent or the time
                series is shorter than ``config.min_observations``.
        """
        T, K = features.shape
        if len(feature_names) != K:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match "
                f"feature matrix columns ({K})."
            )
        if forward_returns.shape[0] != T:
            raise ValueError(
                f"forward_returns length ({forward_returns.shape[0]}) does not "
                f"match feature matrix rows ({T})."
            )
        if T < self.config.min_observations:
            raise ValueError(
                f"Insufficient observations ({T}); need at least "
                f"{self.config.min_observations}."
            )

        # Reset bookkeeping
        self._reset_counts()

        # Step 1: generate candidate specifications
        candidates = self._generate_candidates(feature_names)
        self._total_candidates = len(candidates)
        self._stage_counts["generated"] = len(candidates)
        logger.info("Generated %d candidates from %d features.", len(candidates), K)

        # Step 2: construct signals and run stages 1-3, 5-6 per candidate
        results: List[ValidationResult] = []
        stage1_passed: List[Tuple[SignalCandidate, NDArray[np.float64]]] = []

        for candidate in candidates:
            signal = self._construct_signal(candidate, features, feature_names)

            # -- Stage 1: In-sample IC --
            passed_s1, ic, ic_std = self._validate_stage1_ic(
                signal, forward_returns
            )
            candidate.ic = ic
            candidate.ic_std = ic_std

            if not passed_s1:
                candidate.rejection_reason = (
                    f"Stage 1 IC: |IC|={abs(ic):.4f} < {self.config.min_ic}"
                )
                self._record_rejection("stage1_ic")
                results.append(self._make_result(candidate, signal, "stage1_ic", False))
                continue

            self._stage_counts["stage1_ic"] += 1

            # -- Stage 2: IC stability --
            passed_s2, ic_ir = self._validate_stage2_stability(
                signal, forward_returns
            )
            candidate.ic_ir = ic_ir

            if not passed_s2:
                candidate.rejection_reason = (
                    f"Stage 2 stability: IC_IR={ic_ir:.4f} < {self.config.min_ic_ir}"
                )
                self._record_rejection("stage2_stability")
                results.append(self._make_result(candidate, signal, "stage2_stability", False))
                continue

            self._stage_counts["stage2_stability"] += 1

            # -- Stage 3: Out-of-sample walk-forward --
            passed_s3, oos_ic = self._validate_stage3_oos(
                signal, forward_returns
            )
            candidate.oos_ic = oos_ic

            if not passed_s3:
                candidate.rejection_reason = (
                    f"Stage 3 OOS: OOS_IC={oos_ic:.4f} < {self.config.min_oos_ic}"
                )
                self._record_rejection("stage3_oos")
                results.append(self._make_result(candidate, signal, "stage3_oos", False))
                continue

            self._stage_counts["stage3_oos"] += 1

            # Passed first 3 stages -- hold for joint multiple-testing
            stage1_passed.append((candidate, signal))

        # Step 3: Stage 4 -- multiple testing correction across all
        # candidates that survived stages 1-3
        if stage1_passed:
            surviving_candidates = [c for c, _ in stage1_passed]
            self._validate_stage4_multiple_testing(
                surviving_candidates, self._total_candidates
            )

        # Step 4: continue stages 4-6 for surviving candidates
        for candidate, signal in stage1_passed:
            # -- Stage 4 check --
            if candidate.adjusted_pvalue > self.config.max_adjusted_pvalue:
                candidate.rejection_reason = (
                    f"Stage 4 multiple testing: adj_p={candidate.adjusted_pvalue:.6f} "
                    f"> {self.config.max_adjusted_pvalue}"
                )
                self._record_rejection("stage4_multiple_testing")
                results.append(
                    self._make_result(candidate, signal, "stage4_multiple_testing", False)
                )
                continue

            self._stage_counts["stage4_multiple_testing"] += 1

            # -- Stage 5: orthogonality --
            if existing_signals is not None and existing_signals.shape[1] > 0:
                passed_s5, max_corr = self._validate_stage5_orthogonality(
                    signal, existing_signals
                )
                candidate.max_correlation_to_existing = max_corr

                if not passed_s5:
                    candidate.rejection_reason = (
                        f"Stage 5 orthogonality: max_corr={max_corr:.4f} "
                        f"> {self.config.max_correlation_to_existing}"
                    )
                    self._record_rejection("stage5_orthogonality")
                    results.append(
                        self._make_result(candidate, signal, "stage5_orthogonality", False)
                    )
                    continue
            else:
                candidate.max_correlation_to_existing = 0.0

            self._stage_counts["stage5_orthogonality"] += 1

            # -- Stage 6: alpha decay --
            passed_s6, decay = self._validate_stage6_decay(
                signal, forward_returns
            )
            candidate.decay_rate = decay

            if not passed_s6:
                candidate.rejection_reason = (
                    f"Stage 6 decay: decay_rate={decay:.4f} "
                    f"> {self.config.max_decay_rate}"
                )
                self._record_rejection("stage6_decay")
                results.append(
                    self._make_result(candidate, signal, "stage6_decay", False)
                )
                continue

            self._stage_counts["stage6_decay"] += 1

            # -- All stages passed --
            candidate.passed_all_tests = True
            candidate.rejection_reason = None
            self._stage_counts["accepted"] += 1

            results.append(self._make_result(candidate, signal, "accepted", True))

        logger.info(
            "Scan complete: %d / %d candidates accepted.",
            self._stage_counts["accepted"],
            self._total_candidates,
        )
        return results

    # --------------------------------------------------------------------- #
    # Candidate Generation
    # --------------------------------------------------------------------- #

    def _generate_candidates(
        self, feature_names: List[str]
    ) -> List[SignalCandidate]:
        """
        Generate all candidate signal specifications.

        Creates the Cartesian product of:
        - Each single feature x each lookback window x each holding period.
        - Top-N features (by name order, since we have no IC yet) pairwise
          x multiply/ratio/difference x each lookback x each holding period.

        The total is capped at ``config.max_candidates_per_run``.

        Args:
            feature_names: List of feature name strings.

        Returns:
            List of :class:`SignalCandidate` with only specification fields
            populated (no validation metrics yet).
        """
        candidates: List[SignalCandidate] = []
        cfg = self.config

        # --- Single-feature candidates ---
        for feat in feature_names:
            for lb in cfg.lookback_windows:
                for hp in cfg.holding_periods:
                    candidates.append(
                        SignalCandidate(
                            name=f"{feat}_lb{lb}_hp{hp}",
                            feature_a=feat,
                            feature_b=None,
                            combination_type="single",
                            lookback_window=lb,
                            holding_period=hp,
                        )
                    )

        # --- Interaction candidates (top-N pairwise) ---
        interaction_feats = feature_names[: cfg.max_interaction_features]
        interaction_types = [
            ct for ct in cfg.combination_types if ct != "single"
        ]

        for feat_a, feat_b in combinations(interaction_feats, 2):
            for ctype in interaction_types:
                for lb in cfg.lookback_windows:
                    for hp in cfg.holding_periods:
                        candidates.append(
                            SignalCandidate(
                                name=f"{feat_a}_{ctype}_{feat_b}_lb{lb}_hp{hp}",
                                feature_a=feat_a,
                                feature_b=feat_b,
                                combination_type=ctype,
                                lookback_window=lb,
                                holding_period=hp,
                            )
                        )

        # Cap total
        if len(candidates) > cfg.max_candidates_per_run:
            logger.warning(
                "Capping candidates from %d to %d.",
                len(candidates),
                cfg.max_candidates_per_run,
            )
            # Deterministic sub-sampling: evenly spaced
            indices = np.linspace(
                0, len(candidates) - 1, cfg.max_candidates_per_run, dtype=int
            )
            candidates = [candidates[i] for i in indices]

        return candidates

    # --------------------------------------------------------------------- #
    # Signal Construction
    # --------------------------------------------------------------------- #

    def _construct_signal(
        self,
        candidate: SignalCandidate,
        features: NDArray[np.float64],
        feature_names: List[str],
    ) -> NDArray[np.float64]:
        """
        Build the signal time series from a candidate specification.

        For ``"single"`` type, extracts the raw feature column.  For
        interaction types, applies the operator element-wise to two feature
        columns.  All signals are then normalised to a rolling z-score
        using the candidate's lookback window.

        Args:
            candidate: The signal specification.
            features: Feature matrix ``(T, K)``.
            feature_names: Column name list of length *K*.

        Returns:
            Normalised signal array of shape ``(T,)``.  Leading values that
            fall inside the lookback warm-up are ``NaN``.
        """
        idx_a = feature_names.index(candidate.feature_a)
        col_a = features[:, idx_a].copy()

        if candidate.combination_type == "single":
            raw = col_a

        else:
            idx_b = feature_names.index(candidate.feature_b)  # type: ignore[arg-type]
            col_b = features[:, idx_b].copy()

            if candidate.combination_type == "multiply":
                raw = col_a * col_b
            elif candidate.combination_type == "ratio":
                raw = col_a / (col_b + EPSILON)
            elif candidate.combination_type == "difference":
                raw = col_a - col_b
            else:
                raise ValueError(
                    f"Unknown combination_type: {candidate.combination_type}"
                )

        # Rolling z-score normalisation (returns NaN for warm-up period)
        signal = zscore(raw.astype(np.float64), candidate.lookback_window)

        return signal

    # --------------------------------------------------------------------- #
    # Stage 1: In-Sample IC
    # --------------------------------------------------------------------- #

    def _validate_stage1_ic(
        self,
        signal: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> Tuple[bool, float, float]:
        """
        Stage 1: compute Spearman rank correlation between the signal
        and forward returns over the full in-sample period.

        NaN values in either series are masked out before computing the
        correlation.

        Args:
            signal: Signal array ``(T,)``.
            returns: Forward returns ``(T,)``.

        Returns:
            Tuple of ``(passed, ic, ic_std)`` where *ic* is the Spearman
            rho and *ic_std* is the standard error estimate
            ``1 / sqrt(N - 3)`` (Fisher, 1921).
        """
        mask = np.isfinite(signal) & np.isfinite(returns)
        n_valid = int(np.sum(mask))

        if n_valid < max(30, self.config.min_observations // 2):
            return False, 0.0, 1.0

        sig_clean = signal[mask]
        ret_clean = returns[mask]

        ic, pvalue = stats.spearmanr(sig_clean, ret_clean)
        ic = float(ic) if np.isfinite(ic) else 0.0

        # Standard error of Spearman rho: 1/sqrt(N-3)
        ic_std = 1.0 / np.sqrt(max(n_valid - 3, 1))

        passed = abs(ic) >= self.config.min_ic
        return passed, ic, ic_std

    # --------------------------------------------------------------------- #
    # Stage 2: IC Stability
    # --------------------------------------------------------------------- #

    def _validate_stage2_stability(
        self,
        signal: NDArray[np.float64],
        returns: NDArray[np.float64],
        window: int = 60,
    ) -> Tuple[bool, float]:
        """
        Stage 2: assess signal stability through rolling IC.

        Splits the data into non-overlapping windows of *window* days,
        computes Spearman IC in each window, then checks:
        - IC is positive in >60 % of windows.
        - IC Information Ratio (mean IC / std IC) exceeds ``min_ic_ir``.

        Args:
            signal: Signal array ``(T,)``.
            returns: Forward returns ``(T,)``.
            window: Non-overlapping window size for rolling IC.

        Returns:
            Tuple of ``(passed, ic_ir)`` where *ic_ir* is the IC
            Information Ratio.
        """
        T = len(signal)
        n_windows = T // window
        if n_windows < 3:
            # Not enough data for meaningful stability test
            return False, 0.0

        ic_values: List[float] = []

        for i in range(n_windows):
            start = i * window
            end = start + window

            seg_sig = signal[start:end]
            seg_ret = returns[start:end]

            mask = np.isfinite(seg_sig) & np.isfinite(seg_ret)
            n_valid = int(np.sum(mask))

            if n_valid < 10:
                continue

            rho, _ = stats.spearmanr(seg_sig[mask], seg_ret[mask])
            if np.isfinite(rho):
                ic_values.append(float(rho))

        if len(ic_values) < 3:
            return False, 0.0

        ic_arr = np.array(ic_values, dtype=np.float64)
        ic_mean = float(np.mean(ic_arr))
        ic_std_val = float(np.std(ic_arr, ddof=1))

        # IC Information Ratio
        ic_ir = ic_mean / (ic_std_val + EPSILON)

        # Positive IC fraction
        positive_frac = float(np.mean(ic_arr > 0))

        passed = (ic_ir >= self.config.min_ic_ir) and (positive_frac > 0.60)
        return passed, ic_ir

    # --------------------------------------------------------------------- #
    # Stage 3: Out-of-Sample Walk-Forward
    # --------------------------------------------------------------------- #

    def _validate_stage3_oos(
        self,
        signal: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> Tuple[bool, float]:
        """
        Stage 3: walk-forward out-of-sample validation.

        Uses an expanding-window approach with ``n_folds`` folds.  Each
        fold trains on all data up to a cutoff, purges ``purge_days``,
        then tests on the next block.  The average OOS IC across folds
        must be positive and above ``min_oos_ic``.

        Args:
            signal: Signal array ``(T,)``.
            returns: Forward returns ``(T,)``.

        Returns:
            Tuple of ``(passed, avg_oos_ic)``.
        """
        T = len(signal)
        n_folds = self.config.n_folds
        purge = self.config.purge_days

        # Minimum fold size for meaningful IC
        min_fold_size = max(30, self.config.min_observations // 5)

        fold_size = (T - purge * n_folds) // (n_folds + 1)
        if fold_size < min_fold_size:
            # Fallback: use fewer folds
            n_folds = max(2, (T - purge * 2) // min_fold_size - 1)
            fold_size = (T - purge * n_folds) // (n_folds + 1)
            if fold_size < min_fold_size:
                return False, 0.0

        oos_ics: List[float] = []

        for fold_idx in range(n_folds):
            # Expanding window: train on everything before the test block
            test_start = (fold_idx + 1) * fold_size + fold_idx * purge
            test_end = test_start + fold_size

            if test_end > T:
                break

            # Train end (before purge gap)
            train_end = test_start - purge
            if train_end < min_fold_size:
                continue

            # Test segment
            seg_sig = signal[test_start:test_end]
            seg_ret = returns[test_start:test_end]

            mask = np.isfinite(seg_sig) & np.isfinite(seg_ret)
            n_valid = int(np.sum(mask))

            if n_valid < 20:
                continue

            rho, _ = stats.spearmanr(seg_sig[mask], seg_ret[mask])
            if np.isfinite(rho):
                oos_ics.append(float(rho))

        if len(oos_ics) < 2:
            return False, 0.0

        avg_oos_ic = float(np.mean(oos_ics))

        passed = avg_oos_ic > 0 and avg_oos_ic >= self.config.min_oos_ic
        return passed, avg_oos_ic

    # --------------------------------------------------------------------- #
    # Stage 4: Multiple Testing Correction
    # --------------------------------------------------------------------- #

    def _validate_stage4_multiple_testing(
        self,
        candidates: List[SignalCandidate],
        n_total_tests: int,
    ) -> None:
        """
        Stage 4: adjust p-values for the multiplicity of tests.

        Computes a raw p-value for each candidate from its in-sample
        Spearman IC and then applies the configured correction method
        (Benjamini-Hochberg by default).  Adjusted p-values are written
        back onto each candidate in-place.

        Also computes and stores the t-statistic on each candidate.

        Args:
            candidates: Candidates that survived stages 1-3.
            n_total_tests: Total number of candidates generated (used for
                Bonferroni denominator; BH uses only survivors).
        """
        if not candidates:
            return

        n = len(candidates)
        raw_pvalues = np.empty(n, dtype=np.float64)

        for i, cand in enumerate(candidates):
            # t-stat from IC: t = IC * sqrt(N - 2) / sqrt(1 - IC^2)
            # Use a reasonable N (we don't store it; approximate from IC_std)
            # IC_std ~ 1/sqrt(N-3), so N ~ 1/IC_std^2 + 3
            if cand.ic_std > EPSILON:
                n_obs = int(round(1.0 / (cand.ic_std ** 2) + 3))
            else:
                n_obs = self.config.min_observations

            ic_sq = min(cand.ic ** 2, 1.0 - EPSILON)
            t = cand.ic * np.sqrt(max(n_obs - 2, 1)) / np.sqrt(1.0 - ic_sq + EPSILON)
            cand.t_stat = float(t)

            # Two-tailed p-value from t-distribution
            raw_pvalues[i] = float(
                2.0 * stats.t.sf(abs(t), df=max(n_obs - 2, 1))
            )

        # Apply correction
        adjusted = self._adjust_pvalues(raw_pvalues, n_total_tests)

        for i, cand in enumerate(candidates):
            cand.adjusted_pvalue = float(adjusted[i])

    def _adjust_pvalues(
        self,
        raw_pvalues: NDArray[np.float64],
        n_total_tests: int,
    ) -> NDArray[np.float64]:
        """
        Apply the configured multiple-testing correction.

        Supported methods:
        - ``"bonferroni"``: multiply by total test count.
        - ``"holm"``: step-down Holm-Bonferroni.
        - ``"benjamini_hochberg"``: step-up BH for FDR control.

        Args:
            raw_pvalues: Unadjusted p-values.
            n_total_tests: Total number of hypotheses tested.

        Returns:
            Adjusted p-values clipped to [0, 1].
        """
        n = len(raw_pvalues)
        method = self.config.correction_method

        if method == "bonferroni":
            adjusted = raw_pvalues * n_total_tests
            return np.minimum(adjusted, 1.0)

        elif method == "holm":
            sorted_idx = np.argsort(raw_pvalues)
            sorted_p = raw_pvalues[sorted_idx]
            adjusted_sorted = np.empty(n, dtype=np.float64)

            for i in range(n):
                adjusted_sorted[i] = sorted_p[i] * (n_total_tests - i)

            # Enforce monotonicity (step-down)
            for i in range(1, n):
                adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

            result = np.empty(n, dtype=np.float64)
            result[sorted_idx] = np.minimum(adjusted_sorted, 1.0)
            return result

        elif method == "benjamini_hochberg":
            sorted_idx = np.argsort(raw_pvalues)[::-1]  # Descending
            sorted_p = raw_pvalues[sorted_idx]
            adjusted_sorted = np.empty(n, dtype=np.float64)

            # Start from the largest p-value
            adjusted_sorted[0] = sorted_p[0]

            for i in range(1, n):
                rank = n - i  # Rank in ascending order (1-based)
                bh_adj = sorted_p[i] * n / rank
                adjusted_sorted[i] = min(adjusted_sorted[i - 1], bh_adj)

            result = np.empty(n, dtype=np.float64)
            result[sorted_idx] = np.minimum(adjusted_sorted, 1.0)
            return result

        else:
            logger.warning(
                "Unknown correction method '%s'; returning raw p-values.",
                method,
            )
            return raw_pvalues

    # --------------------------------------------------------------------- #
    # Stage 5: Orthogonality
    # --------------------------------------------------------------------- #

    def _validate_stage5_orthogonality(
        self,
        signal: NDArray[np.float64],
        existing_signals: NDArray[np.float64],
    ) -> Tuple[bool, float]:
        """
        Stage 5: check that the new signal is sufficiently orthogonal to
        every existing accepted signal.

        Computes the absolute Pearson correlation between the candidate
        and each column of *existing_signals*, masking NaN values
        pair-wise.  The maximum is compared against the threshold.

        Args:
            signal: Candidate signal ``(T,)``.
            existing_signals: Existing accepted signals ``(T, M)``.

        Returns:
            Tuple of ``(passed, max_abs_correlation)``.
        """
        max_corr = 0.0
        n_existing = existing_signals.shape[1]

        for j in range(n_existing):
            existing_col = existing_signals[:, j]
            mask = np.isfinite(signal) & np.isfinite(existing_col)
            n_valid = int(np.sum(mask))

            if n_valid < 30:
                continue

            corr = np.corrcoef(signal[mask], existing_col[mask])[0, 1]
            if np.isfinite(corr):
                max_corr = max(max_corr, abs(float(corr)))

        passed = max_corr < self.config.max_correlation_to_existing
        return passed, max_corr

    # --------------------------------------------------------------------- #
    # Stage 6: Alpha Decay
    # --------------------------------------------------------------------- #

    def _validate_stage6_decay(
        self,
        signal: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> Tuple[bool, float]:
        """
        Stage 6: estimate alpha decay by comparing IC across three equal
        sub-periods of the sample.

        Decay rate is defined as ``1 - IC_last / IC_first`` when
        ``IC_first > 0``.  A high decay rate suggests the signal's
        predictive power is eroding and will likely not persist
        out-of-sample.

        Args:
            signal: Signal array ``(T,)``.
            returns: Forward returns ``(T,)``.

        Returns:
            Tuple of ``(passed, decay_rate)``.
        """
        T = len(signal)
        third = T // 3

        if third < 30:
            # Not enough data to assess decay
            return False, 1.0

        period_ics: List[float] = []
        for p in range(3):
            start = p * third
            end = start + third if p < 2 else T

            seg_sig = signal[start:end]
            seg_ret = returns[start:end]

            mask = np.isfinite(seg_sig) & np.isfinite(seg_ret)
            n_valid = int(np.sum(mask))

            if n_valid < 20:
                period_ics.append(0.0)
                continue

            rho, _ = stats.spearmanr(seg_sig[mask], seg_ret[mask])
            period_ics.append(float(rho) if np.isfinite(rho) else 0.0)

        ic_first = period_ics[0]
        ic_last = period_ics[2]

        if abs(ic_first) < EPSILON:
            # Cannot measure decay if first period IC is effectively zero
            decay_rate = 0.0
        else:
            decay_rate = 1.0 - (ic_last / ic_first)

        # Clamp to [0, 2] to avoid pathological negatives when sign flips
        decay_rate = float(np.clip(decay_rate, 0.0, 2.0))

        passed = decay_rate <= self.config.max_decay_rate
        return passed, decay_rate

    # --------------------------------------------------------------------- #
    # Summary & Helpers
    # --------------------------------------------------------------------- #

    def get_scan_summary(self) -> Dict:
        """
        Return summary statistics from the most recent scan.

        Includes total candidates generated, the count surviving each
        validation stage, the final acceptance count, and a breakdown
        of rejection reasons.

        Returns:
            Dictionary with keys ``"total_candidates"``,
            ``"stage_counts"``, ``"acceptance_rate"``, and
            ``"rejection_reasons"``.
        """
        total = self._total_candidates
        accepted = self._stage_counts.get("accepted", 0)
        acceptance_rate = accepted / total if total > 0 else 0.0

        return {
            "total_candidates": total,
            "stage_counts": dict(self._stage_counts),
            "acceptance_rate": acceptance_rate,
            "rejection_reasons": dict(self._rejection_reasons),
        }

    # --------------------------------------------------------------------- #
    # Internal Helpers
    # --------------------------------------------------------------------- #

    def _make_result(
        self,
        candidate: SignalCandidate,
        signal: NDArray[np.float64],
        final_stage: str,
        passed: bool,
    ) -> ValidationResult:
        """
        Construct a :class:`ValidationResult` from accumulated metrics.

        Args:
            candidate: The evaluated candidate.
            signal: The constructed signal array.
            final_stage: The last stage evaluated (pass or fail).
            passed: Whether the candidate passed all stages.

        Returns:
            Fully populated :class:`ValidationResult`.
        """
        # Build stage_results up to the point of failure (or success)
        all_stages = [
            "stage1_ic",
            "stage2_stability",
            "stage3_oos",
            "stage4_multiple_testing",
            "stage5_orthogonality",
            "stage6_decay",
        ]

        stage_results: Dict[str, bool] = {}
        for stage in all_stages:
            if stage == final_stage and not passed:
                stage_results[stage] = False
                break
            stage_results[stage] = True
            if stage == final_stage:
                break

        metrics: Dict[str, float] = {
            "ic": candidate.ic,
            "ic_std": candidate.ic_std,
            "ic_ir": candidate.ic_ir,
            "oos_ic": candidate.oos_ic,
            "t_stat": candidate.t_stat,
            "adjusted_pvalue": candidate.adjusted_pvalue,
            "max_correlation_to_existing": candidate.max_correlation_to_existing,
            "decay_rate": candidate.decay_rate,
            "pbo": candidate.pbo,
        }

        return ValidationResult(
            candidate=candidate,
            passed=passed,
            stage_results=stage_results,
            metrics=metrics,
            signal_values=signal if passed else None,
        )

    def _reset_counts(self) -> None:
        """Reset all internal counters for a fresh scan run."""
        self._total_candidates = 0
        self._stage_counts = {
            "generated": 0,
            "stage1_ic": 0,
            "stage2_stability": 0,
            "stage3_oos": 0,
            "stage4_multiple_testing": 0,
            "stage5_orthogonality": 0,
            "stage6_decay": 0,
            "accepted": 0,
        }
        self._rejection_reasons = {}

    def _record_rejection(self, stage: str) -> None:
        """Increment the rejection counter for a given stage."""
        self._rejection_reasons[stage] = self._rejection_reasons.get(stage, 0) + 1
