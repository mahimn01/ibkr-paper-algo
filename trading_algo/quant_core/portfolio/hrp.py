"""
Hierarchical Risk Parity (HRP)

Implements portfolio construction using hierarchical clustering,
as introduced by López de Prado (2016).

Key Advantages over Mean-Variance:
    - No covariance matrix inversion (numerically stable)
    - Robust to estimation errors
    - Captures hierarchical structure of correlations
    - Better out-of-sample performance

Algorithm:
    1. Tree Clustering: Cluster assets by correlation distance
    2. Quasi-Diagonalization: Reorder assets to place similar ones together
    3. Recursive Bisection: Allocate weights using inverse variance

References:
    - López de Prado, M. (2016). "Building Diversified Portfolios that
      Outperform Out-of-Sample". Journal of Portfolio Management, 42(4), 59-69.
    - López de Prado, M. (2018). "Advances in Financial Machine Learning"
    - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from trading_algo.quant_core.utils.constants import EPSILON


@dataclass
class HRPResult:
    """Result of HRP portfolio construction."""
    weights: NDArray[np.float64]           # Asset weights
    symbols: List[str]                     # Asset symbols
    cluster_order: NDArray[np.int64]       # Quasi-diagonalized order
    dendrogram_linkage: NDArray            # Linkage matrix for dendrogram
    risk_contribution: NDArray[np.float64] # Risk contribution per asset
    portfolio_variance: float              # Total portfolio variance

    def to_dict(self) -> Dict[str, float]:
        """Convert to symbol -> weight dictionary."""
        return {s: float(w) for s, w in zip(self.symbols, self.weights)}


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity portfolio optimizer.

    Constructs portfolios using hierarchical clustering on the
    correlation structure, avoiding the pitfalls of Markowitz
    mean-variance optimization.

    Usage:
        hrp = HierarchicalRiskParity()

        # From returns matrix
        result = hrp.optimize(returns_matrix, symbols)

        # From covariance matrix
        result = hrp.optimize_from_covariance(cov_matrix, symbols)
    """

    def __init__(
        self,
        linkage_method: str = "single",
        risk_measure: str = "variance",
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        """
        Initialize HRP optimizer.

        Args:
            linkage_method: Hierarchical clustering method
                ('single', 'complete', 'average', 'ward')
            risk_measure: Risk measure for allocation
                ('variance', 'std', 'mad', 'cvar')
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
        """
        self.linkage_method = linkage_method
        self.risk_measure = risk_measure
        self.min_weight = min_weight
        self.max_weight = max_weight

    def optimize(
        self,
        returns: NDArray[np.float64],
        symbols: Optional[List[str]] = None,
    ) -> HRPResult:
        """
        Optimize portfolio weights using HRP.

        Args:
            returns: Returns matrix (T x N) where N is number of assets
            symbols: Asset symbols (optional)

        Returns:
            HRPResult with optimal weights
        """
        n_assets = returns.shape[1]
        if symbols is None:
            symbols = [f"Asset_{i}" for i in range(n_assets)]

        # Calculate covariance and correlation matrices
        cov = np.cov(returns, rowvar=False)
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)

        # Ensure proper matrix properties
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)

        return self._optimize_from_matrices(cov, corr, symbols)

    def optimize_from_covariance(
        self,
        cov_matrix: NDArray[np.float64],
        symbols: Optional[List[str]] = None,
    ) -> HRPResult:
        """
        Optimize from precomputed covariance matrix.

        Args:
            cov_matrix: Covariance matrix (N x N)
            symbols: Asset symbols

        Returns:
            HRPResult with optimal weights
        """
        n_assets = cov_matrix.shape[0]
        if symbols is None:
            symbols = [f"Asset_{i}" for i in range(n_assets)]

        # Convert covariance to correlation
        std = np.sqrt(np.diag(cov_matrix))
        std[std < EPSILON] = EPSILON
        corr = cov_matrix / np.outer(std, std)

        # Ensure proper matrix properties
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)

        return self._optimize_from_matrices(cov_matrix, corr, symbols)

    def _optimize_from_matrices(
        self,
        cov: NDArray[np.float64],
        corr: NDArray[np.float64],
        symbols: List[str],
    ) -> HRPResult:
        """
        Core HRP optimization from covariance and correlation matrices.

        Steps:
            1. Compute distance matrix from correlations
            2. Hierarchical clustering
            3. Quasi-diagonalization
            4. Recursive bisection for weights
        """
        n_assets = cov.shape[0]

        # Step 1: Distance matrix
        # d_ij = sqrt(0.5 * (1 - ρ_ij))
        dist = self._correlation_distance(corr)

        # Step 2: Hierarchical clustering
        # Convert to condensed form for linkage
        condensed_dist = squareform(dist, checks=False)
        link = linkage(condensed_dist, method=self.linkage_method)

        # Step 3: Quasi-diagonalization (get optimal leaf order)
        sorted_indices = leaves_list(link).astype(np.int64)

        # Step 4: Recursive bisection
        weights = self._recursive_bisection(cov, sorted_indices)

        # Apply weight constraints
        weights = self._apply_constraints(weights)

        # Calculate risk contributions
        port_var = float(weights @ cov @ weights)
        marginal_contrib = cov @ weights
        risk_contrib = weights * marginal_contrib / (np.sqrt(port_var) + EPSILON)

        return HRPResult(
            weights=weights,
            symbols=symbols,
            cluster_order=sorted_indices,
            dendrogram_linkage=link,
            risk_contribution=risk_contrib,
            portfolio_variance=port_var,
        )

    def _correlation_distance(self, corr: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate distance matrix from correlation.

        Distance: d = sqrt(0.5 * (1 - ρ))
        This satisfies the triangle inequality.
        """
        # Ensure correlations are in valid range
        corr = np.clip(corr, -1, 1)
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)
        return dist

    def _recursive_bisection(
        self,
        cov: NDArray[np.float64],
        sorted_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """
        Allocate weights using recursive bisection.

        At each node, split weight inversely proportional to
        cluster variance.
        """
        n_assets = len(sorted_indices)
        weights = np.zeros(n_assets)

        # Initialize all weights equally
        weights[sorted_indices] = 1.0

        # Clusters to process: (start_idx, end_idx, weight)
        clusters = [(0, n_assets, 1.0)]

        while clusters:
            start, end, cluster_weight = clusters.pop()

            if end - start == 1:
                # Single asset cluster
                weights[sorted_indices[start]] = cluster_weight
                continue

            # Split into two sub-clusters
            mid = (start + end) // 2

            # Calculate variance of each sub-cluster
            left_indices = sorted_indices[start:mid]
            right_indices = sorted_indices[mid:end]

            left_var = self._cluster_variance(cov, left_indices)
            right_var = self._cluster_variance(cov, right_indices)

            # Allocate inversely proportional to variance
            total_inv_var = 1.0 / (left_var + EPSILON) + 1.0 / (right_var + EPSILON)
            left_weight = (1.0 / (left_var + EPSILON)) / total_inv_var
            right_weight = 1.0 - left_weight

            # Add sub-clusters to process
            clusters.append((start, mid, cluster_weight * left_weight))
            clusters.append((mid, end, cluster_weight * right_weight))

        return weights

    def _cluster_variance(
        self,
        cov: NDArray[np.float64],
        indices: NDArray[np.int64],
    ) -> float:
        """
        Calculate variance of a cluster using inverse-variance weighting.
        """
        cluster_cov = cov[np.ix_(indices, indices)]

        if self.risk_measure == "variance":
            # Inverse-variance portfolio within cluster
            inv_var = 1.0 / (np.diag(cluster_cov) + EPSILON)
            weights = inv_var / inv_var.sum()
            return float(weights @ cluster_cov @ weights)

        elif self.risk_measure == "std":
            # Use standard deviation
            return float(np.sqrt(np.mean(np.diag(cluster_cov))))

        else:
            # Default to variance
            inv_var = 1.0 / (np.diag(cluster_cov) + EPSILON)
            weights = inv_var / inv_var.sum()
            return float(weights @ cluster_cov @ weights)

    def _apply_constraints(
        self,
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Apply min/max weight constraints.
        """
        # Clip to bounds
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # Renormalize to sum to 1
        total = weights.sum()
        if total > EPSILON:
            weights = weights / total

        return weights


class NestedClusteredOptimization:
    """
    Nested Clustered Optimization (NCO) - Extension of HRP.

    Combines HRP with optimization within clusters.

    From López de Prado (2018), Chapter 16.
    """

    def __init__(
        self,
        hrp: Optional[HierarchicalRiskParity] = None,
        max_clusters: int = 10,
        intra_cluster_method: str = "min_variance",
    ):
        """
        Initialize NCO.

        Args:
            hrp: HRP optimizer (creates default if None)
            max_clusters: Maximum number of clusters
            intra_cluster_method: Optimization within clusters
                ('min_variance', 'equal_weight', 'risk_parity')
        """
        self.hrp = hrp or HierarchicalRiskParity()
        self.max_clusters = max_clusters
        self.intra_cluster_method = intra_cluster_method

    def optimize(
        self,
        returns: NDArray[np.float64],
        symbols: Optional[List[str]] = None,
    ) -> HRPResult:
        """
        Optimize using NCO.

        1. Cluster assets
        2. Optimize within each cluster
        3. Optimize across clusters
        """
        n_assets = returns.shape[1]
        if symbols is None:
            symbols = [f"Asset_{i}" for i in range(n_assets)]

        # Get HRP clustering
        hrp_result = self.hrp.optimize(returns, symbols)

        # Identify clusters from dendrogram
        from scipy.cluster.hierarchy import fcluster
        n_clusters = min(self.max_clusters, n_assets)
        cluster_labels = fcluster(
            hrp_result.dendrogram_linkage,
            n_clusters,
            criterion='maxclust',
        )

        # Calculate covariance
        cov = np.cov(returns, rowvar=False)

        # Optimize within each cluster
        cluster_weights = {}
        cluster_vars = {}

        for c in range(1, n_clusters + 1):
            mask = cluster_labels == c
            if not np.any(mask):
                continue

            cluster_indices = np.where(mask)[0]
            cluster_cov = cov[np.ix_(cluster_indices, cluster_indices)]

            if self.intra_cluster_method == "min_variance":
                w = self._min_variance_weights(cluster_cov)
            elif self.intra_cluster_method == "equal_weight":
                w = np.ones(len(cluster_indices)) / len(cluster_indices)
            else:  # risk_parity
                w = self._risk_parity_weights(cluster_cov)

            cluster_weights[c] = (cluster_indices, w)
            cluster_vars[c] = float(w @ cluster_cov @ w)

        # Allocate across clusters (inverse variance)
        total_inv_var = sum(1.0 / (v + EPSILON) for v in cluster_vars.values())
        cluster_allocations = {
            c: (1.0 / (v + EPSILON)) / total_inv_var
            for c, v in cluster_vars.items()
        }

        # Combine into final weights
        final_weights = np.zeros(n_assets)
        for c, (indices, w) in cluster_weights.items():
            final_weights[indices] = w * cluster_allocations[c]

        # Normalize
        final_weights = final_weights / final_weights.sum()

        # Calculate risk metrics
        port_var = float(final_weights @ cov @ final_weights)
        marginal_contrib = cov @ final_weights
        risk_contrib = final_weights * marginal_contrib / (np.sqrt(port_var) + EPSILON)

        return HRPResult(
            weights=final_weights,
            symbols=symbols,
            cluster_order=hrp_result.cluster_order,
            dendrogram_linkage=hrp_result.dendrogram_linkage,
            risk_contribution=risk_contrib,
            portfolio_variance=port_var,
        )

    def _min_variance_weights(self, cov: NDArray[np.float64]) -> NDArray[np.float64]:
        """Minimum variance portfolio weights."""
        n = cov.shape[0]
        try:
            inv_cov = np.linalg.inv(cov + np.eye(n) * EPSILON)
            ones = np.ones(n)
            weights = inv_cov @ ones / (ones @ inv_cov @ ones)
            weights = np.maximum(weights, 0)  # Long only
            return weights / weights.sum()
        except np.linalg.LinAlgError:
            return np.ones(n) / n

    def _risk_parity_weights(
        self,
        cov: NDArray[np.float64],
        tol: float = 1e-8,
        max_iter: int = 1000,
    ) -> NDArray[np.float64]:
        """Risk parity weights via iterative optimization."""
        n = cov.shape[0]
        weights = np.ones(n) / n

        for _ in range(max_iter):
            marginal_risk = cov @ weights
            risk_contrib = weights * marginal_risk
            total_risk = np.sqrt(weights @ cov @ weights)

            # Target equal risk contribution
            target_risk = total_risk / n

            # Update weights
            new_weights = target_risk / (marginal_risk + EPSILON)
            new_weights = new_weights / new_weights.sum()

            if np.max(np.abs(new_weights - weights)) < tol:
                break

            weights = new_weights

        return weights


def correlation_to_distance(corr: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert correlation matrix to distance matrix.

    Uses the angular distance: d = sqrt(0.5 * (1 - ρ))
    """
    corr = np.clip(corr, -1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    return dist


def dendrogram_plot_data(
    linkage_matrix: NDArray,
    symbols: List[str],
) -> Dict:
    """
    Prepare data for dendrogram visualization.

    Returns dictionary suitable for plotting libraries.
    """
    from scipy.cluster.hierarchy import dendrogram

    dendro = dendrogram(linkage_matrix, labels=symbols, no_plot=True)

    return {
        "ivl": dendro["ivl"],  # Labels
        "icoord": dendro["icoord"],  # x coordinates
        "dcoord": dendro["dcoord"],  # y coordinates (distances)
        "leaves": dendro["leaves"],  # Leaf order
    }
