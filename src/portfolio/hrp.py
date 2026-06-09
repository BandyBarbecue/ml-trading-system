"""
Hierarchical Risk Parity (Lopez de Prado, 2016)
================================================
Instead of optimising expected returns (which are noisy and unstable),
HRP uses the *structure* of the correlation matrix to allocate risk evenly
across clusters of similar assets.

Analogy: imagine sorting books by topic, then giving each topic shelf an equal
amount of space, and within each shelf dividing space among books by how much
they vary. No topic dominates the shelf just because it has many books.

Algorithm steps:
  1. Correlation matrix  → distance matrix
  2. Hierarchical clustering (Ward linkage) groups similar assets
  3. Quasi-diagonalisation reorders assets so correlated ones are adjacent
  4. Recursive bisection allocates weights inversely proportional to variance
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix.

    Divides each element by the product of the two corresponding standard
    deviations (square roots of the diagonal).  The result is clipped to
    [-1, 1] to guard against tiny floating-point violations.
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1, 1)


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Return the leaf order from a linkage matrix.

    scipy's `leaves_list` traverses the dendrogram and returns asset indices
    sorted so that closely related assets end up next to each other — the
    'quasi-diagonal' arrangement that makes the recursive bisection work well.
    """
    return list(leaves_list(link))


def _recursive_bisection(cov: np.ndarray, items: list[int]) -> dict[int, float]:
    """Allocate weights via recursive bisection on the sorted covariance matrix.

    Idea: at each level, split the current cluster into two halves.
    Each half gets a share of weight *inversely proportional* to its variance
    (riskier half gets less weight).  Recurse within each half until
    individual assets remain.

    Parameters
    ----------
    cov   : square numpy array — covariance matrix already reordered by clustering
    items : list of integer column/row indices to allocate over

    Returns
    -------
    dict mapping each index to its final weight (values sum to ~1.0)
    """
    def cluster_var(idx_list: list[int]) -> float:
        """Variance of an equally-weighted sub-portfolio."""
        # np.ix_ picks the sub-matrix for the given row/column indices
        sub = cov[np.ix_(idx_list, idx_list)]
        w = np.ones(len(idx_list)) / len(idx_list)
        return float(w @ sub @ w)

    weights = {i: 1.0 for i in items}
    clusters = [items]  # start with one big cluster containing everything

    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue  # single asset — nothing to split

            # Split into two halves
            mid   = len(cluster) // 2
            left  = cluster[:mid]
            right = cluster[mid:]

            var_l = cluster_var(left)
            var_r = cluster_var(right)

            # Left's allocation = fraction of total risk that belongs to the right
            # (inverse-variance weighting: riskier side gets smaller share)
            alloc_l = 1.0 - var_l / (var_l + var_r + 1e-12)

            for i in left:
                weights[i] *= alloc_l
            for i in right:
                weights[i] *= (1.0 - alloc_l)

            # Queue sub-clusters for further splitting (only if they have >1 item)
            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)

        clusters = next_clusters

    return weights


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Compute Hierarchical Risk Parity weights (Lopez de Prado, 2016).

    Parameters
    ----------
    returns : pd.DataFrame
        Each column is a ticker; each row is a period's return.
        At least 2 rows required for a meaningful covariance estimate.

    Returns
    -------
    pd.Series
        Weights indexed by ticker, in the same column order as `returns`.
        Weights are positive and sum to exactly 1.0.
    """
    tickers = returns.columns.tolist()
    n = len(tickers)

    # Edge case: single asset gets the whole portfolio
    if n == 1:
        return pd.Series([1.0], index=tickers, name="weight")

    # --- Step 1: covariance & correlation ---
    cov  = returns.cov().values          # (n × n) covariance matrix
    corr = _cov_to_corr(cov)            # (n × n) correlation matrix

    # --- Step 2: distance matrix ---
    # d(i,j) = sqrt(0.5 * (1 - rho_ij))
    # When rho = 1  → d = 0  (identical assets)
    # When rho = 0  → d = 0.707  (uncorrelated)
    # When rho = -1 → d = 1  (perfectly opposite)
    # This formula satisfies the triangle inequality, which linkage requires.
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))

    # scipy's linkage needs a *condensed* distance vector (upper triangle only)
    condensed = squareform(dist, checks=False)

    # --- Step 3: hierarchical clustering ---
    # Ward linkage minimises total within-cluster variance at each merge step —
    # a natural fit for a variance-based portfolio method.
    link = linkage(condensed, method="ward")

    # --- Step 4: quasi-diagonalisation ---
    # `order` is the list of original asset indices, reordered so that
    # similar assets sit adjacent to each other in the matrix.
    order = _quasi_diag(link)

    # Reorder the covariance matrix to match the clustering order
    cov_sorted = cov[np.ix_(order, order)]

    # --- Step 5: recursive bisection ---
    raw_weights = _recursive_bisection(cov_sorted, list(range(n)))

    # Map positional indices (in sorted order) back to ticker names
    w = pd.Series(
        {tickers[order[i]]: raw_weights[i] for i in range(n)},
        name="weight",
    )

    # Normalise (should already sum to ~1, but floating-point can drift slightly)
    w = w / w.sum()

    # Return in the original ticker order (same as input columns)
    return w[tickers]
