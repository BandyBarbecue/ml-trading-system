import pandas as pd
import numpy as np
import pytest
from src.portfolio.hrp import hrp_weights

def make_block_returns(n=100, seed=7):
    """Two groups of 3 tickers — highly correlated within group, uncorrelated between."""
    np.random.seed(seed)
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    returns = pd.DataFrame({
        "A": 0.8 * f1 + 0.2 * np.random.randn(n),
        "B": 0.8 * f1 + 0.2 * np.random.randn(n),
        "C": 0.8 * f1 + 0.2 * np.random.randn(n),
        "D": 0.8 * f2 + 0.2 * np.random.randn(n),
        "E": 0.8 * f2 + 0.2 * np.random.randn(n),
        "F": 0.8 * f2 + 0.2 * np.random.randn(n),
    })
    return returns

def test_weights_sum_to_one():
    returns = make_block_returns()
    w = hrp_weights(returns)
    assert abs(w.sum() - 1.0) < 1e-9

def test_all_weights_positive():
    returns = make_block_returns()
    w = hrp_weights(returns)
    assert (w > 0).all()

def test_block_structure_balanced():
    """Two uncorrelated clusters should each get ~50% total weight."""
    returns = make_block_returns()
    w = hrp_weights(returns)
    cluster1 = w[["A", "B", "C"]].sum()
    cluster2 = w[["D", "E", "F"]].sum()
    assert abs(cluster1 - 0.5) < 0.15
    assert abs(cluster2 - 0.5) < 0.15

def test_single_ticker():
    returns = pd.DataFrame({"AAPL": np.random.randn(50)})
    w = hrp_weights(returns)
    assert abs(w["AAPL"] - 1.0) < 1e-9
