import math
from itertools import combinations

def compute_realNSW_in_scenario(S, scenario, K, aggregator_func, eps=1e-9):
    """
    Compute NSW for a single scenario
    """
    prod_val = 1.0
    for k in range(K):
        val_k = aggregator_func(S, scenario, k)
        if val_k < eps:
            return 0.0   # Return 0 if utility is too small
        prod_val *= val_k
    return prod_val ** (1.0 / K)

def avg_realNSW(S, scenarios, K, aggregator_func, eps=1e-9):
    """
    Calculate expected "real geometric mean NSW" across all scenarios:
    avg_{sc}( ( \prod_k aggregator(S,sc,k) )^(1/K) )
    """
    total = 0.0
    for sc in scenarios:
        val_sc = compute_realNSW_in_scenario(S, sc, K, aggregator_func, eps)
        total += val_sc
    return total / len(scenarios)

def brute_force_optimal_subset(M, scenarios, K, aggregator_func, size_limit=None):
    """
    Find optimal subset S in [0..M-1] with size <= size_limit that maximizes avg_realNSW.
    Returns (S_opt, best_val) where best_val = max_S avg_realNSW(S).
    """
    if size_limit is None:
        size_limit = M

    best_val = float('-inf')
    best_S = set()
    for r in range(size_limit+1):
        for combo in combinations(range(M), r):
            S_cand = set(combo)
            val_cand = avg_realNSW(S_cand, scenarios, K, aggregator_func)
            if val_cand > best_val:
                best_val = val_cand
                best_S = S_cand
    return best_S, best_val

def compare_sets_logNSW(S1, S2, scenarios, K, aggregator_func):
    """
    Compute and compare average logNSW for subsets S1 and S2.
    Return (val1, val2, difference).
    """
    val1 = avg_realNSW(S1, scenarios, K, aggregator_func)
    val2 = avg_realNSW(S2, scenarios, K, aggregator_func)
    diff = val1 - val2
    return val1, val2, diff
