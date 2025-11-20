# f_unprob.py

import math

def f_unprobed(S, scenarios, K, aggregator_func, eps=1e-9):
    """
    A function that directly computes average log(NSW) for subset S
    across all scenarios, ignoring any 'probing' submodular approximation.
    Typically used in offline_greedy_probing(...) final comparison:
      If (1-alpha(|S|))*f_prob(S) < f_unprobed(∅) then choose ∅, etc.

    This is NOT submodular, but used as a baseline or fallback check.
    """
    INF_PENALTY = -1e9
    total_val = 0.0
    for sc in scenarios:
        sum_log = 0.0
        for k in range(K):
            val_k = aggregator_func(S, sc, k)
            if val_k < eps:
                sum_log = float('-inf')
                break
            sum_log += math.log(val_k)
        if sum_log == float('-inf'):
            total_val += INF_PENALTY
        else:
            total_val += sum_log

    return total_val / len(scenarios)
