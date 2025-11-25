import math


def piecewise_linear_nojump(x, breakpoints, slopes, intercepts):
    """
    A generic piecewise-linear, nondecreasing function phi(x),
    typically used to upper-bound log(1 + x) or a similar concave function.

    breakpoints: [b_0, b_1, ..., b_L],  b_0 = 0, b_L = some max
    slopes[i], intercepts[i] define the line on [b_i, b_{i+1}].
    """
    L = len(slopes)
    for i in range(L):
        if x <= breakpoints[i + 1]:
            return slopes[i] * x + intercepts[i]
    # If x > breakpoints[-1], clamp or continue with last segment
    return slopes[-1] * x + intercepts[-1]


def scenario_aggregator(S, scenario, k):
    """
    Compute g_k^{(scenario)}(S) - optimal reward for agent k from PROBED arms in S.

    Parameters
    ----------
    S : set of int
        Probing set (arms being probed)
    scenario : dict
        Scenario with random reward realizations, scenario["X"][m][k]
    k : int
        Agent index

    Returns
    -------
    float
        Best reward agent k can get from arms in S
    """
    if len(S) == 0:
        return 0.0

    # Greedy: agent k selects the best arm from S
    best_reward = max(scenario["X"][m][k] for m in S)
    return best_reward


def scenario_value_hprob(S, scenario, K, phi_functions):
    """
    For a single scenario, compute g(S) - NSW over probed arms.

    Returns NSW itself.

    Parameters
    ----------
    S : set of int
        Probing set
    scenario : dict
        Single scenario
    K : int
        Number of agents
    phi_functions : list of callables
        Not used anymore (kept for compatibility)

    Returns
    -------
    float
        NSW (geometric mean) for this scenario
    """
    # Compute NSW = geometric mean of rewards
    prod_val = 1.0
    eps = 1e-9
    for k in range(K):
        g_val = scenario_aggregator(S, scenario, k)  # Only probed arms, normalized
        prod_val *= max(g_val, eps)

    # Geometric mean
    nsw = prod_val ** (1.0 / K)
    return nsw


def f_prob_with_samples_piecewise_upper(S, scenarios, K, phi_functions):
    """
    Compute g(S) - average NSW over PROBED arms only.

    Parameters
    ----------
    S : set of int
        Probing set (arms to probe)
    scenarios : list of dict
        Random scenarios, each with reward realizations scenario["X"][m][k]
    K : int
        Number of agents
    phi_functions : list of callables
        Not used anymore (kept for compatibility)

    Returns
    -------
    float
        g(S) - average NSW over probed arms
    """
    total_val = 0.0
    for scenario in scenarios:
        total_val += scenario_value_hprob(S, scenario, K, phi_functions)
    return total_val / len(scenarios)
