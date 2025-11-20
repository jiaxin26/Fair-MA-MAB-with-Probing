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
    Compute g_k^{(scenario)}(S).
    'scenario' includes {X_{m,k}, N_m} for all arms m, players k.

    For demonstration, let's assume:
      g_k^{(scenario)}(S) = sum_{m in S} min(X_{m,k}^{(scenario)}, someLimit)
    or you might have a resource-based partial allocation logic.

    This function must be submodular in S. For a simple example, a linear sum:
      g_k^{(scenario)}(S) = sum_{m in S} scenario["X"][m][k]

    Adjust to your real setting.
    """
    # Example: X_{m,k} is stored as scenario["X"][m][k]
    total = 0.0
    for m in S:
        total += scenario["X"][m][k]
    return total


def scenario_value_hprob(S, scenario, K, phi_functions):
    """
    For a single scenario, define h^{(scenario)}(S) = sum_k phi_k(g_k^{(scenario)}(S)).
    This yields the scenario's submodular reward w.r.t S if each phi_k o g_k^{(scenario)}
    is submodular (which typically holds if g_k^{(scenario)} is monotone submodular
    and phi_k is nondecreasing piecewise linear).
    """
    val = 0.0
    for k in range(K):
        g_val = scenario_aggregator(S, scenario, k)
        val += phi_functions[k](g_val)
    return val


def f_prob_with_samples_piecewise_upper(S, scenarios, K, phi_functions):
    """
    This is our "offline f_prob" that sums/averages over all random realizations.

    f_prob(S) = (1/|scenarios|) * sum_{scenario in scenarios} h_prob(S, scenario)
              = E_scenario[ sum_k phi_k( g_k^{(scenario)}(S) ) ].

    If each phi_k(g_k^{(scenario)}(S)) is submodular in S, the average remains submodular.

    Parameters
    ----------
    S : a set of arms
    scenarios : list of scenario dicts, each scenario has random draws for X_{m,k}, N_m, etc.
    K : number of players (or "plays").
    phi_functions : list of callables [phi_0, ..., phi_{K-1}] to handle piecewise bounding.

    Returns
    -------
    float
        The average submodular value across all scenarios.
    """
    total_val = 0.0
    for scenario in scenarios:
        total_val += scenario_value_hprob(S, scenario, K, phi_functions)
    return total_val / len(scenarios)
