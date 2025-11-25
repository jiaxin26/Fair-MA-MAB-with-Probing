"""
Functions for computing h(S) - the utility from UNPROBED arms only.
"""

import math


def aggregator_only_unprobed(S, scenario, k, mu_k, all_arms):
    """
    Compute optimal reward for agent k from ONLY UNPROBED arms (arms NOT in S).

    Parameters
    ----------
    S : set of int
        Probing set
    scenario : dict
        Scenario (not used for unprobed arms - we use estimates)
    k : int
        Agent index
    mu_k : list of float
        Mean rewards mu_k[a] for agent k and arm a
    all_arms : set of int
        Set of all available arms [A]

    Returns
    -------
    float
        Best estimated reward from unprobed arms
    """
    unprobed_rewards = [mu_k[a] for a in all_arms if a not in S]

    if len(unprobed_rewards) > 0:
        return max(unprobed_rewards)
    else:
        return 0.0


def aggregator_all_arms(S, scenario, k, mu_k, all_arms):
    """
    Compute optimal reward for agent k from ALL arms (both probed and unprobed).

    For probed arms in S: use scenario realization
    For unprobed arms not in S: use mean mu_k[a]

    Parameters
    ----------
    S : set of int
        Probing set
    scenario : dict
        Scenario with realizations scenario["X"][a][k]
    k : int
        Agent index
    mu_k : list of float
        Mean rewards for agent k
    all_arms : set of int
        Set of all arms

    Returns
    -------
    float
        Best reward from all arms
    """
    rewards = []
    for a in all_arms:
        if a in S:
            # Probed: use scenario realization
            rewards.append(scenario["X"][a][k])
        else:
            # Unprobed: use mean
            rewards.append(mu_k[a])

    return max(rewards) if len(rewards) > 0 else 0.0


def f_unprobed(S, scenarios, K, aggregator_func, eps=1e-9):
    """
    Compute NSW (geometric mean) using the provided aggregator function.

    IMPORTANT: Returns NSW itself, not log(NSW)!

    Parameters
    ----------
    S : set of int
        Probing set
    scenarios : list of dict
        Random scenarios
    K : int
        Number of agents
    aggregator_func : callable
        Function (S, scenario, k) -> reward for agent k
        Should be aggregator_only_unprobed for h(S) or
        aggregator_all_arms for full R(S)
    eps : float
        Minimum value to avoid division by zero

    Returns
    -------
    float
        Average NSW (geometric mean) across scenarios
    """
    total_nsw = 0.0
    for sc in scenarios:
        # Compute NSW = geometric mean of utilities
        prod_val = 1.0
        for k in range(K):
            val_k = aggregator_func(S, sc, k)
            prod_val *= max(val_k, eps)

        # Geometric mean: (∏ val_k)^(1/K)
        nsw = prod_val ** (1.0 / K)
        total_nsw += nsw

    return total_nsw / len(scenarios)
