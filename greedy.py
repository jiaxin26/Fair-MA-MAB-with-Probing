# offline_greedy.py
# Implementation of Algorithm 1 from the paper

def offline_greedy_probing(M,
                           f_prob,
                           f_unprobed,
                           alpha,
                           I,
                           zeta=1.0,
                           compute_R=None,
                           verbose=False):
    """
    Greedy algorithm that implements Algorithm 1 from the paper.

    Key features:
    1. Loop from i=0 to I (inclusive), creating I+1 sets
    2. Sort candidates by (1-alpha(|S_j|))*f_upper(S_j)
    3. Check condition: (1-alpha(|S_j|))*f_upper(S_j) > zeta*R(S_j)
    4. Return optimal probing set S_pr

    Parameters:
    -----------
    M : int
        Number of arms
    f_prob : callable
        Function f_upper(S) or f_prob(S) for probed arms
    f_unprobed : callable
        Function h(S) for unprobed arms
    alpha : callable
        Overhead function alpha(|S|)
    I : int
        Maximum probing budget
    zeta : float, optional (default=1.0)
        Scaling factor for R(S) comparison
    compute_R : callable, optional
        Function to compute R(S) = (1-alpha(|S|)) * E[NSW(S, R, mu, pi*(S))]
        If None, approximates R(S) ≈ (1-alpha(|S|)) * (f_prob(S) + h(S))
    verbose : bool, optional
        Print debug information

    Returns:
    --------
    set
        Optimal probing set S_pr
    """
    S_list = [set() for _ in range(I + 1)]  # S_0 to S_I

    for i in range(1, I + 1):
        S_prev = S_list[i - 1]
        best_inc = float('-inf')
        best_arm = None

        current_val = f_prob(S_prev)
        available_arms = set(range(M)) - S_prev

        for m in available_arms:
            cand_set = S_prev | {m}
            val_cand = f_prob(cand_set)
            inc = val_cand - current_val
            if inc > best_inc:
                best_inc = inc
                best_arm = m

        if best_arm is None:
            S_list[i] = S_prev.copy()
        else:
            S_list[i] = S_prev | {best_arm}

        if verbose:
            print(f"Greedy step i={i}, chosen arm={best_arm}, "
                  f"marginal gain={best_inc:.4f}, |S_{i}|={len(S_list[i])}")

    Pi = list(range(I + 1))

    def sort_key(i):
        S_i = S_list[i]
        return (1.0 - alpha(len(S_i))) * f_prob(S_i)

    Pi.sort(key=sort_key, reverse=True)

    if verbose:
        print("\nSorted candidates by (1-alpha(|S|))*f_upper(S):")
        for idx, j in enumerate(Pi):
            val = sort_key(j)
            print(f"  Rank {idx}: j={j}, |S_j|={len(S_list[j])}, value={val:.4f}")

    h_empty = f_unprobed(set())

    for j in Pi:  # Iterate from largest to smallest upper-bound
        S_j = S_list[j]
        alpha_j = alpha(len(S_j))
        f_upper_j = f_prob(S_j)
        upper_bound = (1.0 - alpha_j) * f_upper_j

        if upper_bound < h_empty:
            if verbose:
                print(f"\nCondition failed: (1-alpha({len(S_j)}))*f_upper < h(∅)")
                print(f"  {upper_bound:.4f} < {h_empty:.4f}")
                print(f"Returning S_pr = ∅")
            return set()

        if compute_R is not None:
            R_j = compute_R(S_j)
        else:
            # Approximation: R(S) ≈ (1-alpha(|S|)) * (f_prob(S) + h(S))
            h_j = f_unprobed(S_j)
            R_j = (1.0 - alpha_j) * (f_upper_j + h_j)

        if upper_bound > zeta * R_j:
            if verbose:
                print(f"\nSkipping j={j}: upper_bound={upper_bound:.4f} > "
                      f"zeta*R(S_j)={zeta * R_j:.4f}")
            continue

        if verbose:
            print(f"\nSelected S_pr with j={j}, |S_j|={len(S_j)}")
            print(f"  (1-alpha)*f_upper = {upper_bound:.4f}")
            print(f"  zeta*R(S_j) = {zeta * R_j:.4f}")
            print(f"  h(∅) = {h_empty:.4f}")

        return S_j

    # If no valid candidate found, return empty set
    if verbose:
        print("\nNo valid candidate found, returning S_pr = ∅")
    return set()
