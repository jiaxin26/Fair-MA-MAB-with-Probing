# offline_greedy.py

def offline_greedy_probing(M,
                           f_prob,
                           f_unprobed,
                           alpha,
                           I,
                           verbose=False):
    """
    Greedy algorithm that iteratively selects arms to probe while considering overhead costs.
    Returns the optimal probing set S that maximizes (1-alpha(|S|))*f_prob(S).
    """
    S_list = [set()]  # S_0 = ∅

    for i in range(1, I):
        S_prev = S_list[i-1]
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
            S_list.append(S_prev.copy())
        else:
            S_new = S_prev | {best_arm}
            S_list.append(S_new)

        if verbose:
            print(f"Greedy step i={i}, chosen arm={best_arm}, "
                  f"marginal gain={best_inc:.4f}, set={S_list[i]}")

    best_combo_val = float('-inf')
    best_j = 0
    for i in range(I):
        S_i = S_list[i]
        disc = (1.0 - alpha(i))
        val_i = disc * f_prob(S_i)
        if val_i > best_combo_val:
            best_combo_val = val_i
            best_j = i

    S_pr = S_list[best_j]
    final_val = best_combo_val
    if verbose:
        print(f"Best j = {best_j}, (1 - alpha({best_j})) * f_prob(S_j) = {final_val:.4f}")
        print(f"S_j = {S_pr}")

    # Compare with f_unprobed(∅):
    val_unp = f_unprobed(set())
    if final_val < val_unp:
        if verbose:
            print(f"(1 - alpha({best_j})) * f_prob(S_j) < f_unprobed(∅). Returning ∅.")
        S_pr = set()

    return S_pr
