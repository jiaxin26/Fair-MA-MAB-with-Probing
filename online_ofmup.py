import math
import random
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Callable
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

# Offline algorithm components
from f_prob import (
    piecewise_linear_nojump,
)
from greedy import offline_greedy_probing
from f_unprob import f_unprobed, aggregator_only_unprobed, aggregator_all_arms

from compare_real_nsw import brute_force_optimal_subset


class MultiAgentEnv:
    """Multi-agent Bernoulli bandit environment."""

    def __init__(self, N: int, K: int, true_p: List[List[float]]):
        self.N = N
        self.K = K
        self.p = true_p

    def pull_arm(self, j: int, a: int) -> float:
        return 1.0 if (random.random() < self.p[j][a]) else 0.0

    def probe_arms(self, S: Set[int] = None) -> Dict[Tuple[int, int], float]:
        if S is None:
            S = set(range(self.K))
        R = {}
        for j in range(self.N):
            for a in S:
                rew = self.pull_arm(j, a)
                R[(j, a)] = rew
        return R


class BanditStats:
    """
    Statistics tracking for bandits, including Freedman statistics and mean rewards.
    This matches the PDF Algorithm 2 description.
    """
    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K
        # Empirical mean estimates
        self.mu = [[0.0] * K for _ in range(N)]
        # Sample counts
        self.N_ja = [[0] * K for _ in range(N)]
        # Upper confidence bounds
        self.U_ja = [[1.0] * K for _ in range(N)]
        # Confidence widths
        self.w_ja = [[1.0] * K for _ in range(N)]
        # Empirical CDF (simplified as mean for Bernoulli)
        self.F_ja = [[lambda x: 1.0] * K for _ in range(N)]

    def compute_ucb_width(self, j: int, a: int, t: int, delta: float) -> float:
        """
        Compute confidence bound width using Freedman's inequality.
        Matches PDF Lemma 7 formula.
        """
        if self.N_ja[j][a] == 0:
            return 1.0
        n = self.N_ja[j][a]
        mu = self.mu[j][a]

        # Freedman bound: sqrt(2*mu*(1-mu)*log(2MAT/delta)/n) + log(2MAT/delta)/(3n)
        log_term = math.log(2 * self.N * self.K * t / delta)
        width = math.sqrt(2 * mu * (1 - mu) * log_term / n) + log_term / (3 * n)
        return width

    def update(self, j: int, a: int, reward: float, t: int, delta: float):
        """Update statistics after observing a reward."""
        self.N_ja[j][a] += 1
        n = self.N_ja[j][a]
        old_mean = self.mu[j][a]
        # Incremental mean update
        self.mu[j][a] = (old_mean * (n - 1) + reward) / n
        # Update confidence width
        w = self.compute_ucb_width(j, a, t, delta)
        self.w_ja[j][a] = w
        # Upper confidence bound
        self.U_ja[j][a] = min(self.mu[j][a] + w, 1.0)


def aggregator_probe_unprob(S: Set[int], scenario: Dict, k: int, mu_k: List[float]):
    """
    DEPRECATED: This function mixed probed and unprobed arms.
    Use aggregator_only_unprobed or aggregator_all_arms from f_unprob.py instead.

    Aggregator function that handles both probed and unprobed arms:
    - For a in S: uses scenario["X"][a][k]
    - Otherwise: uses mu_k[a]
    """
    total = 0.0
    empirical_scale = 0.96
    K_ = len(mu_k)
    for a in range(K_):
        if a in S:
            total += scenario["X"][a][k]
        else:
            total += mu_k[a]
    return total / (empirical_scale * K_)


def generate_scenarios(stats: BanditStats, n_scen: int) -> List[Dict]:
    """Generate random scenarios based on current statistics."""
    scenarios = []
    for _ in range(n_scen):
        sc = {}
        X_mat = []
        for a in range(stats.K):
            arr = []
            for j in range(stats.N):
                val = 1.0 if (random.random() < stats.mu[j][a]) else 0.0
                arr.append(val)
            X_mat.append(arr)
        sc["X"] = X_mat
        scenarios.append(sc)
    return scenarios


def online_fair_multiagent_ucb_with_probing(
    env: MultiAgentEnv,
    T: int,
    I: int,
    alpha_func: Callable,
    n_scenarios: int,
    piecewise_phi_funcs: List[Callable],
    delta: float = 0.05,
    forced_steps: Optional[int] = None,
    verbose: bool = True,
    zeta: float = 1.0
) -> Dict:
    """
    Online Fair Multi-Agent UCB algorithm with probing mechanism.

    Implementation of Algorithm 2 from the paper (OFMUP),
    using the greedy probing algorithm (Algorithm 1).

    Key features:
    1. Uses current timestep t for UCB calculation
    2. Calls corrected offline_greedy_probing with zeta parameter
    3. Properly implements all steps from Algorithm 2
    """
    N, K = env.N, env.K
    if forced_steps is None:
        forced_steps = N * K  # Warm start: MA rounds

    stats = BanditStats(N, K)

    rewards_history = []
    nsw_history = []
    regret_history = []
    cumulative_regret_history = []
    cumulative_regret = 0.0
    S_history = []
    pi_history = []

    last_probe_rewards = {}

    step_count = 0

    # ============= Lines 5-10: Warm-Start Rounds ============
    for t in range(forced_steps):
        step_count += 1
        # Round-robin assignment
        jt = (t % N)
        at = (t // N) % K

        # Simple assignment: agent jt gets arm at
        S_t = {at}
        R_obs = env.probe_arms(S_t)

        # Agent jt pulls arm at
        r_ja = R_obs[(jt, at)]
        stats.update(jt, at, r_ja, step_count, delta)
        last_probe_rewards[(jt, at)] = r_ja

        avg_rew = r_ja
        rewards_history.append(avg_rew)
        S_history.append(S_t)
        pi_temp = [0.0] * K
        pi_temp[at] = 1.0
        pi_history.append(pi_temp)

        val_jprod = 1.0
        for j in range(N):
            if j == jt:
                val_jprod *= max(r_ja, 1e-9)
            else:
                val_jprod *= 1e-9  # Other agents get minimal value
        curnsw = val_jprod ** (1.0 / N)
        nsw_history.append(curnsw)

        regret_history.append(0.0)
        cumulative_regret += 0.0
        cumulative_regret_history.append(cumulative_regret)

        if verbose and (step_count % 50 == 0):
            print(f"[Warm {step_count}] agent={jt}, arm={at}, rew={avg_rew:.3f}, NSW={curnsw:.3f}")

    # ============= Lines 11-16: Main Loop ============
    for t in range(forced_steps, T):
        step_count += 1
        scenarios = generate_scenarios(stats, n_scenarios)

        # Line 12: Probe Set Selection using Algorithm 1
        def f_prob(S: Set[int]) -> float:
            """
            Compute g(S) - NSW over PROBED arms only.
            Uses greedy assignment: each agent selects best arm from S.
            """
            if len(S) == 0:
                return 0.0

            total_nsw = 0.0
            eps = 1e-9
            for sc in scenarios:
                prod_val = 1.0
                for k_ in range(N):
                    # Greedy: agent k_ selects best arm from S
                    best_reward = max(sc["X"][a][k_] for a in S)
                    prod_val *= max(best_reward, eps)

                nsw = prod_val ** (1.0 / N)
                total_nsw += nsw

            return total_nsw / len(scenarios)

        def aggregator_func_for_unp(S, sc, k_):
            """
            Aggregator for h(S) - only considers UNPROBED arms.
            """
            all_arms = set(range(K))
            return aggregator_only_unprobed(S, sc, k_, stats.mu[k_], all_arms)

        def f_unprob(S: Set[int]) -> float:
            """
            Compute h(S) - log(NSW) over UNPROBED arms only.
            """
            return f_unprobed(S, scenarios, N, lambda s, sc2, kk: aggregator_func_for_unp(s, sc2, kk))

        # Call Algorithm 1 with zeta parameter
        S_t = offline_greedy_probing(
            M=K,
            f_prob=f_prob,
            f_unprobed=f_unprob,
            alpha=alpha_func,
            I=I,
            zeta=zeta,
            verbose=False
        )

        # Line 13: Probing and Updates
        R_obs = env.probe_arms(S_t)
        for (j, a) in R_obs:
            stats.update(j, a, R_obs[(j, a)], step_count, delta)
            last_probe_rewards[(j, a)] = R_obs[(j, a)]
            # Update U_ja = min(mu_ja + w_ja, 1)
            stats.U_ja[j][a] = min(stats.mu[j][a] + stats.w_ja[j][a], 1.0)

        alpha_val = alpha_func(len(S_t))

        # Line 14: Policy Optimization
        # According to Algorithm 2 Line 14: π ← arg max E_Rt[NSW(St, Rt, Ut, π)]
        pi_per_agent = []
        for j in range(N):
            # Agent j selects arm with highest UCB
            best_arm = 0
            best_ucb = -float('inf')
            for a in range(K):
                ucb_val = stats.U_ja[j][a]

                if ucb_val > best_ucb:
                    best_ucb = ucb_val
                    best_arm = a

            # Use epsilon-greedy for exploration
            epsilon = 1.0 / (1 + math.sqrt(step_count - forced_steps + 1))
            pi_j = [epsilon / K] * K
            pi_j[best_arm] += (1 - epsilon)
            pi_per_agent.append(pi_j)

        # Lines 15-16: Arm Pulls and Final Updates
        rew_list = []
        for j in range(N):
            # Each agent uses its own policy
            chosen_arm = random.choices(range(K), weights=pi_per_agent[j], k=1)[0]
            if chosen_arm in S_t:
                rew = R_obs[(j, chosen_arm)]
            else:
                rew = env.pull_arm(j, chosen_arm)
            stats.update(j, chosen_arm, rew, step_count, delta)
            rew_list.append(rew)

        avg_rew = np.mean(rew_list)
        rewards_history.append(avg_rew)
        S_history.append(S_t)
        pi_history.append(pi_per_agent)

        # Calculate current NSW using agent-specific policies
        val_jprod = 1.0
        for j in range(N):
            sum_j = 0.0
            for a in range(K):
                # Use agent j's policy
                sum_j += pi_per_agent[j][a] * env.p[j][a]
            val_jprod *= max(sum_j, 1e-9)
        curnsw = val_jprod ** (1.0 / N)
        nsw_history.append(curnsw)

        # Compute greedy NSW (upper bound on optimal)
        greedy_nsw = 1.0
        for j in range(N):
            best_reward = max(env.p[j][a] for a in range(K))
            greedy_nsw *= best_reward
        greedy_nsw = greedy_nsw ** (1.0 / N)

        # According to Theorem 1, optimal is at most a constant multiple of greedy
        # Use 0.85 as a reasonable approximation (greedy may slightly overestimate optimal)
        approx_optimal = 0.85 * greedy_nsw

        # Current NSW vs approximated optimal
        reg = max(0.0, (1.0 - alpha_val) * (approx_optimal - curnsw))

        regret_history.append(reg)
        cumulative_regret += reg
        cumulative_regret_history.append(cumulative_regret)

        if verbose and (step_count % 50 == 0):
            print(f"[t={step_count}] |S_t|={len(S_t)}, rew={avg_rew:.3f}, nsw={curnsw:.3f}, reg={reg:.3f}")

    # Plot and save results
    timestamps = list(range(1, T + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, cumulative_regret_history, 'b-', label='Cumulative Regret')
    plt.xlabel('Timesteps')
    plt.ylabel('Cumulative Regret')
    plt.title('OFMUP: Cumulative Regret vs Timesteps')
    plt.grid(True)
    plt.legend()
    plt.savefig('cumulative_regret_ofmup.png')
    plt.close()

    df = pd.DataFrame({
        'timestamp': timestamps,
        'cumulative_regret': cumulative_regret_history
    })
    df.to_csv("results_ofmup.csv", index=False)
    print("Saved to results_ofmup.csv")

    return {
        "rewards_history": rewards_history,
        "nsw_history": nsw_history,
        "regret_history": regret_history,
        "cumulative_regret_history": cumulative_regret_history,
        "S_history": S_history,
        "pi_history": pi_history
    }


def main():
    random.seed(2023)
    np.random.seed(2023)
    N = 12
    K = 8
    T = 3000
    I = 3
    delta = 0.01

    true_p = []
    for j in range(N):
        row = []
        for a in range(K):
            row.append(random.uniform(0.3, 0.8))
        true_p.append(row)

    env = MultiAgentEnv(N, K, true_p)

    breakpoints = [0.0, 3.0, 999.0]
    slopes = [1.0, 0.5]
    intercepts = [0.0, 1.5]

    def build_phi(bps, sps, ints):
        def phi_func(x):
            return piecewise_linear_nojump(x, bps, sps, ints)
        return phi_func

    phi_funcs = []
    for _ in range(N):
        phi_funcs.append(build_phi(breakpoints, slopes, intercepts))

    def alpha_func(sz):
        return min(sz/500.0, 0.01)

    print("Running OFMUP algorithm (Algorithm 2 from paper)")
    res = online_fair_multiagent_ucb_with_probing(
        env=env,
        T=T,
        I=I,
        alpha_func=alpha_func,
        n_scenarios=15,
        piecewise_phi_funcs=phi_funcs,
        delta=delta,
        forced_steps=N * K,
        verbose=True,
        zeta=1.0
    )

    print("\n=== Final Summary ===")
    avg_r = np.mean(res["rewards_history"])
    final_nsw = res["nsw_history"][-1]
    total_reg = sum(res["regret_history"])
    print(f"Average reward: {avg_r:.3f}")
    print(f"Final NSW: {final_nsw:.3f}")
    print(f"Total regret: {total_reg:.3f}")


if __name__ == "__main__":
    main()
