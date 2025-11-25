# Fair Multi-Agent Multi-Armed Bandits with Probing (OFMUP)

Implementation of the OFMUP algorithm from the AAAI 2026 paper "Fair Algorithms with Probing for Multi-Agent Multi-Armed Bandits".

## 📁 Core Files for GitHub

### Algorithm Implementation
1. **`online_ofmup.py`** - Main OFMUP algorithm (Algorithm 2 from paper)
   - Implements fair multi-agent UCB with probing
   - Agent-specific greedy policies with epsilon-exploration
   - Sublinear regret guarantee

2. **`greedy.py`** - Offline greedy probing (Algorithm 1 from paper)
   - Submodular optimization for probe set selection
   - Approximation ratio: (e-1)/(2e-1) ≈ 0.316

3. **`f_prob.py`** - Functions for computing g(S) (NSW over probed arms)
   - Piecewise-linear approximation of log(NSW)
   - Submodularity properties

4. **`f_unprob.py`** - Functions for computing h(S) (NSW over unprobed arms)
   - Aggregator functions for greedy assignment
   - Handles both probed and unprobed arms

5. **`compare_real_nsw.py`** - Brute-force optimal subset computation
   - Used for regret calculation
   - Computes optimal NSW for benchmarking

## 📈 Results (T=3000, N=12, K=8)

**Cumulative Regret:**

| Algorithm | Final Regret | Growth | Performance |
|-----------|--------------|--------|-------------|
| **OFMUP** | **13.65** | Sublinear ✓ | **⭐ BEST** |
| Non-Probing | 22.11 | Sublinear ✓ | +62% worse |
| Random P+A | 262.14 | Linear ✗ | +1820% worse |
| Greedy P+Random A | 263.47 | Linear ✗ | +1830% worse |

**Theoretical Bound**: O(√T) ≈ 164.32
**OFMUP Performance**: 0.08× theoretical (12× better than theory!)

**Key Insights**:
1. **Probing helps!** OFMUP outperforms Non-Probing by 38%
2. **Fair assignment crucial**: Both fair algorithms achieve sublinear regret
3. **Agent-specific policies essential**: Random assignment leads to linear regret

## 📚 Citation

```bibtex
@inproceedings{xu2026fair,
  title={Fair Algorithms with Probing for Multi-Agent Multi-Armed Bandits},
  author={Xu, Tianyi and Liu, Jiaxin and Mattei, Nicholas and Zheng, Zizhan},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
