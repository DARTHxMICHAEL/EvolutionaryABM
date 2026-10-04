# Evolutionary ABM

A grid-based agent-based model where agents forage, fight and reproduce. The model measures how sensitive the dynamics are to small perturbations, using a finite-time Lyapunov exponent, Shannon entropy and population regime statistics. Agents move either at random or with a small neural network that evolves through basic crossover and mutation.

## Model

- **World:** a `width × height` grid with walls (green), apples (red, $+5$ energy), oranges (orange, $+10$ energy) and agents (blue, with sex $\in \{0, 1\}$).
- **Each tick:** agents act in random order. Each agent pays a metabolic cost $c$ and dies when $E \le 0$. It then moves one cell in one of 8 directions. Walls and the grid edges block movement.
- **Food:** stepping onto food adds its energy. Each tick, $\lfloor \rho \cdot N_{\text{empty}} \rfloor$ new food items spawn: 60% apples, 40% oranges.
- **Same-sex encounter (fight):** the agent with more energy wins and takes the loser's energy, $E_w \leftarrow E_w + E_l$. The loser is removed.
- **Opposite-sex encounter (mating):** children are placed on free cells within radius 2 of the parents' midpoint, and the parents are removed. $E_1, E_2$ are the parents' energies before the reproduction cost:

$$
n = \min\left(n_{\text{free}},\ \left\lfloor \frac{E_1 + E_2}{E_{\min}} \right\rfloor\right), \qquad E_{\text{child}} = \frac{E_1 + E_2}{n}
$$

  Each parent also pays the reproduction cost $c_r$. Because the children's energy uses the totals from before that cost, the cost only has an effect when no child is born ($n = 0$). The parents then survive with $E_i - c_r$.

## Neural agents (`use_nn=True`)

- **Vision:** 8 rays, each with range 8. Each ray returns $[R, G, B, r/8]$ for the first object it hits, giving a 32-value input vector.
- **Network:** $32 \to 16 \to 8$ with
  $$\mathbf{y} = W_2 \tanh(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2, \qquad \text{move} = \arg\max_k y_k$$
- **Inheritance:** each weight comes from either parent with probability 0.5 (uniform crossover). Then each weight mutates with probability 0.1: $w \leftarrow w + \mathcal{N}(0, 0.05^2)$.

## Analysis

Each trial builds two identical grids from the same seed, then perturbs $k$ agents in the second grid by adding $E_{\min}$ to their energy. Both grids then run in lockstep.

**Phase-space distance** (normalized over all cells):

$$
d(t) = \frac{1}{WH} \sum_{i,j} \delta_{ij}, \qquad
\delta_{ij} =
\begin{cases}
1 & \text{cell types differ} \\
\tfrac12 \mathbb{1}[s_1 \ne s_2] + \tfrac12 \dfrac{|E_1 - E_2|}{\max(|E_1|, |E_2|, 1)} & \text{both cells hold agents} \\
0 & \text{otherwise}
\end{cases}
$$

**Lyapunov exponent:** a linear fit over the first $\max(10,\ \text{cutoff} \cdot T)$ ticks:

$$
d(t) \approx d_0 e^{\lambda t} \quad\Rightarrow\quad \log d(t) \approx \log d_0 + \lambda t
$$

**Shannon entropy:** computed over coarse-grained cell states (empty, wall, apple, orange, and agent × sex × high/low energy). The model reports $\Delta H = |H_2 - H_1|$ at the final tick.

$$
H = -\sum_i p_i \log_2 p_i
$$

**Regime metrics**, computed on grid 1 after the first 30% of ticks (burn-in):
- growth rate $r$, the slope of $\log N(t)$
- coefficient of variation $\mathrm{CV} = \sigma_N / \mu_N$
- viability: both sexes are present at the end
- preservation: the final population is at least the initial population

Results are averaged over `num_runs` trials. Before the trials start, a determinism check confirms that two runs from the same seed give identical grids.

## Usage

```bash
pip install numpy matplotlib
python evolution_abm.py
```

The script runs three experiments one after another:
1. Random agents in the near-critical regime
2. Neural agents in the near-critical regime
3. Random agents under the same constraints as experiment 2

Settings are at the bottom of the file: `grid_params`, `num_runs`, `num_ticks`, `num_prtrb_agents`, `init_seed` and `cutoff`. With the defaults (20 runs × 15,000 ticks each), a full run is slow and opens many plot windows. Lower `num_ticks` and `num_runs` for quick tests.
