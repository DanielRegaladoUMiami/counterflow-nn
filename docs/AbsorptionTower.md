# AbsorptionTower — physically-exact absorber as a PyTorch layer

`src/absorption_tower.py` implements a multistage countercurrent gas
absorber in closed form. Every physical parameter (Henry's constant,
solvent-to-gas ratio, Murphree plate efficiency, equilibrium intercept)
is a learnable tensor, so the whole tower is a single differentiable
operation that can be dropped into any PyTorch model.

This document explains the physics, the derivation of the closed-form
solver, the Python API, and the validation record.

---

## 1. Physical model

At every tray `n`, three equations hold simultaneously:

| equation | meaning |
|---|---|
| `y_n* = m · x_n + b` | Henry's law (linear equilibrium) |
| `y_n = y_{n+1} + E · (y_n* - y_{n+1})` | Murphree plate efficiency, `0 < E ≤ 1` |
| `y_{n+1} = y_1 + (L/G) · (x_n - x_0)` | operating line (overall mass balance) |

Boundary conditions at the column ends:

- `y_{N+1} = y_feed` — feed gas enters the **bottom** (tray `N`)
- `x_0 = x_top` — lean solvent enters the **top** (tray `1`)
- `y_1` and `x_N` are the outputs of interest

The absorption factor is `A = L / (m·G)`. `A > 1` is the feasible
regime; `A = 1` is the design pinch; `A < 1` reverses the direction
(gas stripping).

---

## 2. Closed-form derivation

Substitute the operating line into the Murphree relation to eliminate
`x_n`:

```
y_n = (1 - E)·y_{n+1} + E·(m · x_n + b)
    = (1 - E)·y_{n+1} + E·(m · (x_0 + (y_{n+1} - y_1)/(L/G)) + b)
    = β · y_{n+1} + γ · y_1 + δ
```

with

```
β  = (1 - E) + E · S       where S = m·G/L = 1/A
γ  = -E · S
δ  = E · (m · x_0 + b)
```

Apply the recurrence from `n = N` down to `n = 1`, using the known
boundary `y_{N+1} = y_feed`:

```
y_1 = β^N · y_feed  +  γ · y_1 · S_N  +  δ · S_N
```

where `S_N = 1 + β + β² + … + β^{N-1} = (β^N − 1)/(β − 1)` (with the
obvious limit `S_N = N` when `β = 1`).

Solving for `y_1`:

```
          β^N · y_feed  +  δ · S_N
y_1  =  ─────────────────────────────
              1  −  γ · S_N
```

Once `y_1` is known, the overall mass balance gives the rich-liquid
exit:

```
x_N  =  x_0  +  (y_feed − y_1) / (L/G)
```

Every intermediate stage composition follows by marching the same
recurrence.

**This collapses the whole tower into a single closed-form step.**
No tray-by-tray iteration, no fixed-point solver, no internal loop.
All operations are native PyTorch tensor ops, so gradients flow
through every parameter — including `N`, treated as a fixed integer
hyperparameter.

The `β → 1` (pinch, `A = 1`) removable singularity is handled by a
`torch.where` branch that switches to the `S_N = N` limit.

---

## 3. Parameter encoding

To keep physics valid during training:

| parameter | constraint | encoding |
|---|---|---|
| `L/G` | `> 0` | `exp(log_L_over_G)` |
| `m`   | `> 0` | `exp(log_m)` |
| `E`   | `∈ (0, 1)` | `sigmoid(logit_E)` |
| `b`   | free | stored as-is |

Each parameter has shape `(d,)` — one value per feature channel —
so the module solves `d` independent towers in parallel in a single
forward pass.

---

## 4. API

```python
from src.absorption_tower import AbsorptionTower

tower = AbsorptionTower(
    d=4,                   # feature dimension / # parallel species
    n_stages=6,            # number of equilibrium trays N
    L_over_G_init=1.5,     # initial L/G ratio
    m_init=0.7,            # initial Henry's constant
    E_init=0.85,           # initial Murphree efficiency
    b_init=0.0,            # initial equilibrium intercept
)

y_feed = torch.rand(batch, 4)       # gas composition at bottom
x_top  = torch.zeros(batch, 4)      # lean solvent at top

# Forward solve — O(1) in N, fully differentiable
y_top, x_bot = tower(y_feed, x_top)

# Full tray-by-tray profiles (for diagnostics / plotting)
profiles = tower.profiles(y_feed, x_top)
#   profiles["y_stages"]  shape (batch, N+1, d)
#   profiles["x_stages"]  shape (batch, N+1, d)

# Design diagnostics
print("A = L/(m·G):", tower.A)
print("Fraction absorbed:", tower.fraction_absorbed(y_feed, x_top))
```

The companion `AbsorptionNetwork` wraps the tower in sigmoid-bounded
encoders and an MLP head for drop-in classification/regression.

---

## 5. Validation record

### Unit tests — `tests/test_absorption_tower.py`

30 tests covering:

- Construction, parameter initialisation, range enforcement
- Kremser closed form, six parameter combinations of `(A, N)`
- Hand-computed case `A=2, N=2 ⇒ y_1 = 1/7`
- Overall mass balance across random parameter draws
- Operating line at every internal stage
- Murphree relation at every tray
- `β → 1` pinch stability, `E → 0` no-transfer limit, `A ≫ 1` limit
- Gradients through all learnable parameters, through inputs, and
  through a full training step that drops the loss by ≥80 %
- `AbsorptionNetwork` shape and gradient flow

### Textbook validation — `experiments/tier0_physical_validation.py`

| example | A | N | E | absorbed | rel. error vs reference |
|---|---|---|---|---|---|
| Treybal 8.2 — acetone/air/water | 1.119 | 6 | 1.00 | 90.06 % | 1.3·10⁻⁵ |
| Seader 6.1 — n-butane absorber  | 2.630 | 8 | 1.00 | 99.97 % | 1.3·10⁻⁴ |
| Pinch (A = 1)                   | 1.000 | 5 | 1.00 | 83.33 % | 8.3·10⁻⁶ |
| Real trays (Murphree E = 0.70)  | 1.119 | 6 | 0.70 | 84.74 % | 3.1·10⁻⁸ |

All four reproduce the reference value to better than `10⁻³`
relative error. Every comparison is against either the classical
Kremser formula (ideal stages) or a brute-force tray-by-tray
iterative solver (non-ideal stages).

---

## 6. When to prefer this over CFNN-A

`AbsorptionTower` is **rigid**: its forward pass is exactly a
countercurrent linear-equilibrium absorber. On generic datasets it
underperforms an MLP because the rigid bias costs expressiveness.

Use it when:

- the problem **is** a tower (digital twin of a real absorber, inverse
  design of operating conditions, fitting plant data);
- interpretability of learned parameters matters — `m`, `L/G`, `E`
  are legible physical quantities, not opaque weights;
- sample efficiency matters — a strong physics prior needs far fewer
  observations than a generic MLP.

Prefer the softer `CounterFlowNetwork` (CFNN-A) when you want the
counterflow inductive bias without hard-coded Henry's law — the
equilibrium mapping is then a learned function instead of a linear
constraint.

---

## 7. References

- Treybal, R.E. *Mass Transfer Operations*, 3rd ed., McGraw-Hill, 1980
  — Ch. 8, eqs. 8.44–8.50 (Kremser equation)
- Seader, J.D., Henley, E.J., Roper, D.K. *Separation Process
  Principles*, 3rd ed., Wiley, 2011 — Ch. 6
- Murphree, E.V. "Rectifying column calculations — with particular
  reference to n-component mixtures", *Ind. Eng. Chem.* **17**
  (7): 747–750, 1925
