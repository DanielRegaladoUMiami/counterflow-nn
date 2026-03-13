# CounterFlow Neural Network (CFNN)

**A Neural Network Architecture Inspired by Chemical Engineering Unit Operations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> *"The same equations that govern mass transfer in absorption towers,
> separation in distillation columns, and reaction in cascaded reactors
> can define a new class of neural network architectures."*

## The Idea

Standard neural networks propagate information in a single direction. **CFNN** introduces two coupled streams flowing in **opposite directions** with continuous exchange — exactly like a chemical absorption tower:

```
Input x                          Initial context c₀
   ↓                                    ↓
┌──────────────────────────────────────────┐
│  g₀ = Encoder_g(x)    l₀ = Encoder_l(c₀)│
└──────────────────────────────────────────┘
   ↓                                    ↑
┌──────────────────────────────────────────┐
│              Plate 1                      │
│  Δ₁ = T₁(g₀ - E₁(l₁))    ← driving    │
│  g₁ = g₀ - Δ₁     l₁ = l₂ + Δ₁  force │
└──────────────────────────────────────────┘
   ↓                                    ↑
┌──────────────────────────────────────────┐
│              Plate 2                      │
│  Δ₂ = T₂(g₁ - E₂(l₂))                  │
│  g₂ = g₁ - Δ₂     l₂ = l₃ + Δ₂        │
└──────────────────────────────────────────┘
   ↓                                    ↑
              ...N plates...
   ↓                                    ↑
┌──────────────────────────────────────────┐
│              Plate N                      │
│  g_N = g_{N-1} - Δ_N  l_N = 0 + Δ_N    │
└──────────────────────────────────────────┘
   ↓                                    ↓
 Output_gas = g_N              Output_liquid = l_1
 (cleaned features)            (extracted representation)
```

### Key Properties

- **Conservation**: What the gas stream loses, the liquid stream gains (Δg = -Δl)
- **Driving force**: Information exchange is proportional to the difference between streams
- **Equilibrium mapping**: A learned function defines "equilibrium" between the two streams
- **Counterflow advantage**: Maintains a non-zero gradient at every layer (may help with vanishing gradients)

## Architecture Variants

| Variant | Inspired by | Key Feature |
|---------|------------|-------------|
| **CFNN-A** | Absorption tower | Unidirectional transfer (gas → liquid) |
| **CFNN-D** | Distillation column | Bidirectional transfer + feed plate + reflux |
| **CFNN-R** | Reactor cascade | CSTR pre/post processing + counterflow core |

## ChemE → Neural Network Mapping

| Chemical Engineering | Neural Network |
|---------------------|---------------|
| Gas stream (ascending) | Feature stream (forward) |
| Liquid stream (descending) | Context stream (backward) |
| Mass transfer coefficient Kya | Learnable transfer coefficient α |
| Equilibrium curve Y* = f(X) | Learned equilibrium function E(l) |
| Driving force (Y - Y*) | Stream difference δ = g - E(l) |
| Number of transfer units (NTU) | Network depth (number of plates) |
| Conservation of mass | Information conservation constraint |
| McCabe-Thiele diagram | Neural McCabe-Thiele visualization |

## Project Structure

```
counterflow-nn/
├── src/
│   ├── __init__.py
│   ├── plates.py              # CounterFlowPlate (absorption)
│   ├── network.py             # CounterFlowNetwork (CFNN-A)
│   ├── distillation.py        # DistillationPlate + DistillationNetwork (CFNN-D)
│   ├── activations.py         # Michaelis-Menten, Arrhenius, Hill, Autocatalytic
│   ├── diagnostics.py         # Damköhler number, Murphree efficiency, NTU
│   └── utils.py               # Training utilities, data loading
├── experiments/
│   ├── tier1_synthetic.py     # Moons, circles, XOR
│   ├── compare_baselines.py   # CFNN-A vs MLP vs ResMLP (Phase 1)
│   ├── tier2_distillation.py  # CFNN-D vs CFNN-A vs MLP (Phase 2)
│   └── tier3_mnist.py         # MNIST / FashionMNIST (Phase 2)
├── notebooks/
│   ├── 00_run_experiments.ipynb          # Phase 1 Colab notebook
│   └── 01_phase2_distillation.ipynb      # Phase 2 Colab notebook
├── tests/
│   ├── test_plates.py         # Unit tests: conservation, dimensions (16 tests)
│   └── test_distillation.py   # Unit tests: CFNN-D bidirectional, reflux (20 tests)
├── docs/
│   ├── CFNN_Technical_Documentation.md
│   └── CFNN_Execution_Plan.md
├── app.py                     # Gradio Space for HuggingFace
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Quick Start

```bash
# Clone
git clone https://github.com/DanielRegaladoUMiami/counterflow-nn.git
cd counterflow-nn

# Install
pip install -e ".[dev,demo]"

# Run Phase 1 experiments
python experiments/tier1_synthetic.py
python experiments/compare_baselines.py

# Run Phase 2 experiments
python experiments/tier2_distillation.py
python experiments/tier3_mnist.py
```

## Usage

```python
from src.network import CounterFlowNetwork
from src.distillation import DistillationNetwork

# CFNN-A: Absorption mode (unidirectional transfer)
model_a = CounterFlowNetwork(
    d_in=784, d_gas=64, d_liquid=64,
    n_plates=5, d_out=10, n_sweeps=2,
)

# CFNN-D: Distillation mode (bidirectional + feed plate + reflux)
model_d = DistillationNetwork(
    d_in=784, d_gas=64, d_liquid=64,
    n_plates_rect=3, n_plates_strip=3,
    d_out=10, n_sweeps=2,
    reflux_ratio=0.3, reboil_ratio=0.2,
)

# Forward pass (both share the same interface)
output = model_a(x)  # x: (batch_size, 784)
output = model_d(x)

# Diagnostics
from src.diagnostics import print_diagnostics
print_diagnostics(model_a, x, model_name="CFNN-A")
```

## References

- Treybal, R.E. *"Mass Transfer Operations,"* 3rd Ed., McGraw-Hill, 1980
- Fogler, H.S. *"Elements of Chemical Reaction Engineering,"* 6th Ed.
- Bai, S., Kolter, J.Z., Koltun, V. *"Deep Equilibrium Models,"* NeurIPS 2019
- Chen, R.T.Q. et al. *"Neural Ordinary Differential Equations,"* NeurIPS 2018

## Author

**Daniel Regalado Cardoso** — BS Chemical Engineering + MSBA
University of Miami | [GitHub](https://github.com/DanielRegaladoUMiami)

## License

MIT License — see [LICENSE](LICENSE) for details.
