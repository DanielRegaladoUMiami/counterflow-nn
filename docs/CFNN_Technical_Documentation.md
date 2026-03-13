# CounterFlow Neural Network (CFNN)

## A Neural Network Architecture Inspired by Chemical Engineering Unit Operations

**Author:** Daniel Regalado Cardoso  
**Date:** March 2026  
**Status:** Research Concept / Pre-Implementation

> *"The same equations that govern mass transfer in absorption towers,  
> separation in distillation columns, and reaction in cascaded reactors  
> can define a new class of neural network architectures."*

---

## 1. Motivation

### 1.1 The Problem with Standard Architectures

Standard feedforward neural networks propagate information in a single direction — from input to output. While effective, this architecture has a fundamental limitation: each layer only refines the representation based on what came before it. There is no mechanism for later layers to "inform" earlier processing stages during a single forward pass.

Existing approaches to address this include:

| Architecture | Mechanism | Limitation |
|---|---|---|
| ResNets | Skip connections | Same flow direction, additive only |
| Transformers | Self-attention | Bidirectional attention but single representation stream |
| Bidirectional LSTMs | Two directional passes | Flows are independent, no exchange between them |
| Deep Equilibrium Models (DEQ) | Fixed-point iteration | Single representation seeking equilibrium, no dual streams |
| U-Net | Encoder-decoder with skip | Two phases (down then up), not simultaneous counterflow |

None of these architectures implement **two coupled streams flowing in opposite directions with continuous exchange** — which is exactly what happens in a chemical absorption tower.

### 1.2 The Chemical Engineering Inspiration

In a packed absorption tower (Treybal, "Mass Transfer Operations"):

- A **gas stream** enters at the bottom carrying a solute (e.g., NH₃ in air) and flows **upward**
- A **liquid stream** enters at the top (clean solvent, e.g., water) and flows **downward**
- At every differential element `dZ`, mass transfer occurs from gas to liquid
- The driving force for transfer is the concentration difference `(y - y*)`, where `y*` is the equilibrium concentration
- After N stages (or height Z), the gas exits clean and the liquid exits loaded with solute

This countercurrent arrangement is thermodynamically optimal — it maintains a driving force throughout the entire column, unlike co-current flow which reaches equilibrium and stops transferring.

### 1.3 The Hypothesis

**A neural network with two coupled countercurrent information streams can learn richer representations than equivalent single-stream architectures**, because:

1. The counterflow maintains a "driving force" (information gradient) at every layer
2. Each stream specializes: one carries raw features, the other carries refined context
3. The exchange mechanism naturally implements iterative refinement
4. The architecture has an inductive bias toward separation/extraction of relevant features

---

## 2. Chemical Engineering Foundations

### 2.1 Absorption Tower Equations

#### Material Balance (Steady-State, Countercurrent)

For a differential element `dZ` of the tower:

**Gas phase:**
$$G_s \, dY = K_{Ya} \, (Y - Y^*) \, dZ$$

**Liquid phase:**
$$L_s \, dX = K_{Xa} \, (X^* - X) \, dZ$$

Where:
- $G_s$ = molar flux of inert gas (mol/time·area)
- $L_s$ = molar flux of solvent (mol/time·area)
- $Y$ = moles solute / moles inert gas (gas phase)
- $X$ = moles solute / moles solvent (liquid phase)
- $K_{Ya}$ = overall gas-phase mass transfer coefficient × interfacial area
- $Y^*$ = gas-phase composition in equilibrium with liquid composition $X$
- $X^*$ = liquid-phase composition in equilibrium with gas composition $Y$

#### Overall Material Balance

$$G_s (Y_1 - Y_2) = L_s (X_1 - X_2)$$

This is the **operating line** — a linear relationship between gas and liquid compositions at any point in the tower.

#### Equilibrium Relationship

For dilute systems (Henry's Law):
$$Y^* = m \cdot X$$

Where $m$ is the slope of the equilibrium curve (Henry's constant in appropriate units).

#### Number of Transfer Units (NTU)

$$N_{tOG} = \int_{Y_2}^{Y_1} \frac{dY}{Y - Y^*}$$

For linear equilibrium and operating lines:

$$N_{tOG} = \frac{1}{1 - \frac{mG_s}{L_s}} \ln \left[ \left(1 - \frac{mG_s}{L_s}\right) \frac{Y_1 - mX_2}{Y_2 - mX_2} + \frac{mG_s}{L_s} \right]$$

#### Height of Transfer Unit (HTU)

$$H_{tOG} = \frac{G_s}{K_{Ya}}$$

#### Total Tower Height

$$Z = N_{tOG} \times H_{tOG}$$

### 2.2 Key Physical Principles to Translate

| Chemical Principle | Mathematical Form | Neural Analog |
|---|---|---|
| Driving force | $(Y - Y^*)$ | Difference between raw features and equilibrium mapping of context |
| Mass transfer rate | $N_A = K \cdot (Y - Y^*)$ | Information exchange proportional to driving force |
| Equilibrium | $Y^* = f(X)$ | Learned mapping between the two streams |
| Conservation | $G_s \, dY = L_s \, dX$ | What one stream loses, the other gains |
| Operating line | Linear relationship G-L | Constraint on total information flow |
| Minimum L/G ratio | $\left(\frac{L}{G}\right)_{min}$ | Minimum context capacity needed |
| NTU | Number of ideal stages | Network depth (number of layers) |
| HTU | Efficiency per stage | Learning rate per layer |

---

## 3. The CFNN Architecture

### 3.1 Overview

The CounterFlow Neural Network consists of:

1. **Two streams**: Gas stream $\mathbf{g}$ (ascending) and Liquid stream $\mathbf{l}$ (descending)
2. **N exchange plates**: Each plate performs bidirectional information transfer
3. **An equilibrium function** $\mathcal{E}$: Learned mapping that defines "equilibrium" between streams
4. **A transfer function** $\mathcal{T}$: Controls how much information transfers per plate

```
Input x                          Initial context c₀
   ↓                                    ↓
┌──────────────────────────────────────────┐
│  g₀ = Encoder_g(x)    l₀ = Encoder_l(c₀)│
└──────────────────────────────────────────┘
   ↓                                    ↑
┌──────────────────────────────────────────┐
│              Plate 1                      │
│  Δ₁ = T₁(g₀ - E₁(l₁))                  │
│  g₁ = g₀ - Δ₁          l₁ = l₂ + Δ₁    │
└──────────────────────────────────────────┘
   ↓                                    ↑
┌──────────────────────────────────────────┐
│              Plate 2                      │
│  Δ₂ = T₂(g₁ - E₂(l₂))                  │
│  g₂ = g₁ - Δ₂          l₂ = l₃ + Δ₂    │
└──────────────────────────────────────────┘
   ↓                                    ↑
              ...N plates...
   ↓                                    ↑
┌──────────────────────────────────────────┐
│              Plate N                      │
│  Δ_N = T_N(g_{N-1} - E_N(l_N))          │
│  g_N = g_{N-1} - Δ_N    l_N = 0 + Δ_N   │
└──────────────────────────────────────────┘
   ↓                                    ↓
 Output_gas = g_N              Output_liquid = l_accumulated
 (cleaned features)            (extracted representation)
```

### 3.2 Formal Definition

#### Initialization

Given input $\mathbf{x} \in \mathbb{R}^{d_{in}}$ and optional initial context $\mathbf{c}_0 \in \mathbb{R}^{d_l}$:

$$\mathbf{g}_0 = W_g \mathbf{x} + \mathbf{b}_g \quad \in \mathbb{R}^{d_g}$$

$$\mathbf{l}_{N+1} = W_l \mathbf{c}_0 + \mathbf{b}_l \quad \in \mathbb{R}^{d_l}$$

If no initial context is available, $\mathbf{l}_{N+1} = \mathbf{0}$ (clean solvent entering the top of the tower).

#### Plate Dynamics (for plate $n = 1, 2, \ldots, N$)

**Step 1: Compute equilibrium mapping**

$$\mathbf{e}_n = \mathcal{E}_n(\mathbf{l}_{n+1}) = \sigma(W_n^{eq} \, \mathbf{l}_{n+1} + \mathbf{b}_n^{eq})$$

This maps the current liquid-stream state to its "equilibrium" gas-phase equivalent. Analogous to $Y^* = f(X)$.

Note: The liquid stream at plate $n$ depends on what flows down from plate $n+1$. This creates a **backward dependency** — plates must be solved simultaneously or iteratively (see Section 3.3).

**Step 2: Compute driving force**

$$\boldsymbol{\delta}_n = \mathbf{g}_{n-1} - \mathbf{e}_n$$

This is the analog of $(Y - Y^*)$ — the difference between the gas stream's current state and the equilibrium value dictated by the liquid stream.

**Step 3: Compute transfer amount**

$$\boldsymbol{\Delta}_n = \mathcal{T}_n(\boldsymbol{\delta}_n) = \alpha_n \cdot \text{tanh}(W_n^{tr} \, \boldsymbol{\delta}_n + \mathbf{b}_n^{tr})$$

Where $\alpha_n > 0$ is a learnable transfer coefficient (analogous to $K_{Ya}$). The `tanh` ensures bounded transfer per plate, preventing instability.

**Step 4: Update both streams (conservation)**

$$\mathbf{g}_n = \mathbf{g}_{n-1} - \boldsymbol{\Delta}_n \quad \text{(gas loses "solute")}$$

$$\mathbf{l}_n = \mathbf{l}_{n+1} + \boldsymbol{\Delta}_n \quad \text{(liquid gains "solute")}$$

This enforces the **conservation law**: $\Delta \mathbf{g} = -\Delta \mathbf{l}$. What the gas stream loses, the liquid stream gains. This is directly from the material balance $G_s \, dY = L_s \, dX$.

#### Output

$$\hat{\mathbf{y}} = W_{out} \, [\mathbf{g}_N \| \mathbf{l}_1] + \mathbf{b}_{out}$$

Where $\|$ denotes concatenation. The output uses both:
- $\mathbf{g}_N$: the "cleaned" gas (features with relevant information extracted)
- $\mathbf{l}_1$: the "loaded" liquid (accumulated extracted representation)

### 3.3 The Backward Dependency Problem

There is a subtlety: plate $n$ needs $\mathbf{l}_{n+1}$ (liquid from the plate above), but plate $n+1$ needs $\mathbf{g}_n$ (gas from the plate below). This creates a **coupled system** — exactly as in a real absorption tower.

Three approaches to resolve this:

#### Approach A: Iterative Solution (True Counterflow)

Initialize all $\mathbf{l}_n$ to zero (or a guess), then iterate:

```
for iteration in range(K):  # K convergence iterations
    # Forward pass (gas ascending)
    for n in range(1, N+1):
        e_n = E_n(l[n+1])
        delta_n = T_n(g[n-1] - e_n)
        g[n] = g[n-1] - delta_n
    
    # Backward pass (liquid descending)
    for n in range(N, 0, -1):
        e_n = E_n(l[n+1])  # l[n+1] is now updated
        delta_n = T_n(g[n-1] - e_n)
        l[n] = l[n+1] + delta_n
```

This is analogous to how real towers are solved (tray-by-tray calculation with iteration). It naturally connects to **Deep Equilibrium Models** (Bai et al., 2019) which find fixed points of implicit layers.

**Pros:** Most physically faithful. Maintains true counterflow.  
**Cons:** Multiple forward passes per training step. Memory-intensive for backprop through iterations.

#### Approach B: Sequential Approximation (Practical)

Solve from bottom to top using the **previous iteration's** liquid values (or initial guess for the first pass):

```
# Initialize liquid stream (clean solvent from top)
l[N+1] = initial_context  # or zeros

# Single sweep: bottom to top for gas, top to bottom for liquid
# Use delayed liquid values (from previous training step or initialization)
for n in range(1, N+1):
    e_n = E_n(l_prev[n+1])  # use previous/initial liquid
    delta_n = T_n(g[n-1] - e_n)
    g[n] = g[n-1] - delta_n
    l[n] = l_prev[n+1] + delta_n
```

**Pros:** Single forward pass. Standard backpropagation works.  
**Cons:** Approximate counterflow. Liquid values are "stale" by one step.

#### Approach C: Unrolled Bidirectional (Recommended for MVP)

Process in alternating sweeps within a single forward pass:

```
# Initialize
g[0] = encode(x)
l[N+1] = zeros or context

# Sweep 1: Top-down liquid (using initial gas guess = g[0] everywhere)
for n in range(N, 0, -1):
    l[n] = l[n+1] + T_n(g[0] - E_n(l[n+1]))

# Sweep 2: Bottom-up gas (using computed liquid values)
for n in range(1, N+1):
    delta_n = T_n(g[n-1] - E_n(l[n]))
    g[n] = g[n-1] - delta_n

# Optional: Sweep 3+ for refinement
```

**Pros:** Differentiable. Reasonable approximation. Can add more sweeps for accuracy.  
**Cons:** Not a true simultaneous solution.

### 3.4 Learnable Parameters per Plate

Each plate $n$ has the following learnable parameters:

| Parameter | Shape | Description | ChemE Analog |
|---|---|---|---|
| $W_n^{eq}$ | $(d_g, d_l)$ | Equilibrium mapping weights | Shape of equilibrium curve $Y^* = f(X)$ |
| $\mathbf{b}_n^{eq}$ | $(d_g,)$ | Equilibrium mapping bias | Offset of equilibrium curve |
| $W_n^{tr}$ | $(d_g, d_g)$ | Transfer function weights | Mass transfer characteristics |
| $\mathbf{b}_n^{tr}$ | $(d_g,)$ | Transfer function bias | Transfer offset |
| $\alpha_n$ | $(1,)$ or $(d_g,)$ | Transfer coefficient | $K_{Ya}$ — mass transfer coefficient |

**Total parameters per plate:** $d_g \times d_l + d_g + d_g^2 + d_g + d_g = d_g(d_l + d_g + 2) + d_g$

**Total for N plates:** $N \times [d_g(d_l + d_g + 2) + d_g]$

### 3.5 Parameter Sharing Options

Analogous to different tower designs:

| Sharing Strategy | ChemE Analog | Description |
|---|---|---|
| No sharing | Different packing per section | Each plate has unique parameters |
| Full sharing | Uniform packing | All plates share parameters (like a weight-tied RNN) |
| Group sharing | Sectioned column | Groups of plates share parameters |

For the MVP, **full sharing** is recommended — it dramatically reduces parameters and is analogous to a uniformly packed tower.

---

## 4. Theoretical Properties

### 4.1 Conservation as Inductive Bias

The update rule $\mathbf{g}_n + \mathbf{l}_n = \mathbf{g}_{n-1} + \mathbf{l}_{n+1}$ (conservation) means:

$$\sum_{\text{all plates}} \boldsymbol{\Delta}_n = \mathbf{g}_0 - \mathbf{g}_N = \mathbf{l}_1 - \mathbf{l}_{N+1}$$

The total information "extracted" from the gas stream equals the total information "absorbed" by the liquid stream. This is a hard constraint, not a soft regularization.

**Implication:** The network cannot create or destroy information — only redistribute it between streams. This is a form of **information conservation** that standard networks don't have.

### 4.2 Relationship to the Operating Line

In absorption, the operating line relates gas and liquid compositions at any cross-section:

$$Y = \frac{L_s}{G_s} X + \left(Y_1 - \frac{L_s}{G_s} X_1\right)$$

In CFNN, the analogous relationship at plate $n$ is:

$$\mathbf{g}_n = \mathbf{g}_0 - \sum_{k=1}^{n} \boldsymbol{\Delta}_k = \mathbf{g}_0 - (\mathbf{l}_1 - \mathbf{l}_{n+1})$$

This means the gas and liquid states are linearly coupled at every plate — exactly the operating line.

### 4.3 Driving Force and Vanishing Gradients

In a real tower, the driving force $(Y - Y^*)$ decreases along the column but never reaches zero (in a properly designed tower). This means there is always a non-zero gradient for transfer.

In CFNN, the driving force $\boldsymbol{\delta}_n = \mathbf{g}_{n-1} - \mathcal{E}_n(\mathbf{l}_{n+1})$ plays a similar role. If the equilibrium function $\mathcal{E}$ is properly calibrated, the driving force maintains a non-zero signal through all plates.

**Hypothesis:** This natural maintenance of driving force may help CFNN avoid vanishing gradients in deep configurations, similar to how ResNets use skip connections but through a different mechanism.

### 4.4 Minimum Depth (Analogy to Minimum L/G Ratio)

In absorption, if $L/G < (L/G)_{min}$, you need infinite plates (the operating line touches the equilibrium curve → pinch point → zero driving force).

In CFNN, there is likely a minimum ratio of context dimension to gas dimension:

$$\frac{d_l}{d_g} > \left(\frac{d_l}{d_g}\right)_{min}$$

Below this ratio, the liquid stream doesn't have enough "capacity" to absorb the relevant information from the gas stream, regardless of depth.

### 4.5 Connection to Existing Work

| Existing Architecture | Shared Element | CFNN Difference |
|---|---|---|
| Deep Equilibrium Models (DEQ) | Fixed-point iteration | DEQ has one stream; CFNN has two coupled streams |
| Neural ODEs | Continuous dynamics | CFNN is discrete (plates) with two coupled ODEs |
| Reversible Networks (RevNets) | Information conservation | RevNets split one stream; CFNN has two streams with exchange |
| Ladder Networks | Top-down + bottom-up | Ladder uses additive noise; CFNN uses equilibrium-driven exchange |
| U-Net | Encoder-decoder | U-Net has sequential phases; CFNN has simultaneous counterflow |

---

## 5. Implementation Plan

### 5.1 Phase 1: Core Architecture (Days 1-2)

**Deliverable:** PyTorch implementation of CFNN with Approach C (unrolled bidirectional).

```python
# Pseudocode structure
class CounterFlowPlate(nn.Module):
    """Single exchange plate."""
    def __init__(self, d_gas, d_liquid):
        self.equilibrium = nn.Linear(d_liquid, d_gas)  # E: liquid → gas-equivalent
        self.transfer = nn.Sequential(
            nn.Linear(d_gas, d_gas),
            nn.Tanh()
        )
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)  # transfer coefficient
    
    def forward(self, g, l):
        e = torch.sigmoid(self.equilibrium(l))  # equilibrium mapping
        delta_raw = g - e                         # driving force
        delta = self.alpha * self.transfer(delta_raw)  # transfer amount
        g_new = g - delta   # gas loses
        l_new = l + delta   # liquid gains (conservation!)
        return g_new, l_new, delta

class CounterFlowNetwork(nn.Module):
    """Full countercurrent absorption network."""
    def __init__(self, d_in, d_gas, d_liquid, n_plates, d_out, n_sweeps=2):
        self.gas_encoder = nn.Linear(d_in, d_gas)
        self.plates = nn.ModuleList([
            CounterFlowPlate(d_gas, d_liquid) for _ in range(n_plates)
        ])
        self.output_head = nn.Linear(d_gas + d_liquid, d_out)
        self.n_sweeps = n_sweeps
        self.n_plates = n_plates
    
    def forward(self, x, context=None):
        g = [self.gas_encoder(x)]  # g[0]
        l = [None] * (self.n_plates + 2)
        l[self.n_plates + 1] = context if context is not None else torch.zeros(...)
        
        for sweep in range(self.n_sweeps):
            # Descending sweep (liquid, top → bottom)
            for n in range(self.n_plates, 0, -1):
                g_input = g[n-1] if len(g) > n-1 else g[0]
                _, l[n], _ = self.plates[n-1](g_input, l[n+1])
            
            # Ascending sweep (gas, bottom → top)
            g = [g[0]]  # reset gas, keep g[0]
            for n in range(1, self.n_plates + 1):
                g_new, _, _ = self.plates[n-1](g[n-1], l[n])
                g.append(g_new)
        
        # Output: concatenate cleaned gas + loaded liquid
        output_features = torch.cat([g[-1], l[1]], dim=-1)
        return self.output_head(output_features)
```

### 5.2 Phase 2: Benchmarking (Days 3-4)

Compare CFNN against baselines on standard tasks:

**Datasets:**
- **Tabular:** UCI datasets (classification + regression)
- **Sequence:** Simple time series (to test if counterflow helps with temporal patterns)
- **Image:** MNIST/FashionMNIST (flattened, to test on high-dimensional input)

**Baselines:**
- MLP (same total parameters)
- ResNet-style MLP (skip connections)
- Bidirectional processing (two independent streams concatenated)

**Metrics:**
- Accuracy / MSE
- Training convergence speed (epochs to target performance)
- Gradient flow analysis (gradient norms per plate)
- Representation quality (t-SNE of gas vs liquid streams)

### 5.3 Phase 3: Analysis & Visualization (Day 5)

**Visualizations inspired by ChemE:**
- **McCabe-Thiele analog:** Plot gas state vs liquid state at each plate, overlaid with learned equilibrium curve
- **Concentration profile:** Plot feature norms along the tower (analogous to Y vs Z)
- **Driving force profile:** Plot $\|\boldsymbol{\delta}_n\|$ at each plate
- **Transfer profile:** Plot $\|\boldsymbol{\Delta}_n\|$ at each plate — where does the network "absorb" the most?

### 5.4 Phase 4: HuggingFace Deployment (Days 6-7)

- Publish the model architecture as a HF model/repo
- Create a Gradio Space with:
  - Interactive demo: pick a dataset, train CFNN vs baselines
  - McCabe-Thiele visualization: see how the network learns
  - Architecture explorer: adjust plates, dimensions, see the effect
- Write a HF blog post explaining the ChemE → ML connection

---

## 6. Research Questions

### Primary

1. **Does counterflow exchange improve representation learning** compared to single-stream architectures with the same parameter count?

2. **Does conservation (Δg = -Δl) act as beneficial regularization**, reducing overfitting compared to unconstrained architectures?

3. **Does the driving force mechanism help with gradient flow** in deeper networks?

### Secondary

4. What is the optimal number of sweeps (iterations) for convergence?
5. How does the ratio $d_l / d_g$ affect performance (analogy to L/G ratio)?
6. Is shared-plate (weight-tied) CFNN competitive with unique-plate CFNN?
7. Can we visualize the learned equilibrium curves and interpret them?

### Stretch

8. Can CFNN be extended to multi-component absorption (multiple gas-liquid pairs)?
9. Is there a continuous-depth version (Neural ODE analog)?
10. Does CFNN have advantages for specific task types (e.g., feature selection, denoising)?

---

## 7. Extended Architecture: Distillation Mode (CFNN-D)

### 7.1 Why Distillation is More General Than Absorption

In absorption, mass transfer is **unidirectional** — solute always moves from gas to liquid. In distillation, transfer is **bidirectional**:

- **Rectifying section** (above feed): Vapor rises, liquid (reflux) descends. Heavy component transfers from vapor → liquid. Analogous to absorption.
- **Stripping section** (below feed): Liquid descends, vapor (boilup) rises. Light component transfers from liquid → vapor. This is the **reverse** — desorption.

In each plate, BOTH directions of transfer happen simultaneously for different components. The net direction depends on the local driving force.

### 7.2 McCabe-Thiele: Two Operating Lines

In absorption, there is one operating line:

$$Y = \frac{L_s}{G_s} X + \left(Y_1 - \frac{L_s}{G_s} X_1\right)$$

In distillation, there are TWO:

**Rectifying section** (above feed plate $f$):

$$y_{n+1} = \frac{R}{R+1} x_n + \frac{x_D}{R+1}$$

Where $R$ = reflux ratio, $x_D$ = distillate composition.

**Stripping section** (below feed plate $f$):

$$y_{m+1} = \frac{\bar{L}}{\bar{V}} x_m - \frac{B \cdot x_B}{\bar{V}}$$

Where $\bar{L}, \bar{V}$ = liquid and vapor flows below feed, $B$ = bottoms flow, $x_B$ = bottoms composition.

The two lines intersect at the **feed plate** — where fresh feed enters the column.

### 7.3 CFNN-D Architecture

The distillation variant introduces three key changes:

#### Change 1: Bidirectional Transfer

In CFNN (absorption), transfer is always gas → liquid:

```python
# CFNN (absorption): Δ always extracts from gas
delta = alpha * tanh(transfer(driving_force))  # bounded [−1, 1] but effectively one-way
g_new = g - delta  # gas always loses
l_new = l + delta  # liquid always gains
```

In CFNN-D (distillation), transfer direction is **learned per plate**:

```python
# CFNN-D (distillation): Δ can go either direction
delta = alpha * transfer(driving_force)  # NO tanh — unbounded direction
# If delta > 0: gas→liquid (rectifying behavior)
# If delta < 0: liquid→gas (stripping behavior)
# Conservation still holds: Δg = -Δl
```

The network learns which plates should rectify and which should strip.

#### Change 2: Feed Plate

A special plate $f$ where external information is injected:

$$\mathbf{g}_f = \mathbf{g}_{f-1} - \boldsymbol{\Delta}_f + \phi \cdot \mathbf{z}_{feed}$$

$$\mathbf{l}_f = \mathbf{l}_{f+1} + \boldsymbol{\Delta}_f + (1-\phi) \cdot \mathbf{z}_{feed}$$

Where:
- $\mathbf{z}_{feed}$ = encoded feed information (could be auxiliary features, context, etc.)
- $\phi \in [0, 1]$ = feed quality parameter (learned). Analogous to:
  - $\phi = 1$: saturated vapor feed (all goes to gas stream)
  - $\phi = 0$: saturated liquid feed (all goes to liquid stream)
  - $0 < \phi < 1$: partial vaporization (split between both streams)

#### Change 3: Two Operating Regimes

The plates above the feed operate with one set of flow ratios (rectifying), and plates below operate with another (stripping). This translates to:

```python
class DistillationNetwork(nn.Module):
    def __init__(self, d_in, d_gas, d_liquid, n_plates, feed_plate, d_out):
        self.rectifying_plates = nn.ModuleList([
            CounterFlowPlate(d_gas, d_liquid, bidirectional=True) 
            for _ in range(feed_plate, n_plates)
        ])
        self.stripping_plates = nn.ModuleList([
            CounterFlowPlate(d_gas, d_liquid, bidirectional=True) 
            for _ in range(feed_plate)
        ])
        self.feed_encoder = nn.Linear(d_feed, d_gas)
        self.feed_quality = nn.Parameter(torch.tensor(0.5))  # learnable φ
```

### 7.4 Distillation Products as Network Outputs

A distillation column produces two products:
- **Distillate** (top): enriched in light component → $\mathbf{g}_N$ (cleaned gas exiting top)
- **Bottoms** (bottom): enriched in heavy component → $\mathbf{l}_1$ (loaded liquid exiting bottom)

In CFNN-D, both outputs are meaningful:

$$\hat{\mathbf{y}}_{primary} = W_D \cdot \mathbf{g}_N + \mathbf{b}_D \quad \text{(distillate — refined features)}$$

$$\hat{\mathbf{y}}_{secondary} = W_B \cdot \mathbf{l}_1 + \mathbf{b}_B \quad \text{(bottoms — extracted representation)}$$

This naturally supports **multi-task learning** — one network producing two different outputs from its two product streams, exactly like a distillation column producing distillate and bottoms.

### 7.5 Reflux and Boilup as Regularization

In distillation:
- **Reflux**: Part of the distillate is returned to the top of the column as liquid
- **Boilup**: Part of the bottoms is re-vaporized and returned as vapor

These recycle streams improve separation but cost energy. In CFNN-D:

```python
# Reflux: part of final gas output feeds back into liquid input
l[N+1] = reflux_ratio * g[N]  # instead of zeros

# Boilup: part of final liquid output feeds back into gas input  
g[0] = gas_encoder(x) + boilup_ratio * l[1]
```

**Reflux ratio** $R$ becomes a **regularization hyperparameter**:
- $R = 0$: No reflux. Minimum computation but worst separation.
- $R → ∞$: Total reflux. Best separation but no net output (all processing, no production).
- Optimal $R$: Balance between representation quality and computational cost.

This gives a principled, physics-motivated way to control the tradeoff between model capacity and efficiency.

---

## 8. Reactor-Inspired Building Blocks

### 8.1 CSTRs in Series = Network Layers

The classic CSTRs-in-series model from Berkeley Madonna:

**Single CSTR at steady state:**

$$0 = F(C_{in} - C_{out}) - V \cdot r(C_{out})$$

$$\tau = V/F \quad \text{(residence time)}$$

$$C_{out} = \frac{C_{in}}{1 + \tau \cdot k} \quad \text{(for first-order reaction } r = kC\text{)}$$

**N CSTRs in series:**

$$C_n = \frac{C_{n-1}}{1 + \tau_n \cdot k_n}$$

This is exactly a neural network layer with a specific activation function:

| CSTR Element | Neural Network Equivalent |
|---|---|
| $C_{n-1}$ (inlet concentration) | Layer input $\mathbf{h}_{n-1}$ |
| $C_n$ (outlet concentration) | Layer output $\mathbf{h}_n$ |
| $\tau_n \cdot k_n$ (Damköhler number) | Learnable weight scaling |
| $r(C)$ (reaction rate) | Activation function |
| $V$ (tank volume) | Layer width (capacity) |
| $F$ (flow rate) | Learning rate / throughput |
| $\tau = V/F$ (residence time) | How long information "stays" in the layer |

**The CSTR activation function:**

For a first-order reaction: $h_n = \frac{h_{n-1}}{1 + w_n}$ — this is a learnable damping.

For Michaelis-Menten kinetics: $h_n = h_{n-1} - \frac{w_n \cdot h_{n-1}}{K_m + h_{n-1}}$ — this is a soft thresholding (like a learnable ReLU with saturation).

For autocatalytic reaction: $h_n = h_{n-1} + w_n \cdot h_{n-1}(1 - h_{n-1}/C_{max})$ — this is logistic growth, naturally bounded.

### 8.2 Dynamic CSTR = Recurrent Layer

The transient CSTR (what you solved in Berkeley Madonna):

$$\frac{dC}{dt} = \frac{1}{\tau}(C_{in} - C) - r(C)$$

This is an ODE that you integrate over time. Discretized with Euler:

$$C^{t+1} = C^t + \Delta t \left[ \frac{1}{\tau}(C_{in}^t - C^t) - r(C^t) \right]$$

This is structurally identical to a **GRU/LSTM update**:

$$\mathbf{h}^{t+1} = \mathbf{h}^t + \Delta t \left[ \frac{1}{\tau}(\mathbf{x}^t - \mathbf{h}^t) - \sigma(W\mathbf{h}^t) \right]$$

The $\frac{1}{\tau}(\mathbf{x}^t - \mathbf{h}^t)$ term is the "input gate" — new information diluting old state. The $r(\mathbf{h}^t)$ term is the "forget gate" — information being "consumed" by the reaction.

### 8.3 Parallel Reactors = Parallel Feature Processing

Reactors in parallel (which you also modeled):

```
           ┌→ Reactor A (r_A, V_A) →┐
Input F →  ├→ Reactor B (r_B, V_B) →├→ Mixed Output
           └→ Reactor C (r_C, V_C) →┘
```

The flow splits: $F = F_A + F_B + F_C$

Each reactor processes the same input but with different kinetics (different $r$, different $V$). The outputs are mixed.

This is exactly a **multi-head** operation:

```python
# Parallel reactors = multi-head processing
heads = []
for reactor in parallel_reactors:
    heads.append(reactor(input * split_ratio))
output = mix(heads)  # could be concat, sum, or learned mixing
```

In transformers, multi-head attention does this — multiple parallel "reactors" processing the same input with different parameters. But the CSTR analogy gives a physical reason for WHY parallel processing works: different reactors have different selectivities (they extract different "products" from the same feed).

### 8.4 The Complete Reactor Network Vocabulary

Combining all elements:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CFNN-R: Reactor Network                       │
│                                                                  │
│  ┌─────┐   ┌──────────────────────┐   ┌─────┐                  │
│  │CSTR │──→│ Absorption Tower     │──→│CSTR │──→ Output         │
│  │(pre)│   │ (counterflow plates) │   │(post│                   │
│  └─────┘   └──────────────────────┘   └─────┘                  │
│     ↑              ↑↓                    │                      │
│     │         ┌─────────┐                │                      │
│     │         │Parallel │                │                      │
│     │         │CSTRs    │                │                      │
│     │         │(heads)  │                │                      │
│     │         └─────────┘                │                      │
│     └────────── recycle ←────────────────┘                      │
│                                                                  │
│  ■ CSTRs in series    = feedforward layers                      │
│  ■ Absorption tower   = counterflow exchange (CFNN core)        │
│  ■ Parallel CSTRs     = multi-head processing                  │
│  ■ Recycle stream     = skip connections / feedback             │
│  ■ Feed injection     = auxiliary input / conditioning          │
│  ■ Reaction kinetics  = activation functions                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Unified Framework: ChemNet

### 9.1 A Modular Architecture from Chemical Engineering Primitives

The key insight is that **all common neural network components have chemical engineering analogs**, and some chemical engineering configurations suggest **novel neural components** that don't exist yet:

| ChemE Primitive | Neural Analog | Status |
|---|---|---|
| CSTR | Dense layer with specific activation | Exists (standard NN) |
| CSTRs in series | Deep feedforward network | Exists (MLP) |
| PFR | Neural ODE (continuous-depth) | Exists (Chen 2018) |
| Parallel reactors | Multi-head processing | Exists (Transformers) |
| Recycle stream | Skip connections | Exists (ResNet) |
| Absorption tower | **Counterflow exchange (CFNN)** | **NEW — this work** |
| Distillation column | **Bidirectional counterflow + feed (CFNN-D)** | **NEW — this work** |
| Reflux / boilup | **Physics-motivated regularization** | **NEW — this work** |
| Feed plate | **Auxiliary information injection** | **NEW — this work** |
| Reaction kinetics → activation | **ChemE-derived activation functions** | **NEW — this work** |
| Damköhler number | Depth-to-width scaling law | **NEW interpretation** |

### 9.2 Novel Activation Functions from Reaction Kinetics

Standard activations (ReLU, sigmoid, tanh) were chosen empirically. Chemical kinetics suggests physically-motivated alternatives:

#### Michaelis-Menten Activation

$$f(x) = \frac{V_{max} \cdot x}{K_m + |x|} \quad \text{where } V_{max}, K_m > 0 \text{ are learnable}$$

Properties: Smooth, bounded, saturating. Unlike sigmoid, it passes through origin. The saturation level ($V_{max}$) and transition sharpness ($K_m$) are independently learnable.

#### Arrhenius Activation

$$f(x) = x \cdot \exp\left(-\frac{E_a}{x + \epsilon}\right) \quad \text{where } E_a > 0 \text{ is learnable}$$

Properties: Near-zero for small inputs (below "activation energy"), then rapidly increasing. Natural thresholding behavior — signals must exceed a minimum "energy" to propagate. This is like a learnable, smooth ReLU with a physics-motivated shape.

#### Autocatalytic Activation

$$f(x) = k \cdot x \cdot (1 - x/C_{max}) \quad \text{where } k, C_{max} > 0$$

Properties: Logistic growth. Self-amplifying for small signals, self-limiting for large signals. Naturally bounded. Useful for features that should exhibit "critical mass" behavior.

#### Hill Equation Activation (Cooperative Binding)

$$f(x) = \frac{x^n}{K^n + x^n} \quad \text{where } n, K > 0 \text{ are learnable}$$

Properties: Sigmoidal but with learnable steepness ($n$). When $n=1$, reduces to Michaelis-Menten. When $n→∞$, approaches step function. Naturally models cooperative/threshold effects.

### 9.3 The Damköhler Number as a Design Principle

In reactor design, the Damköhler number relates reaction rate to flow rate:

$$Da = \frac{\text{reaction rate}}{\text{flow rate}} = \frac{k \cdot \tau \cdot C^{n-1}}{1}$$

- $Da \ll 1$: Flow dominates. Reactant passes through without reacting. → **Underfitting**: information passes through layers without being transformed.
- $Da \gg 1$: Reaction dominates. Complete conversion in each reactor. → **Overfitting**: each layer transforms too aggressively.
- $Da \approx 1$: Optimal balance. → **Good training regime**.

For CFNN, the analog is:

$$Da_{CFNN} = \frac{\alpha \cdot \|W^{tr}\|}{\text{throughput}} \approx \frac{\text{transfer rate}}{\text{information flow rate}}$$

This could serve as a **diagnostic tool**: monitor $Da$ during training to detect if layers are under- or over-processing.

---

## 10. Experimental Plan (Updated)

### 10.1 Architecture Variants to Test

| Variant | Description | Plates | Bidirectional | Feed | Reflux |
|---|---|---|---|---|---|
| CFNN-A | Absorption (original) | N | No | No | No |
| CFNN-D | Distillation | N | Yes | Yes | Optional |
| CFNN-R | Reactor cascade + counterflow | N+M | Yes | Yes | Yes |

### 10.2 Baselines

| Baseline | Parameter-Matched | Description |
|---|---|---|
| MLP | Yes | Standard feedforward, same total params |
| ResNet-MLP | Yes | MLP with skip connections |
| Bi-MLP | Yes | Two independent streams, concatenated |
| GRU | Yes | Recurrent baseline (for sequence tasks) |

### 10.3 Datasets (Progressive Difficulty)

**Tier 1 — Proof of concept:**
- Synthetic: XOR, spiral classification, concentric circles
- Purpose: Can CFNN learn at all? Debug the architecture.

**Tier 2 — Tabular benchmarks:**
- UCI: Iris, Wine, Breast Cancer, California Housing
- Purpose: Compare with MLP baselines on standard tasks.

**Tier 3 — Structured data:**
- MNIST (flattened to 784-dim vector)
- FashionMNIST
- Purpose: Higher-dimensional input, harder task.

**Tier 4 — Where CFNN might shine:**
- Noisy feature selection tasks (can the counterflow "absorb" relevant features and leave noise?)
- Multi-task learning (distillate = task A, bottoms = task B)
- Denoising autoencoders (counterflow as natural denoiser)

### 10.4 Analyses

1. **McCabe-Thiele Neural Plot**: Plot $\|\mathbf{g}_n\|$ vs $\|\mathbf{l}_n\|$ at each plate. Overlay the learned equilibrium curve $\mathcal{E}$. Do we see the characteristic staircase pattern?

2. **Concentration Profile**: Plot feature norms $\|\mathbf{g}_n\|$ and $\|\mathbf{l}_n\|$ vs plate number $n$. Does the gas get "cleaner" and the liquid get "richer" as expected?

3. **Driving Force Analysis**: Plot $\|\boldsymbol{\delta}_n\|$ vs $n$. Is there a pinch point? Does driving force vanish?

4. **Transfer Direction Map** (CFNN-D only): For each plate, color by sign of mean $\Delta$. Do we see a natural split into rectifying and stripping sections?

5. **Damköhler Diagnostic**: Track $Da_{CFNN}$ during training. Does it converge to ~1?

6. **Gradient Flow**: Compare gradient norms per plate in CFNN vs per layer in MLP. Does counterflow help?

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Architecture doesn't outperform baselines | Medium | High | Focus on interpretability and novelty even if performance is comparable |
| Iterative solving is too slow to train | Medium | Medium | Start with Approach C (unrolled), only try Approach A if needed |
| Conservation constraint is too restrictive | Low | High | Add a "leakage" parameter that relaxes strict conservation |
| Vanishing driving force (analog to pinch point) | Medium | Medium | Monitor driving force during training, add regularization if needed |
| Bidirectional transfer in CFNN-D causes instability | Medium | Medium | Start with CFNN-A, graduate to CFNN-D only after A works |
| Scope creep | High | Medium | Strict MVP: CFNN-A on Tier 1-2 datasets first |

---

## 12. Research Questions

### Primary

1. **Does counterflow exchange improve representation learning** compared to single-stream architectures with the same parameter count?

2. **Does conservation ($\Delta g = -\Delta l$) act as beneficial regularization**, reducing overfitting compared to unconstrained architectures?

3. **Does the driving force mechanism help with gradient flow** in deeper networks?

### Secondary

4. Does bidirectional transfer (CFNN-D) outperform unidirectional (CFNN-A)?
5. What is the optimal number of sweeps for convergence?
6. How does the ratio $d_l / d_g$ affect performance (analogy to L/G ratio)?
7. Is shared-plate (weight-tied) CFNN competitive with unique-plate?
8. Do ChemE-derived activations (Michaelis-Menten, Arrhenius) outperform standard ones?
9. Does the Damköhler diagnostic correlate with training health?

### Stretch

10. Can CFNN handle multi-component separation (multiple gas-liquid pairs)?
11. Is there a continuous-depth version (Neural ODE counterflow)?
12. Does reflux ratio act as effective regularization?
13. Can CFNN-D naturally learn to do multi-task via distillate/bottoms split?

---

## 13. Notation Summary

| Symbol | Domain | Description |
|---|---|---|
| $\mathbf{x}$ | $\mathbb{R}^{d_{in}}$ | Input data |
| $\mathbf{g}_n$ | $\mathbb{R}^{d_g}$ | Gas stream state at plate $n$ |
| $\mathbf{l}_n$ | $\mathbb{R}^{d_l}$ | Liquid stream state at plate $n$ |
| $\mathcal{E}_n$ | $\mathbb{R}^{d_l} \to \mathbb{R}^{d_g}$ | Equilibrium function at plate $n$ |
| $\mathcal{T}_n$ | $\mathbb{R}^{d_g} \to \mathbb{R}^{d_g}$ | Transfer function at plate $n$ |
| $\boldsymbol{\delta}_n$ | $\mathbb{R}^{d_g}$ | Driving force at plate $n$ |
| $\boldsymbol{\Delta}_n$ | $\mathbb{R}^{d_g}$ | Transfer amount at plate $n$ |
| $\alpha_n$ | $\mathbb{R}^+$ | Transfer coefficient at plate $n$ |
| $N$ | $\mathbb{Z}^+$ | Number of plates (network depth) |
| $K$ | $\mathbb{Z}^+$ | Number of convergence sweeps |
| $d_g$ | $\mathbb{Z}^+$ | Gas stream dimension |
| $d_l$ | $\mathbb{Z}^+$ | Liquid stream dimension |
| $R$ | $\mathbb{R}^+$ | Reflux ratio (CFNN-D) |
| $\phi$ | $[0,1]$ | Feed quality parameter (CFNN-D) |
| $f$ | $\mathbb{Z}^+$ | Feed plate location (CFNN-D) |
| $Da$ | $\mathbb{R}^+$ | Damköhler number (diagnostic) |

---

## 14. References

### Chemical Engineering — Mass Transfer
- Treybal, R.E. "Mass Transfer Operations," 3rd Ed., McGraw-Hill, 1980
- McCabe, W.L., Smith, J.C. "Unit Operations of Chemical Engineering"
- Bird, Stewart, Lightfoot. "Transport Phenomena," 2nd Ed.

### Chemical Engineering — Reactor Design
- Fogler, H.S. "Elements of Chemical Reaction Engineering," 6th Ed.
- Levenspiel, O. "Chemical Reaction Engineering," 3rd Ed.
- Berkeley Madonna — ODE solver for reactor modeling (www.berkeleymadonna.com)

### Neural Architecture Inspiration
- Bai, S., Kolter, J.Z., Koltun, V. "Deep Equilibrium Models," NeurIPS 2019
- Chen, R.T.Q. et al. "Neural Ordinary Differential Equations," NeurIPS 2018
- Gomez, A.N. et al. "The Reversible Residual Network," NeurIPS 2017
- Rasmus, A. et al. "Semi-supervised Learning with Ladder Networks," NeurIPS 2015
- He, K. et al. "Deep Residual Learning for Image Recognition," CVPR 2016

### Physics-Informed Neural Networks
- Raissi, M., Perdikaris, P., Karniadakis, G.E. "Physics-informed neural networks," JCP 2019
- PANACHE framework for adsorption simulation (U. Alberta, 2022)
- Purdue OINN project: Optimization-Inspired Neural Networks (NSF 2025)
- Koksal & Aydin (2023) — Physics-informed ANNs for distillation columns

### Market Diffusion (Analogy Validation)
- Bass, F.M. "A New Product Growth Model for Consumer Durables," Management Science 1969
- Kikuchi (2024) "Dual-Channel Technology Diffusion: Spatial Decay and Network Contagion"

---

## 15. Project Structure

```
counterflow-nn/
├── README.md                      # Overview and quick start
├── docs/
│   └── CFNN_Technical_Doc.md      # This document
├── src/
│   ├── __init__.py
│   ├── plates.py                  # CounterFlowPlate, DistillationPlate
│   ├── network.py                 # CFNN_A, CFNN_D, ReactorCascade
│   ├── activations.py             # Michaelis-Menten, Arrhenius, Hill activations
│   ├── diagnostics.py             # Damköhler number, driving force tracking
│   ├── visualization.py           # McCabe-Thiele plots, concentration profiles
│   └── utils.py                   # Training utilities
├── experiments/
│   ├── tier1_synthetic.py         # XOR, spirals, concentric circles
│   ├── tier2_tabular.py           # UCI dataset experiments
│   ├── tier3_images.py            # MNIST / FashionMNIST
│   ├── tier4_denoising.py         # Where CFNN might shine
│   └── compare_baselines.py       # Head-to-head comparisons
├── notebooks/
│   ├── 01_concept_and_theory.ipynb     # Visual explanation of ChemE → NN mapping
│   ├── 02_cfnn_a_absorption.ipynb      # Training CFNN-A
│   ├── 03_cfnn_d_distillation.ipynb    # Training CFNN-D
│   ├── 04_reactor_activations.ipynb    # Testing ChemE activations
│   ├── 05_mccabe_thiele_neural.ipynb   # Visualization and analysis
│   └── 06_full_comparison.ipynb        # All variants vs baselines
├── app.py                         # Gradio Space for HuggingFace
├── requirements.txt
└── LICENSE
```

