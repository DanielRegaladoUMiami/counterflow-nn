# Changelog

All notable changes to CounterFlow NN will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`AbsorptionTower`** (`src/absorption_tower.py`): a physically-exact
  countercurrent absorber as a differentiable PyTorch layer. Closed-form
  Kremser solve with learnable Henry's constant, `L/G` ratio, Murphree
  plate efficiency, and equilibrium intercept per feature channel.
  `O(1)` in the stage count `N`, with a numerically stable branch for
  the `β → 1` pinch case.
- **`AbsorptionNetwork`**: thin ML wrapper around `AbsorptionTower`
  with sigmoid-bounded encoders and an MLP head.
- **Physical validation** against textbook examples
  (`experiments/tier0_physical_validation.py`): Treybal Ex. 8.2 (acetone
  absorber), Seader Ex. 6.1 (n-butane absorber), `A = 1` pinch case,
  real-tray Murphree `E = 0.70`. All four match reference values to
  better than `10⁻³` relative error.
- **Test suite** (`tests/test_absorption_tower.py`): 30 tests covering
  Kremser correctness, mass balance, operating line, Murphree relation,
  limiting cases, gradient flow, and end-to-end training recovery.
- **ML benchmark** (`experiments/tier1_absorption_benchmark.py`):
  AbsorptionNetwork vs MLP on moons and an inverse-Kremser regression
  task, with matched parameter budgets.
- **Interactive Gradio app** (`app.py`): three-tab editorial UI with a
  live McCabe–Thiele diagram, parameter-recovery training demo, and
  physics recap. Every figure renders from the same module used in
  training.
- **Documentation** (`docs/AbsorptionTower.md`): full derivation,
  parameter encoding, Python API, and validation record.

### Changed
- README: added `AbsorptionTower` to the variants table, expanded the
  project structure, added a usage block for the exact tower, and
  referenced `tier0` validation in the Quick Start.

## [0.1.0] — Phase 2 (baseline)

Initial release with learnable CFNN-A (`CounterFlowNetwork`) and CFNN-D
(`DistillationNetwork`) variants, synthetic benchmarks, and the Phase 2
notebooks.
