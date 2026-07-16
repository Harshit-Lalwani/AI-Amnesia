# Healing AI Amnesia in Physics-Informed Neural Networks

This workspace contains a Fisher-KPP Physics-Informed Neural Network (PINN) project that replicates and
extends [Aberqi & Miloudi (arXiv:2601.11406v1)](2601.11406v1.pdf) to test a simple question: does
preserving Adam optimizer state across fine-tuning phases improve convergence and final accuracy,
compared to the paper's own protocol of resetting the optimizer at each retraining phase?

**Full write-up: [docs/Report.pdf](docs/Report.pdf).**

**Headline results:**
- **1D Fisher-KPP (replication + validated fix):** with the optimizer reset every phase (the paper's
  protocol), relative L2 error *worsens* with each retraining pass, 5.01e-2 → 7.69e-2 → 8.70e-2.
  Preserving Adam's state across phases instead (Phase 3 fine-tuned with L-BFGS) *improves* it every
  phase, 5.82e-2 → 4.55e-2 → 4.56e-2 — a **47.6% reduction in final error**, flipping retraining from
  harmful to helpful. The paper only recommends this fix (Section 5.2) without implementing or
  quantifying it; this project does both.
- **2D Fisher-KPP (original extension, not covered by the paper):** a 32-configuration architecture ×
  schedule sweep found a 7×50 tanh network converges reliably (L2 = 0.023 ± 0.006, ~2.31 min), while a
  wider 6×100 network diverges on every configuration tested (L2 ≈ 1.0) and is still ~83% slower.

The main implementation is the 1D Fisher-KPP experiment in [pinn.py](pinn.py). It generates synthetic training data, trains a PINN against the PDE residual plus initial and boundary conditions, compares the result with an exact traveling-wave solution and an explicit finite-difference baseline, and saves plots and checkpoints for later analysis.

## What is in this workspace

### Main 1D Fisher-KPP project without preserving Adam state

- [pinn.py](pinn.py): the most complete end-to-end script for the 1D Fisher-KPP experiment. It generates data, trains the PINN in three phases, benchmarks against FDM, and exports plots/results.
- [pinn_early.py](pinn_early.py): an earlier version of the same experiment, kept as a simpler or legacy reference.
- [pinn_early.ipynb](pinn_early.ipynb): notebook form of the early implementation.
- [exact_solution.py](exact_solution.py): computes the analytical traveling-wave reference solution.
- [fdm_solver.py](fdm_solver.py): explicit 1D finite-difference baseline solver.
- [data_samples.npz](data_samples.npz): generated 1D training samples.
- [fdm_traveling_wave.npz](fdm_traveling_wave.npz): saved 1D FDM output.
- [pinn_training_history.png](pinn_training_history.png): training-history figure from the 1D run.

### 2D experiment track

- [2d_pinn/dataset_generation_2d.py](2d_pinn/dataset_generation_2d.py): generates 2D training data.
- [fdm_solver_2d.py](fdm_solver_2d.py): explicit 2D finite-difference solver.
- [2d_pinn/2d_pinn.ipynb](2d_pinn/2d_pinn.ipynb): 2D PINN notebook.
- [data_samples_2d.npz](data_samples_2d.npz): generated 2D dataset.
- [fdm_solution_2d.npz](fdm_solution_2d.npz): saved 2D FDM output.
- [2d_pinn/l2_2d_tanh__7x50__cosine_exponential_lbfgs.png](2d_pinn/l2_2d_tanh__7x50__cosine_exponential_lbfgs.png): example 2D training result plot.

### Memory aware PINN preserving Adam state

- [memoryAwarePINN/modiified_pinn.ipynb](memoryAwarePINN/modiified_pinn.ipynb): an alternate PINN notebook experiment.
- [memoryAwarePINN/pinn_results_single.npz](memoryAwarePINN/pinn_results_single.npz): single-run result artifact.
- [memoryAwarePINN/pinn_results_lbfgs.npz](memoryAwarePINN/pinn_results_lbfgs.npz): LBFGS result artifact.
- [memoryAwarePINN/training_history_single_run.png](memoryAwarePINN/training_history_single_run.png): training-history plot.
- [memoryAwarePINN/sweep_results_32.csv](memoryAwarePINN/sweep_results_32.csv): sweep output data.

## Project idea

The core experiment studies a PINN for the 1D Fisher-KPP equation,

$$
u_t = D u_{xx} + R u (1-u),
$$

with a traveling-wave analytical solution used as reference. The model is trained on three types of synthetic points:

- collocation points in the interior domain,
- initial-condition points at $t=0$,
- boundary-condition points at $x=0$ and $x=1$.

The main hypothesis is that reloading Adam’s internal moments during retraining should preserve optimization progress better than restarting Adam from scratch.

## Main workflow

1. Generate synthetic training data.
2. Train a 7-layer, 50-neuron tanh PINN on the 1D Fisher-KPP problem.
3. Compare the learned solution against the exact traveling-wave solution.
4. Benchmark with an explicit finite-difference method.
5. Save the trained model, loss history, and plots.

The main 1D script runs this workflow in three phases:

- Phase 1: initial training.
- Phase 2: retraining with a fresh optimizer state.
- Phase 3: additional retraining and reporting.