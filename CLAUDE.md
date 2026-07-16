# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Healing AI Amnesia in Physics-Informed Neural Networks" — a fix for a PINN retraining bug: resetting
the Adam optimizer's internal state at each retraining phase boundary (the common default, and the
protocol used by Aberqi & Miloudi, arXiv:2601.11406v1, present in this repo as `2601.11406v1.pdf`)
silently degrades accuracy with every retraining pass. Preserving optimizer state across phases instead
fixes this — see `README.md` / `docs/Report.pdf` for the quantified result (a 47.6% final-error
reduction on the paper's 1D Fisher-KPP problem, `u_t = D u_xx + R u(1-u)`, plus an original 2D
extension). There are two experiment tracks (1D and 2D Fisher-KPP), each run in both a "reset optimizer"
and a "preserve optimizer state" variant.

## Running the code

There is no build system, package manifest, or test suite — this is a numerics research project of
standalone scripts and Jupyter notebooks. Dependencies (verify with `pip show`) are `torch`, `numpy`,
`scipy`, `matplotlib`, `pandas`, `seaborn`. Scripts use `matplotlib.use('Agg')` and can run headless.

- `python pinn.py` — the primary, most complete 1D experiment. Self-contained: on import it generates
  `data_samples.npz`, trains in three phases, benchmarks against FDM, prints a paper-comparison summary
  table, and writes `results/1d/pinn_training_history.png` / `pinn_results.npz` / `pinn_initial.pth` /
  `pinn_final.pth`. Takes several minutes (50k+ Adam iterations total). There is no CLI/argparse — change
  behavior by editing constants/calls at module level (e.g. the `iterations=`, `learning_rate=` args to
  the three `pinn.train_pinn(...)` calls near the bottom of the file).
- `python pinn_early.py` — earlier/legacy version of the same 1D experiment, wrapped in `main()`; kept as
  a simpler reference, not actively developed. Writes `results/1d/pinn_training_history.png`.
- `python exact_solution.py` — analytical traveling-wave reference solution (`ExactSolution` class); has
  its own `main()` for standalone sanity-checking/plotting.
- `python 2d_pinn/dataset_generation_2d.py` then the 2D notebooks (`2d_pinn/2d_pinn.ipynb`,
  `2d_pinn/2d_pinn_without_memory.ipynb`) — the 2D track. Notebooks are the primary artifact here (not
  scripts); run them cell-by-cell in Jupyter. They write checkpoints to `checkpoints/2d/` and results
  (single-run plots/npz, per-sweep-configuration plots) to `results/2d/`.
- `python fdm_solver_2d.py` — explicit finite-difference baseline for the 2D problem; writes
  `results/2d/fdm_solution_2d.npz`.
- `memoryAwarePINN/modiified_pinn.ipynb` — the "preserves Adam state across phases" variant of the 1D
  experiment, compared against `pinn_early.ipynb` (which matches the paper's reset-each-phase protocol).
  Writes `results/1d/training_history_lbfgs.png`.
- `generate_2d_plots.py`, `generate_comparison_plots.py` — post-hoc plotting scripts that read sweep CSVs
  (`results/2d/sweep/sweep_results_2d_32.csv`, `memoryAwarePINN/sweep_results_32.csv`) and single-run
  `.npz` results (`results/2d/pinn_results_single_2d.npz`, `results/2d/fdm_solution_2d.npz`) to produce
  publication figures into `figures_2d/` and `figures_comparison/`. Paths are resolved relative to
  `Path(__file__).parent`, so these run correctly regardless of the working directory.
- `python generate_summary_figures.py` — builds the two headline bar charts embedded in `README.md`
  (`figures_summary/`) from the already-verified result numbers; does not read any data files, only
  hardcoded numbers transcribed from the sweep CSVs and notebook outputs.

## Architecture

### The PINN model (`PINN_FisherKPP` in `pinn.py`, mirrored with variations in `pinn_early.py` and the
notebooks)

- Fully-connected `tanh` network, default 7 hidden layers × 50 neurons, Xavier-normal weight init /
  zero bias init, input `(x, t)` (2 units) → output `u(x,t)` (1 unit). `forward(x, t)` concatenates in
  that order — training-data tensors must be loaded as `(x, t)`, not `(t, x)`, to match.
- Composite loss = `lambda_ic * L_IC + lambda_bc * L_BC + lambda_res * L_Res` (paper Eq. 1), computed
  over three disjoint point sets: collocation (interior, PDE residual via `torch.autograd.grad`),
  initial (`t=0`), and boundary (`x=0`/`x=1`, or the 4 edges in 2D).
  - `L_Res` uses second-order autograd through `pde_residual()` — `requires_grad_(True)` is set once on
    the collocation tensors before the training loop, not inside the residual method itself.
  - Adaptive IC/BC weights (I-PINN scheme, paper Fig. 2) ratchet up monotonically toward `lambda_max`
    based on the `loss_res / loss_ic(or bc)` ratio, frozen during a warmup window; `lambda_res` is fixed
    at 1.0 throughout — see `_update_adaptive_weights()`.
- `train_pinn(...)` is called multiple times per script/notebook to represent training **phases**
  (e.g. Phase 1 initial training, Phase 2/3 retraining). Each call builds a **fresh `optim.Adam`**, so by
  default Adam's moment estimates reset between phases — this reset-vs-preserve distinction between
  phases is the entire experimental variable the project studies.
- The "memory-aware" variants (`memoryAwarePINN/`, the 2D notebooks with "without_memory" siblings) are
  the same model/training loop, modified to save and reload the Adam optimizer's `state_dict` between
  phases instead of constructing a new optimizer — compare against the paired non-memory script/notebook
  to see the effect.

### Data flow (per experiment: 1D via `pinn.py`, 2D via `2d_pinn/dataset_generation_2d.py`)

1. Generate synthetic points (`generate_samples()` / `generate_2d_data()`) → save to a `.npz` with
   `collocation` / `initial` / `boundary` arrays, columns ordered `[x, (y,) t]`.
2. `prepare_training_data()` loads the `.npz`, computes exact-solution labels for IC/BC points, returns
   tensors already split by point type.
3. Train in phases, evaluating relative L2 error (paper Eq. 8) against the analytical traveling-wave
   solution on a held-out grid at fixed `t` after every `print_every` iterations.
4. Benchmark against an explicit finite-difference solver (`run_fdm()` in `pinn.py`, or
   `fdm_solver_2d.py`) as an independent accuracy reference.
5. Save model checkpoint(s) (`.pth`, containing `model_state_dict` + full loss/L2/lambda history),
   result summary (`.npz`), and a 4-panel diagnostic plot (`plot_training_history()`) reproducing the
   paper's Figure 2 layout plus an L2-error panel with paper reference lines overlaid.

### Sweeps

`results/2d/sweep/sweep_results_2d_32.csv` and `memoryAwarePINN/sweep_results_32.csv` hold results from
architecture/LR-schedule sweeps (e.g. `7x50` vs `6x100`, combinations of
cosine/linear/exponential/exponential_delayed schedules for Phase 1 vs Phase 2), named after the
`l2_<dim>_tanh__<arch>__<phase1>_<phase2>_lbfgs.png` plot files in `results/2d/sweep/runs/` and
`2d_pinn/`. `generate_2d_plots.py` and `generate_comparison_plots.py` consume these CSVs; there is no
script that regenerates the CSVs themselves in this repo — they come from the sweep runs performed in
the notebooks.

## Key numeric conventions to preserve when editing training code

- Tensor column/argument order is always `(x, t)` in 1D and `(x, y, t)` in 2D — never `(t, x, ...)`.
- `requires_grad_(True)` on collocation tensors happens once per phase, outside the residual function.
- LR scheduler (`ExponentialLR`) steps every `scheduler_step_every` iterations, not every iteration —
  stepping every iteration decays the LR to ~0 well before training finishes.
- A new `optim.Adam` per phase is the paper-faithful ("memory-less") baseline; reusing/reloading the
  optimizer `state_dict` is the "memory-aware" experimental condition — don't conflate the two when
  editing either variant.
- `lambda_max` (IC/BC weight cap) intentionally matches the paper's value (10000) even though a much
  smaller cap changes convergence behavior — see the comments in `_update_adaptive_weights()` before
  changing it.
