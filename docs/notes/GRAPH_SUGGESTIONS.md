# Alternative Graph Suggestions for 2D PINN Paper

Based on analysis of the 2D data generation, PINN notebook, FDM solver, and sweep results, here are actionable graph suggestions to visualize the 2D Fisher-KPP results:

---

## 1. **Architecture Collapse Comparison (CRITICAL)**
**Type:** Bar chart with error bars or dual-panel comparison  
**Data Source:** `sweep_results_2d_32.csv` (Phase 3 L2 errors)

**What it shows:**
- Left panel: Final $L_2$ error for all 16 successful $7\times50$ configurations (range: 0.0159–0.0329)
- Right panel: All 16 failed $6\times100$ configurations (all stuck at ~1.0)
- Visualizes the dramatic failure of wider architecture

**Why include it:**
- This is the paper's most striking finding: **wider ≠ better**
- Shows clear separation between converging (7×50) and diverging (6×100) regimes
- Directly supports the claim that architecture interacts non-trivially with the 2D landscape

**Matplotlib code template:**
```python
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 7x50 errors (16 values)
errors_7x50 = [0.0159, 0.0159, 0.0166, 0.0176, 0.0181, 0.0196, 0.0198, 0.0215,
               0.0232, 0.0234, 0.0235, 0.0249, 0.0314, 0.0319, 0.0320, 0.0329]
ax1.bar(range(1, 17), errors_7x50, color='blue', alpha=0.7)
ax1.set_ylabel('Final $L_2$ Error', fontsize=12)
ax1.set_xlabel('Configuration Rank', fontsize=12)
ax1.set_title('7×50 Architecture (Converging)', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)

# 6x100 errors (all ~1.0)
errors_6x100 = np.ones(16) * 1.0
ax2.bar(range(1, 17), errors_6x100, color='red', alpha=0.7)
ax2.set_ylabel('Final $L_2$ Error', fontsize=12)
ax2.set_xlabel('Configuration Rank', fontsize=12)
ax2.set_title('6×100 Architecture (Diverging)', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0.001, 2.0])

plt.suptitle('Architecture-Level Divergence in 2D Fisher-KPP Sweep', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('architecture_collapse_comparison.png', dpi=150, bbox_inches='tight')
```

---

## 2. **Schedule Impact Heatmap (7×50 only)**
**Type:** Heatmap with Phase1/Phase2 schedules as axes  
**Data Source:** `sweep_results_2d_32.csv` (rows 1–16)

**What it shows:**
- X-axis: Phase 1 schedule (exponential, cosine, exp-delayed, linear)
- Y-axis: Phase 2 schedule (same 4 options)
- Color: Final $L_2$ error from Phase 3
- Highlights which schedule *combinations* work best

**Why include it:**
- Shows schedule *interactions*: certain Phase1+Phase2 pairs synergize
- Exponential Phase 1 row is clearly "hot" (blue = good)
- Linear Phase 1 row is "cool" (red = bad)
- Makes the dominance of Exponential Phase 1 visually obvious

**Data arrangement (7×50 results):**
```
                Exp-Delayed  Exponential  Linear    Cosine
Exponential        0.0159      0.0159     0.0166    0.0176
Cosine             0.0181      0.0198     0.0216    0.0215
Exp-Delayed        0.0232      0.0234     0.0235    0.0249
Linear             0.0314      0.0319     0.0320    0.0329
```

**Python template:**
```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = np.array([
    [0.0159, 0.0159, 0.0166, 0.0176],
    [0.0181, 0.0198, 0.0216, 0.0215],
    [0.0232, 0.0234, 0.0235, 0.0249],
    [0.0314, 0.0319, 0.0320, 0.0329]
])

phases = ['Exponential', 'Cosine', 'Exp-Delayed', 'Linear']
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data, annot=True, fmt='.4f', cmap='RdYlGn_r', cbar_kws={'label': 'Final $L_2$'},
            xticklabels=['Exp-Delayed', 'Exponential', 'Linear', 'Cosine'],
            yticklabels=['Exponential', 'Cosine', 'Exp-Delayed', 'Linear'],
            ax=ax, vmin=0.01, vmax=0.04)
ax.set_xlabel('Phase 2 Schedule', fontsize=12, fontweight='bold')
ax.set_ylabel('Phase 1 Schedule', fontsize=12, fontweight='bold')
ax.set_title('2D Schedule Optimization Landscape (7×50 Architecture)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('schedule_heatmap_7x50.png', dpi=150, bbox_inches='tight')
```

---

## 3. **Single Best Run: Loss & L2 Evolution (Phase-Colored)**
**Type:** Dual-axis plot with phase background shading  
**Data Source:** `training_history_single_run_2d.png` (already generated) + `pinn_results_single_2d.npz`

**What it shows:**
- Total Loss (log scale) vs. iteration: smooth descent through Phase 1+2, sharp drop at L-BFGS
- Relative $L_2$ error vs. iteration: overlaid, color-coded by phase
- Vertical dashed lines at phase boundaries (10k, 15k iterations)
- Background shading: Adam Phase 1 (blue), Adam Phase 2 (green), L-BFGS Phase 3 (orange)

**Why include it:**
- Demonstrates that optimizer state preservation maintains smooth descent
- Shows how L-BFGS triggers the "vertical collapse" to $L_2 \approx 0.006$
- Visual proof that the method works: continuous improvement without plateaus

**Notes:** The notebook already generates this plot; ensure the phase boundaries and colors are clear.

---

## 4. **Loss Component Trajectory (IC, BC, Residual)**
**Type:** Log-scale line plot with three curves  
**Data Source:** `pinn_results_single_2d.npz` (loss component histories)

**What it shows:**
- Three lines: $\mathcal{L}_{IC}$, $\mathcal{L}_{BC}$, $\mathcal{L}_{res}$ vs. iteration
- IC and BC drop rapidly to ~$10^{-7}$–$10^{-8}$
- Residual dominates and only begins to collapse in L-BFGS phase
- Demonstrates adaptive weighting effectiveness

**Why include it:**
- Explains why $\mathcal{L}_{res}$ is the binding constraint
- Shows that the IC/BC losses are effectively "solved" by Phase 2
- Illustrates the role of the adaptive weighting mechanism

**Python template:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Load from npz
data = np.load('../pinn_results_single_2d.npz')
loss_ic = data['loss_ic_history']
loss_bc = data['loss_bc_history']
loss_res = data['loss_res_history']
iters = data['iteration_history']

fig, ax = plt.subplots(figsize=(11, 6))
ax.semilogy(iters, loss_ic, 'b-', lw=2, label='$\mathcal{L}_{IC}$')
ax.semilogy(iters, loss_bc, 'g-', lw=2, label='$\mathcal{L}_{BC}$')
ax.semilogy(iters, loss_res, 'r-', lw=2, label='$\mathcal{L}_{res}$ (PDE)')

# Phase boundaries
ax.axvline(10000, color='gray', linestyle='--', lw=1.5, alpha=0.5)
ax.axvline(15000, color='gray', linestyle='--', lw=1.5, alpha=0.5)
ax.text(5000, 1e-2, 'Phase 1\n(Adam)', ha='center', fontsize=10, alpha=0.6)
ax.text(12500, 1e-2, 'Phase 2\n(Adam)', ha='center', fontsize=10, alpha=0.6)
ax.text(17500, 1e-2, 'Phase 3\n(L-BFGS)', ha='center', fontsize=10, alpha=0.6)

ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss Component', fontsize=12, fontweight='bold')
ax.set_title('2D PINN Loss Component Evolution (7×50, Best Config)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_components_2d.png', dpi=150, bbox_inches='tight')
```

---

## 5. **FDM vs. PINN Solution Comparison (2D Surface)**
**Type:** 2D contour plots or 3D surface comparison  
**Data Source:** `fdm_solution_2d.npz` (FDM reference) + model prediction at best configuration

**What it shows:**
- Top: FDM reference solution $u(x,y,t=1.0)$ on $[0,1]^2$
- Middle: PINN prediction (7×50, best schedule) on same grid
- Bottom: Pointwise error $|u_{PINN} - u_{FDM}|$ (color-mapped)

**Why include it:**
- Provides intuition for what the 2D solution looks like (Gaussian-like spreading from center)
- Shows that PINN approximation is smooth and faithful to FDM
- Error map reveals any spurious oscillations or boundary artifacts

**Python template:**
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

# Load FDM and generate PINN predictions
fdm_data = np.load('../fdm_solution_2d.npz')
x_fdm = fdm_data['x'].reshape(51, 51)
y_fdm = fdm_data['y'].reshape(51, 51)
u_fdm = fdm_data['u_exact'].reshape(51, 51)

# ... [Code to run PINN model on same grid] ...
u_pinn = ...  # shape (51, 51)
error = np.abs(u_pinn - u_fdm)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# FDM
im0 = axes[0].contourf(x_fdm, y_fdm, u_fdm, levels=20, cmap='viridis')
axes[0].set_title('FDM Reference', fontsize=12, fontweight='bold')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')
plt.colorbar(im0, ax=axes[0], label='$u(x,y,t=1.0)$')

# PINN
im1 = axes[1].contourf(x_fdm, y_fdm, u_pinn, levels=20, cmap='viridis')
axes[1].set_title('PINN (7×50, Exp/Exp-Delayed)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$y$')
plt.colorbar(im1, ax=axes[1], label='$\hat{u}_{PINN}(x,y,t=1.0)$')

# Error
im2 = axes[2].contourf(x_fdm, y_fdm, error, levels=20, cmap='RdYlBu_r')
axes[2].set_title(f'Pointwise Error (max: {error.max():.2e})', fontsize=12, fontweight='bold')
axes[2].set_xlabel('$x$'); axes[2].set_ylabel('$y$')
plt.colorbar(im2, ax=axes[2], label='$|u_{PINN} - u_{FDM}|$')

fig.suptitle('2D Fisher-KPP Solution: FDM vs. PINN at $t=1.0$', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('solution_comparison_2d_contour.png', dpi=150, bbox_inches='tight')
```

---

## 6. **Schedule Robustness: Best 7×50 Configurations (Box Plot)**
**Type:** Box or violin plot showing L2 distribution by Phase 1 choice  
**Data Source:** `sweep_results_2d_32.csv` (rows 1–16)

**What it shows:**
- Four groups: one for each Phase 1 schedule
- Within each group: 4 box plots (one per Phase 2 schedule)
- Shows median, IQR, and range of Phase 3 L2 errors
- Highlights Exponential dominance with tight spread

**Why include it:**
- Quantifies schedule robustness: Exponential Phase 1 has tightest error band
- Linear Phase 1 has widest spread, indicating instability
- Makes rank ordering clear at a glance

---

## 7. **Convergence Speed Ranking (Time vs. Error)**
**Type:** Scatter plot with time on x-axis, final L2 on y-axis  
**Data Source:** `sweep_results_2d_32.csv` (columns: time_min, l2_phase3)

**What it shows:**
- 7×50 runs: cloud of 16 points clustered near (2.3 min, 0.016–0.033)
- 6×100 runs: cloud of 16 points at (4.2 min, 1.0)
- Shows that wider architecture is slower *and* diverges
- Pareto frontier: no 6×100 run Pareto-dominates any 7×50 run

**Why include it:**
- Efficiency argument: 7×50 is better on both speed and accuracy
- Justifies recommending 7×50 for 2D problems

---

## 8. **Ablation Summary Table with Visualization**
**Type:** Table + accompanying bar chart  
**Data Source:** `sweep_results_2d_32.csv`

**Shows:**
- Best/worst for each Phase 1 choice (among 7×50)
- Speed implications: why 6×100 is slower
- Summary: "Exponential Phase 1 is best; 7×50 mandatory for convergence"

---

## Summary Table: Which Graphs to Include in Paper

| Graph # | Title | Type | Key Message | Priority |
|---------|-------|------|-------------|----------|
| 1 | Architecture Collapse | Bar chart | Wider ≠ Better; 6×100 completely fails | **CRITICAL** |
| 2 | Schedule Heatmap (7×50) | Heatmap | Exponential Phase 1 dominates | **HIGH** |
| 3 | Best Run Evolution | Dual-axis + shading | Smooth descent → L-BFGS collapse | HIGH |
| 4 | Loss Components | Log plot (3 lines) | Residual is binding; IC/BC solved | MEDIUM |
| 5 | FDM vs. PINN 2D | Contour comparison | Visual proof of convergence | MEDIUM |
| 6 | Schedule Robustness | Box plot | Exponential tightest; Linear widest | MEDIUM |
| 7 | Time vs. Error | Scatter plot | 7×50 Pareto-dominates 6×100 | LOW |
| 8 | Ablation Summary | Table + bars | Decision tree for practitioners | LOW |

---

## Implementation Notes

1. **Existing plots:** The notebook already generates `training_history_single_run_2d.png` and individual `l2_2d_tanh__*.png` plots. These are useful but hard to compare; a summary heatmap (Graph 2) would be much more informative.

2. **Data dependencies:** All suggested plots can be generated from:
   - `sweep_results_2d_32.csv` (main results table)
   - `pinn_results_single_2d.npz` (single run history)
   - `fdm_solution_2d.npz` (FDM reference)
   - Model predictions (requires running the best 7×50 config)

3. **Color schemes:**
   - Use viridis for solution fields (FDM vs. PINN)
   - Use RdYlGn_r for error maps (red = bad)
   - Use gray for phase backgrounds, blue/green/orange for phases

4. **Reproducibility:** All plot scripts should use the same seed (42) and refer to relative paths to ensure they work whether run from the main folder or `memoryAwarePINN/`.

---

## Quick Start: Generate All Plots

```bash
cd /path/to/project
python generate_2d_plots.py  # Placeholder script that generates all 8 graphs
```

Would you like me to create a `generate_2d_plots.py` script that automates all of these plots?
