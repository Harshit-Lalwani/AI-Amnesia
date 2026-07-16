#!/usr/bin/env python3
"""
Build the two headline bar charts embedded in README.md, from already-verified
result numbers (see docs/report.tex / README.md for the source of each number:
notebook printed summaries and results/2d/sweep/sweep_results_2d_32.csv).
No experiments are run here — this script only visualizes existing results.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent
output_dir = repo_root / "figures_summary"
output_dir.mkdir(exist_ok=True)

# Validated 2-color categorical palette (dataviz skill, slots 1 & 6; both light-
# and dark-surface CVD/contrast checks pass for this pair).
BLUE = "#2a78d6"    # "reset" / diverging condition
ORANGE = "#eb6834"  # "fix" / converging condition
GRID = "#c9c9c6"
TEXT = "#333331"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.titleweight": "bold",
})


def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=1, linestyle='-')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=0)


# ============================================================================
# Figure 1: 1D reset-vs-preserved, relative L2 error by phase
# ============================================================================
phases = ["Phase 1", "Phase 2", "Phase 3\n(final)"]
reset = [5.01, 7.69, 8.70]        # relative L2 error, %  (optimizer reset each phase)
# Phase 1 is identical setup in both conditions (the reset-vs-preserve distinction only
# applies from Phase 2 onward), so both series share the reset-run's Phase 1 value here.
preserved = [5.01, 4.55, 4.56]    # relative L2 error, %  (optimizer state preserved)

x = np.arange(len(phases))
width = 0.34

fig, ax = plt.subplots(figsize=(7.5, 5))
b1 = ax.bar(x - width / 2, reset, width, color=BLUE, label="Optimizer reset each phase", zorder=3)
b2 = ax.bar(x + width / 2, preserved, width, color=ORANGE, label="Optimizer state preserved (the fix)", zorder=3)

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color=TEXT)

ax.set_xticks(x)
ax.set_xticklabels(phases)
ax.set_ylabel("Relative $L_2$ error (%)")
ax.set_ylim(0, 10)
ax.set_title("1D Fisher-KPP: retraining hurts with a reset optimizer,\nhelps with state preserved", fontsize=12)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
style_axis(ax)

fig.text(0.5, -0.02,
          "8.70%  →  4.56%  final error  (−47.6%)  once optimizer state is preserved across phases",
          ha='center', fontsize=9.5, color=TEXT, style='italic')

plt.tight_layout()
plt.savefig(output_dir / "1d_reset_vs_preserved.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved -> {output_dir / '1d_reset_vs_preserved.png'}")

# ============================================================================
# Figure 2: 2D architecture comparison — final L2 error (log) and training time
# ============================================================================
archs = ["7×50\n(converges)", "6×100\n(diverges)"]
l2_mean = [0.023010, 0.999974]
l2_std = [0.006059, 0.000096]
time_mean = [2.31, 4.23]
time_std = [0.04, 0.01]
colors = [ORANGE, BLUE]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

bars1 = ax1.bar(archs, l2_mean, yerr=l2_std, capsize=5, color=colors, zorder=3,
                error_kw={"ecolor": TEXT, "elinewidth": 1.2})
ax1.set_yscale("log")
ax1.set_ylim(0.01, 2.2)
ax1.set_ylabel("Final $L_2$ error (log scale)")
ax1.set_title("Accuracy", fontsize=12)
for bar, m, s in zip(bars1, l2_mean, l2_std):
    ax1.annotate(f"{m:.3f}", (bar.get_x() + bar.get_width() / 2, m + s),
                 xytext=(0, 8), textcoords="offset points",
                 ha='center', va='bottom', fontsize=10, color=TEXT)
style_axis(ax1)

bars2 = ax2.bar(archs, time_mean, yerr=time_std, capsize=5, color=colors, zorder=3,
                error_kw={"ecolor": TEXT, "elinewidth": 1.2})
ax2.set_ylabel("Training time (minutes)")
ax2.set_title("Cost", fontsize=12)
ax2.set_ylim(0, 5)
for bar, m, s in zip(bars2, time_mean, time_std):
    ax2.annotate(f"{m:.2f}", (bar.get_x() + bar.get_width() / 2, m + s),
                 xytext=(0, 8), textcoords="offset points",
                 ha='center', va='bottom', fontsize=10, color=TEXT)
style_axis(ax2)

fig.suptitle("2D Fisher-KPP: the wider 6×100 network diverges on every\nconfiguration tested, and still costs more to train",
             fontsize=12, y=1.04)
plt.tight_layout()
plt.savefig(output_dir / "2d_architecture_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved -> {output_dir / '2d_architecture_comparison.png'}")
