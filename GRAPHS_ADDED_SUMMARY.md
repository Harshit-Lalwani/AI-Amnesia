# Summary: 6 Graphs Added to test.tex

## Overview
Successfully generated and integrated 6 publication-quality comparison graphs into the LaTeX document (`test.tex`). All graphs are stored in `figures_comparison/` and referenced with proper LaTeX figure environments.

---

## 2D PINN Graphs (3 files)

### 1. **2d_graph1_architecture.png** - Architecture Collapse Comparison
- **Location in tex:** Figure 1 (after 2D sweep table)
- **What it shows:**
  - Left panel: 7×50 architecture converges across all 16 configurations
  - Right panel: 6×100 architecture diverges entirely (all runs stuck at L2≈1.0)
- **Key insight:** Wider networks don't guarantee success in 2D; narrower 7×50 is more stable
- **Reference:** `\label{fig:2d_architecture_schedule}`

### 2. **2d_graph2_schedule.png** - Schedule Robustness (Box Plot)
- **Location in tex:** Figure 1 (same figure as above, panel b)
- **What it shows:** Final L2 errors for 7×50 runs grouped by Phase 1 schedule
- **Key insight:** Exponential Phase 1 has tightest, most robust performance
- **Reference:** `\label{fig:2d_architecture_schedule}`

### 3. **2d_graph3_training_time.png** - Training Time by Architecture
- **Location in tex:** Figure 2 (standalone figure)
- **What it shows:**
  - 7×50: 2.31 ± 0.04 minutes (efficient, converges)
  - 6×100: 4.23 ± 0.01 minutes (slower, diverges)
- **Key insight:** Wider architecture costs more computation AND fails to converge
- **Reference:** `\label{fig:2d_training_time}`

---

## Modified PINN (1D) Graphs (3 files)

### 4. **modified_graph1_l2_distribution.png** - L2 Error Distribution
- **Location in tex:** Figure 3 (after 1D sweep section)
- **What it shows:** Histogram of final L2 errors across 32 configurations
- **Statistics:** Mean: 0.0887 ± 0.0840
- **Key insight:** Shows performance spread across different schedule combinations
- **Reference:** `\label{fig:1d_schedule_analysis}`

### 5. **modified_graph2_schedule_ranking.png** - Phase 1 Schedule Ranking
- **Location in tex:** Figure 3 (same figure as above, panel b)
- **What it shows:** Mean L2 error for each Phase 1 schedule with error bars
- **Key insight:** Demonstrates schedule-dependent performance hierarchy in 1D
- **Reference:** `\label{fig:1d_schedule_analysis}`

### 6. **modified_graph3_training_time.png** - Training Time Distribution
- **Location in tex:** Figure 4 (standalone figure)
- **What it shows:** Histogram of training times with mean and median markers
- **Statistics:** Mean: 2.88 ± 2.60 minutes
- **Key insight:** Larger variance in 1D reflects sensitivity to schedule choices
- **Reference:** `\label{fig:1d_training_time}`

---

## LaTeX Integration Details

### File Locations in tex:
```tex
Line 331-332:  2D Figures 1 - 1D L2 Distribution + Schedule Ranking
Line 341:      2D Figure 4 - 1D Training Time
Line 410-411:  2D Figures 1 - 2D Architecture + Schedule
Line 420:      2D Figure 2 - 2D Training Time
```

### Figure References:
- `\ref{fig:1d_schedule_analysis}` - 1D schedule analysis (Figures 3a-b)
- `\ref{fig:1d_training_time}` - 1D training time (Figure 4)
- `\ref{fig:2d_architecture_schedule}` - 2D architecture comparison (Figures 1a-b)
- `\ref{fig:2d_training_time}` - 2D training time (Figure 2)

---

## Data Sources

All graphs generated from:
- **2D PINN:** `sweep_results_2d_32.csv` (32 configurations)
- **Modified PINN (1D):** `memoryAwarePINN/sweep_results_32.csv` (32 configurations)

---

## Generation Script

All graphs created using: `generate_comparison_plots.py`

```bash
python generate_comparison_plots.py
```

Output directory: `/home/sparsh-bhartia/Documents/mlns/project/MLNS_Project/figures_comparison/`

---

## Comparison Statistics

### 2D PINN Results:
- **7×50 Architecture:** L2 = 0.023010 ± 0.006059, Time = 2.31 ± 0.04 min
- **6×100 Architecture:** L2 = 0.999974 ± 0.000096, Time = 4.23 ± 0.01 min

### Modified PINN (1D) Results:
- **Overall:** L2 = 0.088709 ± 0.084048, Time = 2.88 ± 2.60 min

---

## Next Steps

1. **LaTeX Compilation:** Ensure `figures_comparison/` directory is accessible when compiling
2. **Figure Captions:** All captions include quantitative findings and key insights
3. **Cross-references:** Use `\ref{}` to cite figures in text (e.g., "As shown in Figure~\ref{fig:2d_architecture_schedule}...")
4. **Optional:** Add placeholder figures for loss curves and solution comparisons if needed

---

## Files Modified
- `test.tex` - Added 6 figure environments with proper captions and labels

## Files Created
- `figures_comparison/2d_graph1_architecture.png`
- `figures_comparison/2d_graph2_schedule.png`
- `figures_comparison/2d_graph3_training_time.png`
- `figures_comparison/modified_graph1_l2_distribution.png`
- `figures_comparison/modified_graph2_schedule_ranking.png`
- `figures_comparison/modified_graph3_training_time.png`
