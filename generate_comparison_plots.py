#!/usr/bin/env python3
"""
Generate 6 key comparison graphs:
- 3 graphs for 2D PINN (2d_pinn)
- 3 graphs for Modified PINN (1D, original protocol)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Paths (relative to this script's location, not the current working directory)
repo_root = Path(__file__).resolve().parent
output_dir = str(repo_root / "figures_comparison")
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("2D PINN vs Modified PINN Comparison Graphs (with Training Time)")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[Loading Data]")

# Load 2D data
csv_2d_path = repo_root / "results" / "2d" / "sweep" / "sweep_results_2d_32.csv"
df_2d = pd.read_csv(csv_2d_path)
print(f"✓ Loaded 2D sweep: {len(df_2d)} configurations")
print(f"  Columns: {list(df_2d.columns)}")

# Load Modified PINN data
csv_mod_path = repo_root / "memoryAwarePINN" / "sweep_results_32.csv"
df_mod = pd.read_csv(csv_mod_path)
print(f"✓ Loaded Modified PINN sweep: {len(df_mod)} configurations")
print(f"  Columns: {list(df_mod.columns)}")

# ============================================================================
# GRAPH 1: 2D PINN - Architecture Comparison
# ============================================================================
print("\n[1/4] Creating 2D Graph 1: Architecture Collapse...")
try:
    df_2d_7x50 = df_2d[df_2d['architecture'].str.contains('7x50', na=False)]
    df_2d_6x100 = df_2d[df_2d['architecture'].str.contains('6x100', na=False)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 7x50
    errors_7x50 = sorted(df_2d_7x50['l2_phase3'].values)
    ax1.bar(range(1, len(errors_7x50) + 1), errors_7x50, color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Final $L_2$ Error', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Configuration Rank', fontsize=12, fontweight='bold')
    ax1.set_title('7×50 (Converging)', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0.005, 1.5])
    
    # 6x100
    errors_6x100 = sorted(df_2d_6x100['l2_phase3'].values)
    ax2.bar(range(1, len(errors_6x100) + 1), errors_6x100, color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Final $L_2$ Error', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Configuration Rank', fontsize=12, fontweight='bold')
    ax2.set_title('6×100 (Diverging)', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0.005, 1.5])
    
    fig.suptitle('2D Fisher-KPP: Architecture Impact on Convergence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2d_graph1_architecture.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: 2d_graph1_architecture.png")
    plt.close()
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# GRAPH 2: 2D PINN - Schedule Performance (Box Plot)
# ============================================================================
print("[2/4] Creating 2D Graph 2: Schedule Robustness...")
try:
    fig, ax = plt.subplots(figsize=(11, 6))
    
    schedule_groups = []
    l2_by_schedule = []
    
    for sched in ['exponential', 'cosine', 'exp_delayed', 'linear']:
        mask = df_2d_7x50['sched_p1'].str.contains(sched, na=False)
        l2_vals = df_2d_7x50[mask]['l2_phase3'].values
        schedule_groups.extend([sched] * len(l2_vals))
        l2_by_schedule.extend(l2_vals)
    
    box_data = pd.DataFrame({'Schedule': schedule_groups, 'L2 Error': l2_by_schedule})
    
    # Create box plot
    box_order = ['exponential', 'cosine', 'exp_delayed', 'linear']
    data_for_box = [box_data[box_data['Schedule'] == s]['L2 Error'].values for s in box_order]
    
    bp = ax.boxplot(data_for_box,
                    tick_labels=[s.replace('_', ' ').title() for s in box_order],
                    patch_artist=True, widths=0.6, showmeans=True)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Final $L_2$ Error', fontsize=12, fontweight='bold')
    ax.set_xlabel('Phase 1 Schedule', fontsize=12, fontweight='bold')
    ax.set_title('2D Fisher-KPP: Schedule Robustness (7×50 only)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2d_graph2_schedule.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: 2d_graph2_schedule.png")
    plt.close()
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# GRAPH 3: Modified PINN - L2 Performance Distribution
# ============================================================================
print("[3/4] Creating Modified PINN Graph 1: L2 Distribution...")
try:
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Extract architecture information
    if 'architecture' in df_mod.columns:
        architectures = df_mod['architecture'].unique()
    elif 'arch' in df_mod.columns:
        architectures = df_mod['arch'].unique()
    else:
        # Try to extract from run name
        architectures = ['Unknown']
    
    # Get final L2 errors (try different column names)
    l2_col = None
    for col in ['l2_phase3', 'Final_L2_Error', 'l2_final', 'L2']:
        if col in df_mod.columns:
            l2_col = col
            break
    
    if l2_col is None:
        print("  ⚠ Could not find L2 error column")
    else:
        l2_vals = df_mod[l2_col].values
        
        # Create histogram
        ax.hist(l2_vals, bins=15, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axvline(l2_vals.mean(), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {l2_vals.mean():.4f}')
        ax.axvline(np.median(l2_vals), color='green', linestyle='--', linewidth=2.5, label=f'Median: {np.median(l2_vals):.4f}')
        
        ax.set_xlabel('Final $L_2$ Error', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Modified PINN (1D): L2 Error Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/modified_graph1_l2_distribution.png", dpi=150, bbox_inches='tight')
        print("✓ Saved: modified_graph1_l2_distribution.png")
        plt.close()
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# GRAPH 4: Modified PINN - Schedule Performance Ranking
# ============================================================================
print("[4/4] Creating Modified PINN Graph 2: Schedule Ranking...")
try:
    # Identify schedule columns
    sched_col_p1 = None
    sched_col_p2 = None
    l2_col = None
    
    for col in df_mod.columns:
        if 'sched' in col.lower() and ('p1' in col.lower() or 'phase1' in col.lower() or 'phase_1' in col.lower()):
            sched_col_p1 = col
        if 'sched' in col.lower() and ('p2' in col.lower() or 'phase2' in col.lower() or 'phase_2' in col.lower()):
            sched_col_p2 = col
        if 'l2' in col.lower() and ('phase3' in col.lower() or 'phase_3' in col.lower() or 'final' in col.lower()):
            l2_col = col
    
    if sched_col_p1 is None or l2_col is None:
        print("  ⚠ Could not find schedule or L2 columns")
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by Phase 1 schedule
        phase1_stats = []
        schedules_found = df_mod[sched_col_p1].unique()
        
        for sched in sorted(schedules_found):
            mask = df_mod[sched_col_p1] == sched
            l2_vals = df_mod[mask][l2_col].values
            if len(l2_vals) > 0:
                phase1_stats.append({
                    'Schedule': str(sched).replace('_', '\n').title(),
                    'Mean': l2_vals.mean(),
                    'Std': l2_vals.std(),
                    'Count': len(l2_vals)
                })
        
        if len(phase1_stats) > 0:
            stats_df = pd.DataFrame(phase1_stats)
            x_pos = np.arange(len(stats_df))
            
            ax.bar(x_pos, stats_df['Mean'], yerr=stats_df['Std'], capsize=5,
                   color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(stats_df)], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(stats_df['Schedule'])
            ax.set_ylabel('Mean $L_2$ Error', fontsize=12, fontweight='bold')
            ax.set_xlabel('Phase 1 Schedule', fontsize=12, fontweight='bold')
            ax.set_title('Modified PINN (1D): Phase 1 Schedule Impact', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/modified_graph2_schedule_ranking.png", dpi=150, bbox_inches='tight')
            print("✓ Saved: modified_graph2_schedule_ranking.png")
            plt.close()
        else:
            print("  ⚠ No valid schedule data found")
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# GRAPH 5: 2D PINN - Training Time Distribution
# ============================================================================
print("[5/6] Creating 2D Graph 3: Training Time Analysis...")
try:
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Separate by architecture
    time_7x50 = df_2d_7x50['time_min'].values
    time_6x100 = df_2d_6x100['time_min'].values
    
    # Create box plot
    bp = ax.boxplot([time_7x50, time_6x100],
                    tick_labels=['7×50 (Converging)', '6×100 (Diverging)'],
                    patch_artist=True, widths=0.6, showmeans=True)
    
    colors = ['#2E86AB', '#A23B72']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('2D Fisher-KPP: Training Time by Architecture', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f"7×50: {time_7x50.mean():.2f} ± {time_7x50.std():.2f} min\n"
    stats_text += f"6×100: {time_6x100.mean():.2f} ± {time_6x100.std():.2f} min"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2d_graph3_training_time.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: 2d_graph3_training_time.png")
    plt.close()
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# GRAPH 6: Modified PINN - Training Time Distribution
# ============================================================================
print("[6/6] Creating Modified PINN Graph 3: Training Time Analysis...")
try:
    fig, ax = plt.subplots(figsize=(11, 6))
    
    time_vals = df_mod['time_min'].values
    
    # Create histogram
    ax.hist(time_vals, bins=12, color='#A23B72', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axvline(time_vals.mean(), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {time_vals.mean():.2f} min')
    ax.axvline(np.median(time_vals), color='green', linestyle='--', linewidth=2.5, label=f'Median: {np.median(time_vals):.2f} min')
    
    ax.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Modified PINN (1D): Training Time Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/modified_graph3_training_time.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: modified_graph3_training_time.png")
    plt.close()
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON STATISTICS")
print("=" * 80)

print("\n2D PINN (32 configurations):")
print(f"  Architecture 7×50 (Converging):")
print(f"    - L2 Error:     {df_2d_7x50['l2_phase3'].mean():.6f} ± {df_2d_7x50['l2_phase3'].std():.6f}")
print(f"    - Training Time: {df_2d_7x50['time_min'].mean():.2f} ± {df_2d_7x50['time_min'].std():.2f} min")
print(f"  Architecture 6×100 (Diverging):")
print(f"    - L2 Error:     {df_2d_6x100['l2_phase3'].mean():.6f} ± {df_2d_6x100['l2_phase3'].std():.6f}")
print(f"    - Training Time: {df_2d_6x100['time_min'].mean():.2f} ± {df_2d_6x100['time_min'].std():.2f} min")

print("\nModified PINN (1D, 32 configurations):")
print(f"  Overall Performance:")
print(f"    - L2 Error:     {df_mod['l2_phase3'].mean():.6f} ± {df_mod['l2_phase3'].std():.6f}")
print(f"    - Training Time: {df_mod['time_min'].mean():.2f} ± {df_mod['time_min'].std():.2f} min")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n✓ All 6 graphs saved to: {output_dir}")
print(f"\nGenerated files:")
print(f"  1. 2d_graph1_architecture.png        - 2D Architecture Comparison")
print(f"  2. 2d_graph2_schedule.png            - 2D Schedule Robustness")
print(f"  3. 2d_graph3_training_time.png       - 2D Training Time by Architecture")
print(f"  4. modified_graph1_l2_distribution.png    - Modified PINN L2 Distribution")
print(f"  5. modified_graph2_schedule_ranking.png   - Modified PINN Schedule Ranking")
print(f"  6. modified_graph3_training_time.png      - Modified PINN Training Time Distribution")
print("\n" + "=" * 80)
