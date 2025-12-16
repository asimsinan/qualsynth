#!/usr/bin/env python3
"""
Generate Critical Difference (CD) Diagrams for QualSynth Paper

This script creates CD diagrams showing statistical comparisons between methods
based on the Friedman test and Nemenyi post-hoc test.

Reference: Demšar (2006) "Statistical Comparisons of Classifiers over Multiple Data Sets"
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASETS = ['german_credit', 'breast_cancer', 'pima_diabetes', 'wine_quality', 
            'yeast', 'haberman', 'thyroid', 'htru2']
METHODS = ['qualsynth', 'smote', 'ctgan', 'tabfairgdt']
METHOD_NAMES = ['QualSynth', 'SMOTE', 'CTGAN', 'TabFairGDT']
SEEDS = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]

PROJECT_DIR = Path(__file__).parent.parent
QUALSYNTH_RESULTS = PROJECT_DIR / "results" / "openrouter"
BASELINE_RESULTS = PROJECT_DIR / "results" / "experiments"
OUTPUT_DIR = PROJECT_DIR / "paper" / "figures"


def load_result(dataset: str, method: str, seed: int) -> dict:
    """Load a single result JSON file."""
    if method == 'qualsynth':
        path = QUALSYNTH_RESULTS / dataset / "qualsynth" / f"seed{seed}.json"
    else:
        path = BASELINE_RESULTS / dataset / method / f"seed{seed}.json"
    
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_metric_matrix(metric: str = 'f1') -> np.ndarray:
    """
    Create a matrix of metric values: rows = datasets, columns = methods.
    Values are averaged across seeds.
    """
    matrix = []
    
    for dataset in DATASETS:
        row = []
        for method in METHODS:
            values = []
            for seed in SEEDS:
                result = load_result(dataset, method, seed)
                if result and result.get('success', False):
                    avg_perf = result.get('avg_performance', {})
                    if metric == 'f1':
                        val = avg_perf.get('f1', np.nan)
                    elif metric == 'g_mean':
                        val = avg_perf.get('g_mean', np.nan)
                    elif metric == 'roc_auc':
                        val = avg_perf.get('roc_auc', np.nan)
                    elif metric == 'accuracy':
                        val = avg_perf.get('accuracy', np.nan)
                    else:
                        val = avg_perf.get(metric, np.nan)
                    values.append(val)
            
            # Average across seeds
            if values:
                row.append(np.nanmean(values))
            else:
                row.append(np.nan)
        matrix.append(row)
    
    return np.array(matrix)


def compute_ranks(matrix: np.ndarray) -> np.ndarray:
    """
    Compute ranks for each dataset (row). Higher value = better = lower rank.
    """
    n_datasets, n_methods = matrix.shape
    ranks = np.zeros_like(matrix)
    
    for i in range(n_datasets):
        # Rank from highest to lowest (1 = best)
        ranks[i] = stats.rankdata(-matrix[i], method='average')
    
    return ranks


def nemenyi_cd(n_datasets: int, n_methods: int, alpha: float = 0.05) -> float:
    """
    Compute the critical difference for the Nemenyi test.
    
    CD = q_alpha * sqrt(k(k+1) / (6*N))
    
    where k = number of methods, N = number of datasets
    q_alpha values from Demšar (2006) Table 5
    """
    # Critical values for Nemenyi test (two-tailed)
    # q_alpha for alpha=0.05
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    
    q_alpha = q_alpha_table.get(n_methods, 2.569)  # Default to k=4
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    
    return cd


def draw_cd_diagram(avg_ranks: np.ndarray, method_names: list, cd: float, 
                    title: str, filename: str):
    """
    Draw a Critical Difference diagram.
    
    Based on Demšar (2006) visualization style.
    """
    n_methods = len(method_names)
    
    # Sort methods by rank
    sorted_indices = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_indices]
    sorted_names = [method_names[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Set up the axis
    min_rank = 1
    max_rank = n_methods
    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(0, 1)
    
    # Draw the axis line at y=0.5
    ax.axhline(y=0.5, color='black', linewidth=1.5, zorder=1)
    
    # Draw tick marks
    for rank in range(1, n_methods + 1):
        ax.plot([rank, rank], [0.45, 0.55], color='black', linewidth=1.5, zorder=1)
        ax.text(rank, 0.35, str(rank), ha='center', va='top', fontsize=12)
    
    # Draw CD bar at the top
    cd_y = 0.85
    ax.plot([1, 1 + cd], [cd_y, cd_y], color='black', linewidth=2)
    ax.plot([1, 1], [cd_y - 0.03, cd_y + 0.03], color='black', linewidth=2)
    ax.plot([1 + cd, 1 + cd], [cd_y - 0.03, cd_y + 0.03], color='black', linewidth=2)
    ax.text(1 + cd/2, cd_y + 0.08, f'CD = {cd:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Draw method names and their ranks
    # Split into left side (better ranks) and right side (worse ranks)
    left_methods = []
    right_methods = []
    
    for i, (name, rank) in enumerate(zip(sorted_names, sorted_ranks)):
        if i < n_methods // 2:
            left_methods.append((name, rank))
        else:
            right_methods.append((name, rank))
    
    # Draw left side methods (better ranks)
    y_positions_left = np.linspace(0.7, 0.9, len(left_methods))[::-1]
    for (name, rank), y_pos in zip(left_methods, y_positions_left):
        # Draw line from rank position to label
        ax.plot([rank, rank], [0.55, y_pos - 0.05], color='black', linewidth=0.8)
        ax.plot([rank, 0.5], [y_pos - 0.05, y_pos - 0.05], color='black', linewidth=0.8)
        ax.text(0.4, y_pos - 0.05, f'{name} ({rank:.2f})', ha='right', va='center', fontsize=11)
    
    # Draw right side methods (worse ranks)
    y_positions_right = np.linspace(0.7, 0.9, len(right_methods))[::-1]
    for (name, rank), y_pos in zip(right_methods, y_positions_right):
        # Draw line from rank position to label
        ax.plot([rank, rank], [0.55, y_pos - 0.05], color='black', linewidth=0.8)
        ax.plot([rank, n_methods + 0.5], [y_pos - 0.05, y_pos - 0.05], color='black', linewidth=0.8)
        ax.text(n_methods + 0.6, y_pos - 0.05, f'({rank:.2f}) {name}', ha='left', va='center', fontsize=11)
    
    # Draw connections for methods that are NOT significantly different
    y_bar = 0.2
    bar_height = 0.03
    
    # Find groups of methods that are not significantly different
    groups = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            if abs(sorted_ranks[i] - sorted_ranks[j]) < cd:
                # These two are not significantly different
                groups.append((sorted_ranks[i], sorted_ranks[j]))
    
    # Merge overlapping groups and draw bars
    if groups:
        # Sort groups by start position
        groups = sorted(set(groups))
        
        # Draw bars for connected groups
        drawn_bars = []
        for start, end in groups:
            # Check if this bar overlaps with existing bars
            overlap = False
            for bar_start, bar_end, bar_y in drawn_bars:
                if not (end < bar_start or start > bar_end):
                    overlap = True
                    break
            
            if overlap:
                y_bar -= 0.08
            
            ax.plot([start, end], [y_bar, y_bar], color='black', linewidth=3)
            drawn_bars.append((start, end, y_bar))
    
    # Remove axes
    ax.axis('off')
    
    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {filename}")


def create_simple_cd_diagram(avg_ranks: np.ndarray, method_names: list, cd: float,
                             title: str, filename: str):
    """
    Create a simpler, cleaner CD diagram suitable for publication.
    """
    n_methods = len(method_names)
    
    # Sort methods by rank (lower rank = better)
    sorted_indices = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_indices]
    sorted_names = [method_names[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    # Axis setup
    lowv = 1
    highv = n_methods
    
    ax.set_xlim(lowv - 0.3, highv + 0.3)
    ax.set_ylim(-0.5, 1.5)
    
    # Main axis
    ax.hlines(1, lowv, highv, color='black', linewidth=1)
    
    # Tick marks
    for i in range(lowv, highv + 1):
        ax.vlines(i, 0.95, 1.05, color='black', linewidth=1)
        ax.text(i, 0.8, str(i), ha='center', fontsize=10)
    
    # CD bar
    ax.hlines(1.35, lowv, lowv + cd, color='black', linewidth=2)
    ax.vlines(lowv, 1.3, 1.4, color='black', linewidth=2)
    ax.vlines(lowv + cd, 1.3, 1.4, color='black', linewidth=2)
    ax.text(lowv + cd/2, 1.45, f'CD = {cd:.2f}', ha='center', fontsize=9)
    
    # Method labels
    half = n_methods // 2
    
    # Left side (better ranks)
    for i in range(half):
        rank = sorted_ranks[i]
        name = sorted_names[i]
        ax.vlines(rank, 1.05, 1.2, color='black', linewidth=0.8)
        ax.hlines(1.2, rank, lowv - 0.1, color='black', linewidth=0.8)
        ax.text(lowv - 0.15, 1.2 - i * 0.15, f'{name} ({rank:.2f})', 
                ha='right', va='center', fontsize=10)
    
    # Right side (worse ranks)
    for i in range(half, n_methods):
        rank = sorted_ranks[i]
        name = sorted_names[i]
        ax.vlines(rank, 1.05, 1.2, color='black', linewidth=0.8)
        ax.hlines(1.2, rank, highv + 0.1, color='black', linewidth=0.8)
        ax.text(highv + 0.15, 1.2 - (i - half) * 0.15, f'({rank:.2f}) {name}', 
                ha='left', va='center', fontsize=10)
    
    # Draw cliques (groups of statistically equivalent methods)
    # Find all pairs that are not significantly different
    clique_y = 0.6
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            if abs(sorted_ranks[j] - sorted_ranks[i]) < cd:
                ax.hlines(clique_y, sorted_ranks[i], sorted_ranks[j], 
                         color='black', linewidth=2.5)
        clique_y -= 0.15
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none')
    plt.close()
    
    print(f"  Saved: {filename}")


def main():
    print("=" * 60)
    print("Generating Critical Difference Diagrams")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    metrics = [
        ('f1', 'F1 Score'),
        ('g_mean', 'G-Mean'),
        ('roc_auc', 'ROC-AUC'),
    ]
    
    for metric, metric_name in metrics:
        print(f"\n{metric_name}:")
        
        # Get performance matrix
        matrix = get_metric_matrix(metric)
        print(f"  Matrix shape: {matrix.shape}")
        print(f"  Mean values: {np.nanmean(matrix, axis=0)}")
        
        # Compute ranks
        ranks = compute_ranks(matrix)
        avg_ranks = np.mean(ranks, axis=0)
        print(f"  Average ranks: {dict(zip(METHOD_NAMES, avg_ranks))}")
        
        # Compute CD
        n_datasets = len(DATASETS)
        n_methods = len(METHODS)
        cd = nemenyi_cd(n_datasets, n_methods, alpha=0.05)
        print(f"  Critical Difference (α=0.05): {cd:.3f}")
        
        # Friedman test
        stat, p_value = stats.friedmanchisquare(*[matrix[:, i] for i in range(n_methods)])
        print(f"  Friedman test: χ² = {stat:.2f}, p = {p_value:.4f}")
        
        # Generate CD diagram
        filename = OUTPUT_DIR / f"cd_diagram_{metric}.png"
        create_simple_cd_diagram(
            avg_ranks, METHOD_NAMES, cd,
            f'Critical Difference Diagram - {metric_name}',
            str(filename)
        )
    
    # Also generate a combined summary
    print("\n" + "=" * 60)
    print("Summary of Rankings (averaged across 8 datasets, 10 seeds)")
    print("=" * 60)
    
    matrix_f1 = get_metric_matrix('f1')
    ranks_f1 = compute_ranks(matrix_f1)
    avg_ranks_f1 = np.mean(ranks_f1, axis=0)
    
    print(f"\nF1 Score Rankings:")
    for name, rank in sorted(zip(METHOD_NAMES, avg_ranks_f1), key=lambda x: x[1]):
        print(f"  {rank:.2f}: {name}")
    
    cd = nemenyi_cd(len(DATASETS), len(METHODS))
    print(f"\nCritical Difference (α=0.05): {cd:.3f}")
    print("\nMethods connected by a bar are NOT significantly different.")


if __name__ == "__main__":
    main()
