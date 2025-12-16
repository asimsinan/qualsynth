#!/usr/bin/env python3
"""
Generate pre-computed tables for replication package.
Extracts results from JSON files and creates CSV summary tables.
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Configuration
DATASETS = ['german_credit', 'breast_cancer', 'pima_diabetes', 'wine_quality', 
            'yeast', 'haberman', 'thyroid', 'htru2']
METHODS = ['qualsynth', 'smote', 'ctgan', 'tabfairgdt']
SEEDS = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]

# Paths
PROJECT_DIR = Path(__file__).parent.parent
QUALSYNTH_RESULTS = PROJECT_DIR / "results" / "openrouter"
BASELINE_RESULTS = PROJECT_DIR / "results" / "experiments"
OUTPUT_DIR = PROJECT_DIR / "replication" / "tables"


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


def extract_metrics(result: dict) -> dict:
    """Extract key metrics from a result dict."""
    if result is None or not result.get('success', False):
        return None
    
    metrics = {
        'seed': result.get('seed'),
        'n_generated': result.get('n_generated', 0),
        'execution_time': result.get('execution_time', 0),
    }
    
    # Performance metrics (averaged across classifiers)
    avg_perf = result.get('avg_performance', {})
    metrics['f1_score'] = avg_perf.get('f1', np.nan)
    metrics['accuracy'] = avg_perf.get('accuracy', np.nan)
    metrics['precision'] = avg_perf.get('precision', np.nan)
    metrics['recall'] = avg_perf.get('recall', np.nan)
    metrics['roc_auc'] = avg_perf.get('roc_auc', np.nan)
    metrics['g_mean'] = avg_perf.get('g_mean', np.nan)
    metrics['balanced_accuracy'] = avg_perf.get('balanced_accuracy', np.nan)
    metrics['mcc'] = avg_perf.get('mcc', np.nan)
    
    # Fairness metrics (averaged across classifiers)
    avg_fair = result.get('avg_fairness', {})
    metrics['demographic_parity_diff'] = avg_fair.get('avg_demographic_parity_difference', np.nan)
    metrics['equalized_odds_diff'] = avg_fair.get('avg_equalized_odds_difference', np.nan)
    metrics['max_fairness_violation'] = avg_fair.get('max_fairness_violation', np.nan)
    
    # Validation metrics (from metadata)
    metadata = result.get('metadata', {})
    validation = metadata.get('validation', {})
    
    if 'validation_rate' in metadata:
        metrics['validation_rate'] = metadata['validation_rate']
    elif 'overall_pass_rate' in validation:
        metrics['validation_rate'] = validation['overall_pass_rate']
    else:
        metrics['validation_rate'] = 1.0  # Default for methods without validation
    
    metrics['duplicate_ratio'] = validation.get('duplicate_ratio', 0.0)
    
    return metrics


def generate_tables():
    """Generate all CSV tables for replication."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating tables to: {OUTPUT_DIR}")
    print(f"QualSynth results from: {QUALSYNTH_RESULTS}")
    print(f"Baseline results from: {BASELINE_RESULTS}")
    print()
    
    all_results = []
    
    for dataset in DATASETS:
        print(f"Processing {dataset}...")
        
        for method in METHODS:
            method_results = []
            
            for seed in SEEDS:
                result = load_result(dataset, method, seed)
                metrics = extract_metrics(result)
                
                if metrics:
                    metrics['dataset'] = dataset
                    metrics['method'] = method
                    method_results.append(metrics)
                    all_results.append(metrics)
            
            # Save per-method results
            if method_results:
                df = pd.DataFrame(method_results)
                output_path = OUTPUT_DIR / f"{dataset}_{method}_results.csv"
                df.to_csv(output_path, index=False)
                print(f"  Saved: {output_path.name} ({len(method_results)} seeds)")
    
    # Save combined results
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
        print(f"\nSaved combined results: all_results.csv ({len(all_results)} total)")
    
    # Generate summary statistics
    generate_summary_tables(OUTPUT_DIR)
    
    print("\nDone!")


def generate_summary_tables(output_dir: Path):
    """Generate summary statistics tables."""
    all_results_path = output_dir / "all_results.csv"
    
    if not all_results_path.exists():
        return
    
    df = pd.read_csv(all_results_path)
    
    # F1 Score summary
    f1_summary = df.groupby(['dataset', 'method'])['f1_score'].agg(['mean', 'std']).reset_index()
    f1_pivot = f1_summary.pivot(index='dataset', columns='method', values='mean')
    f1_pivot.to_csv(output_dir / "summary_f1_scores.csv")
    print(f"Saved: summary_f1_scores.csv")
    
    # Validation rate summary
    val_summary = df.groupby(['dataset', 'method'])['validation_rate'].agg(['mean']).reset_index()
    val_pivot = val_summary.pivot(index='dataset', columns='method', values='mean')
    val_pivot.to_csv(output_dir / "summary_validation_rates.csv")
    print(f"Saved: summary_validation_rates.csv")
    
    # G-Mean summary
    gmean_summary = df.groupby(['dataset', 'method'])['g_mean'].agg(['mean', 'std']).reset_index()
    gmean_pivot = gmean_summary.pivot(index='dataset', columns='method', values='mean')
    gmean_pivot.to_csv(output_dir / "summary_gmean.csv")
    print(f"Saved: summary_gmean.csv")
    
    # Fairness summary
    fair_summary = df.groupby(['dataset', 'method'])['demographic_parity_diff'].agg(['mean']).reset_index()
    fair_pivot = fair_summary.pivot(index='dataset', columns='method', values='mean')
    fair_pivot.to_csv(output_dir / "summary_fairness.csv")
    print(f"Saved: summary_fairness.csv")


if __name__ == "__main__":
    generate_tables()
