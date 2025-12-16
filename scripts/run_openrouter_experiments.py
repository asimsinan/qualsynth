#!/usr/bin/env python3
"""
Run Experiments with OpenRouter API (Gemma 3 27B Free)

This script runs QualSynth experiments using OpenRouter's free Gemma 3 27B model
instead of local Ollama. Results are stored in results/openrouter/

Usage:
    python scripts/run_openrouter_experiments.py --datasets german_credit --seeds 42
    python scripts/run_openrouter_experiments.py --datasets thyroid --seeds 42 123 456
    python scripts/run_openrouter_experiments.py --all  # Run all datasets and seeds
"""

# CRITICAL: Set environment variables BEFORE any imports
import os
os.environ['PYTORCH_MPS_METAL'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPS_DISABLE'] = '1'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# OpenRouter Configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemma-3-27b-it:free"

# Rate Limiting Configuration
# Free tier: 20 req/min, 50 req/day
# Paid tier: UNLIMITED requests (no rate limit), 1000 req/day
RATE_LIMIT_REQUESTS_PER_MINUTE_FREE = 20
RATE_LIMIT_DELAY_SECONDS_FREE = 60.0 / RATE_LIMIT_REQUESTS_PER_MINUTE_FREE  # 3 seconds for free tier
RATE_LIMIT_DELAY_SECONDS_PAID = 1  # Minimal delay for paid tier (just to be safe)

# OpenRouter-optimized iteration parameters
# Cloud API is slower but larger model (27B) has better quality
OPENROUTER_TIME_PER_ITERATION_FREE = 1.5  # minutes (3s rate limit + ~60s API call + processing)
OPENROUTER_TIME_PER_ITERATION_PAID = 0.5  # minutes (no rate limit, just API call time)
OPENROUTER_EXPECTED_VALIDATION_RATE = 0.50  # 50% - larger model = better quality
OPENROUTER_MAX_TIME_HOURS = 24.0  # Allow overnight runs for large datasets

# Daily request limits
# See: https://openrouter.ai/docs/api-reference/limits
OPENROUTER_FREE_DAILY_LIMIT = 50  # Free tier: 50 req/day
OPENROUTER_PAID_DAILY_LIMIT = 1000  # Paid (any credit purchase): 1000 req/day

# Set OpenAI-compatible environment variables for OpenRouter
os.environ['OPENAI_API_BASE'] = OPENROUTER_BASE_URL
os.environ['OPENAI_API_KEY'] = OPENROUTER_API_KEY
# Rate limit delay will be set based on --paid-tier flag in run_openrouter_experiments()

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import traceback
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.qualsynth.experiments.experiment_runner import ExperimentRunner
from src.qualsynth.utils.config_loader import ConfigLoader


def calculate_openrouter_iterations(
    dataset_name: str,
    batch_size: int = 50,  # Match config default
    rate_limit_delay: float = None,  # None = auto based on tier
    max_time_hours: float = OPENROUTER_MAX_TIME_HOURS,
    paid_tier: bool = False,  # Set True if you have $10+ credits
    verbose: bool = True
) -> int:
    """
    Calculate optimal iterations for OpenRouter experiments.
    
    OpenRouter-specific considerations:
    - Rate limiting: 20 req/min (3s delay between calls)
    - Larger model (27B): Higher quality = better validation rate
    - Cloud latency: Slower than local
    - Daily limits: 50 req/day (free) or 1000 req/day (paid)
    
    Args:
        dataset_name: Name of the dataset
        batch_size: Samples per iteration
        rate_limit_delay: Seconds between API calls
        max_time_hours: Maximum runtime in hours
        paid_tier: Whether you have paid credits ($10+)
        verbose: Print calculation details
    
    Returns:
        Recommended number of iterations
    """
    from src.qualsynth.data.splitting import load_split
    
    # Load dataset to get class distribution
    try:
        split_data = load_split(dataset_name, seed=42)
        y_train = split_data['y_train']
    except Exception as e:
        print(f"⚠️  Could not load dataset {dataset_name}: {e}")
        print(f"   Using default: 30 iterations")
        return 30
    
    # Calculate target samples (gap to 1:1 balance)
    class_counts = y_train.value_counts()
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    gap = majority_count - minority_count
    target_samples = min(gap, 10000)  # Cap at 10k
    
    # OpenRouter-specific parameters
    validation_rate = OPENROUTER_EXPECTED_VALIDATION_RATE  # 50% for 27B model
    time_per_iteration = OPENROUTER_TIME_PER_ITERATION_PAID if paid_tier else OPENROUTER_TIME_PER_ITERATION_FREE
    
    # Set rate limit delay based on tier if not provided
    if rate_limit_delay is None:
        rate_limit_delay = RATE_LIMIT_DELAY_SECONDS_PAID if paid_tier else RATE_LIMIT_DELAY_SECONDS_FREE
    
    # Calculate iterations needed
    # Formula: iterations = target_samples / (batch_size * validation_rate * efficiency)
    efficiency = 0.7  # Higher efficiency for 27B model (less duplicates)
    samples_per_iteration = batch_size * validation_rate * efficiency
    base_iterations = int(np.ceil(target_samples / samples_per_iteration))
    
    # Apply time constraint
    max_iterations_by_time = int((max_time_hours * 60) / time_per_iteration)
    
    # Apply daily limit constraint based on tier
    daily_limit = OPENROUTER_PAID_DAILY_LIMIT if paid_tier else OPENROUTER_FREE_DAILY_LIMIT
    max_iterations_by_limit = daily_limit
    
    # Take minimum of all constraints
    recommended = min(base_iterations, max_iterations_by_time, max_iterations_by_limit)
    recommended = max(5, recommended)  # At least 5 iterations
    
    if verbose:
        tier_name = "PAID ($10+)" if paid_tier else "FREE"
        print()
        print("🔮 OpenRouter Iteration Calculator")
        print("=" * 70)
        print(f"📊 Dataset: {dataset_name}")
        print(f"   Majority: {majority_count}, Minority: {minority_count}")
        print(f"   Gap (target): {gap} samples")
        print(f"   Capped target: {target_samples} samples")
        print()
        print(f"⚙️  OpenRouter Parameters ({tier_name} tier):")
        print(f"   Batch size: {batch_size} samples/call")
        print(f"   Expected validation rate: {validation_rate*100:.0f}%")
        print(f"   Time per iteration: {time_per_iteration:.1f} min")
        print(f"   Rate limit delay: {rate_limit_delay:.1f}s")
        print(f"   Daily limit: {daily_limit} requests")
        print()
        print("📈 Calculation:")
        print(f"   Samples/iteration: {samples_per_iteration:.0f}")
        print(f"   Iterations to reach target: {base_iterations}")
        print(f"   Max by time ({max_time_hours}h): {max_iterations_by_time}")
        print(f"   Max by daily limit: {max_iterations_by_limit}")
        print()
        
        # Show recommended for current tier
        expected_samples = int(recommended * samples_per_iteration)
        expected_time = recommended * time_per_iteration
        print(f"✅ Recommended ({tier_name}): {recommended} iterations")
        print(f"   Expected samples: ~{expected_samples}")
        print(f"   Expected time: ~{expected_time:.0f} minutes ({expected_time/60:.1f} hours)")
        
        # Show what's possible with paid tier if currently free
        if not paid_tier and base_iterations > OPENROUTER_FREE_DAILY_LIMIT:
            paid_recommended = min(base_iterations, max_iterations_by_time, OPENROUTER_PAID_DAILY_LIMIT)
            paid_samples = int(paid_recommended * samples_per_iteration)
            paid_time = paid_recommended * time_per_iteration
            print()
            print(f"💰 With PAID tier (any credit purchase, e.g. $1):")
            print(f"   Iterations: {paid_recommended}")
            print(f"   Expected samples: ~{paid_samples}")
            print(f"   Expected time: ~{paid_time:.0f} minutes ({paid_time/60:.1f} hours)")
            print(f"   🎯 Can reach full target: {'YES' if paid_samples >= target_samples else 'NO'}")
            
            # Show multi-day option for free tier
            days_needed = int(np.ceil(base_iterations / OPENROUTER_FREE_DAILY_LIMIT))
            print()
            print(f"📅 FREE tier multi-day option:")
            print(f"   Days needed: {days_needed}")
            print(f"   Run 50 iterations/day, resume next day")
            print(f"   Total samples after {days_needed} days: ~{int(base_iterations * samples_per_iteration)}")
            print(f"   🎯 Can reach full target: YES (over {days_needed} days)")
        
        print("=" * 70)
        print()
    
    return recommended


def run_openrouter_experiments(
    datasets: List[str] = None,
    seeds: List[int] = None,
    max_iterations: Optional[int] = None,  # None = auto-calculate
    batch_size: int = 50,  # Match config default (will be overridden by config if specified)
    rate_limit_delay: float = None,  # None = auto-select based on tier
    auto_iterations: bool = True,  # Auto-calculate optimal iterations
    paid_tier: bool = False,  # Use paid tier limits (unlimited req/min, 1000 req/day)
    resume: bool = True,
    verbose: bool = True
):
    """
    Run QualSynth experiments using OpenRouter API.
    
    Args:
        datasets: List of datasets to run
        seeds: List of seeds to run
        max_iterations: Maximum iterations per experiment (None = auto-calculate)
        batch_size: Samples per iteration (default 150 for 27B model)
        rate_limit_delay: Seconds between API calls (None = auto based on tier)
        auto_iterations: Auto-calculate optimal iterations per dataset
        paid_tier: Use paid tier (UNLIMITED req/min, 1000 req/day vs FREE: 20 req/min, 50 req/day)
        resume: Skip already completed experiments
        verbose: Print progress
    """
    # Set rate limit based on tier (paid = no limit, free = 20 req/min)
    if rate_limit_delay is None:
        rate_limit_delay = RATE_LIMIT_DELAY_SECONDS_PAID if paid_tier else RATE_LIMIT_DELAY_SECONDS_FREE
    os.environ['OPENROUTER_RATE_LIMIT_DELAY'] = str(rate_limit_delay)
    # Default configuration
    if datasets is None:
        datasets = ['german_credit']
    if seeds is None:
        seeds = [42]
    
    # Output directory for OpenRouter results
    output_dir = project_root / 'results' / 'openrouter'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log directory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print()
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║     OpenRouter Experiment Runner - Gemma 3 27B (Free)                    ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"API Base: {OPENROUTER_BASE_URL}")
    print(f"Model: {OPENROUTER_MODEL}")
    print(f"Output: {output_dir}")
    print()
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Max iterations: {max_iterations if max_iterations else 'auto-calculate'}")
    print(f"Auto iterations: {auto_iterations}")
    print(f"Batch size: {batch_size} samples/iteration (optimized for 27B)")
    if rate_limit_delay > 0:
        print(f"Rate limit: {rate_limit_delay:.1f}s between API calls ({60/rate_limit_delay:.0f} req/min)")
    else:
        print(f"Rate limit: NONE (paid tier - unlimited requests)")
    print()
    
    # Pre-calculate iterations per dataset if auto mode
    dataset_iterations = {}
    if auto_iterations and max_iterations is None:
        tier_name = "PAID" if paid_tier else "FREE"
        print(f"📊 Calculating optimal iterations per dataset ({tier_name} tier)...")
        print()
        for dataset in datasets:
            dataset_iterations[dataset] = calculate_openrouter_iterations(
                dataset_name=dataset,
                batch_size=batch_size,
                rate_limit_delay=rate_limit_delay,
                paid_tier=paid_tier,
                verbose=verbose
            )
    
    # Generate experiment list
    # NOTE: With target-based looping, max_iterations is just a safety limit
    # The workflow will loop until target samples are reached
    experiments = []
    for dataset in datasets:
        for seed in seeds:
            # Get iterations for this dataset (now just a safety limit)
            if max_iterations:
                exp_iterations = max_iterations
            elif dataset in dataset_iterations:
                # Use calculated as baseline, but add buffer since it's now a safety limit
                exp_iterations = max(dataset_iterations[dataset] * 2, 50)  # 2x buffer or at least 50
            else:
                exp_iterations = 100  # High default - loop until target reached
            
            experiments.append({
                'dataset': dataset,
                'method': 'qualsynth',
                'seed': seed,
                'max_iterations': exp_iterations,
                'experiment_id': f"{dataset}_qualsynth_seed{seed}"
            })
    
    total_experiments = len(experiments)
    print(f"Total experiments: {total_experiments}")
    print("=" * 70)
    print()
    
    # Track results
    results_summary = []
    start_time = time.time()
    
    for idx, exp in enumerate(experiments, 1):
        dataset = exp['dataset']
        seed = exp['seed']
        exp_id = exp['experiment_id']
        
        # Check if already completed SUCCESSFULLY AND reached target
        result_file = output_dir / dataset / 'qualsynth' / f'seed{seed}.json'
        if resume and result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    existing_result = json.load(f)
                # Only skip if experiment was successful AND reached target
                if existing_result.get('success', False):
                    # Check if target was reached by looking at validated samples CSV
                    csv_file = output_dir / 'logs' / f'{dataset}_qualsynth_seed{seed}_validated_samples.csv'
                    target_reached = True  # Assume reached unless we can check
                    
                    if csv_file.exists():
                        # Load CSV to check sample count
                        import pandas as pd
                        try:
                            existing_samples = pd.read_csv(csv_file)
                            n_existing = len(existing_samples)
                            
                            # Calculate target (same logic as calculate_openrouter_iterations)
                            from src.qualsynth.data.splitting import load_split
                            split_data = load_split(dataset, seed=seed)
                            y_train = split_data['y_train']
                            class_counts = y_train.value_counts()
                            gap = class_counts.max() - class_counts.min()
                            target = min(gap, 10000)  # Same cap as in calculator
                            
                            if n_existing < target * 0.95:  # Allow 5% tolerance
                                target_reached = False
                                print(f"[{idx}/{total_experiments}] Continuing {exp_id} (only {n_existing}/{target} samples, {n_existing/target*100:.1f}%)")
                        except Exception as e:
                            print(f"[{idx}/{total_experiments}] Could not check target: {e}")
                    
                    if target_reached:
                        print(f"[{idx}/{total_experiments}] Skipping {exp_id} (already completed successfully)")
                        results_summary.append({
                            'experiment_id': exp_id,
                            'status': 'skipped',
                            'reason': 'already_completed'
                        })
                        continue
                else:
                    print(f"[{idx}/{total_experiments}] Re-running {exp_id} (previous run failed)")
            except (json.JSONDecodeError, KeyError, IOError):
                print(f"[{idx}/{total_experiments}] Re-running {exp_id} (invalid result file)")
        
        # Target-based looping: no max_iterations limit (0 = unlimited)
        # The workflow will loop until target samples are reached
        
        print(f"[{idx}/{total_experiments}] Running: {exp_id}")
        print(f"   Dataset: {dataset}")
        print(f"   Seed: {seed}")
        print(f"   Model: {OPENROUTER_MODEL}")
        print(f"   Mode: Target-based (loop until target samples reached)")
        print()
        
        exp_start = time.time()
        
        try:
            # Create experiment runner with OpenRouter settings
            runner = ExperimentRunner(
                output_dir=str(output_dir),
                verbose=verbose
            )
            
            # Run the experiment - use config values (no overrides except model)
            # Config specifies: batch_size=20, temperature=0.7, anchor_selection=stratified
            result = runner.run_experiment(
                dataset_name=dataset,
                method_name='qualsynth',
                seed=seed,
                max_iterations_override=0,  # 0 = no limit, loop until target reached
                model_name_override=OPENROUTER_MODEL
                # batch_size NOT overridden - uses config value (20)
            )
            
            exp_duration = time.time() - exp_start
            
            if result.success:
                print(f"   ✅ SUCCESS in {exp_duration:.1f}s")
                print(f"   Samples generated: {result.n_generated}")
                
                # Extract F1 score from performance_metrics
                f1_score = None
                if result.performance_metrics:
                    # Average F1 across classifiers
                    f1_scores = [m.get('f1', 0) for m in result.performance_metrics.values() if isinstance(m, dict)]
                    if f1_scores:
                        f1_score = sum(f1_scores) / len(f1_scores)
                        print(f"   F1 Score (avg): {f1_score:.4f}")
                
                results_summary.append({
                    'experiment_id': exp_id,
                    'status': 'success',
                    'duration': exp_duration,
                    'n_generated': result.n_generated,
                    'f1': f1_score
                })
            else:
                print(f"   ❌ FAILED: {result.error}")
                results_summary.append({
                    'experiment_id': exp_id,
                    'status': 'failed',
                    'error': str(result.error),
                    'duration': exp_duration
                })
                
        except Exception as e:
            exp_duration = time.time() - exp_start
            print(f"   ❌ ERROR: {e}")
            traceback.print_exc()
            results_summary.append({
                'experiment_id': exp_id,
                'status': 'error',
                'error': str(e),
                'duration': exp_duration
            })
        
        print()
    
    # Summary
    total_duration = time.time() - start_time
    
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results_summary if r['status'] == 'success')
    failed = sum(1 for r in results_summary if r['status'] in ['failed', 'error'])
    skipped = sum(1 for r in results_summary if r['status'] == 'skipped')
    
    print(f"Total: {total_experiments}")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"Total time: {total_duration/60:.1f} minutes")
    print()
    
    # Save summary
    summary_file = output_dir / f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'config': {
                'api_base': OPENROUTER_BASE_URL,
                'model': OPENROUTER_MODEL,
                'datasets': datasets,
                'seeds': seeds,
                'max_iterations': max_iterations
            },
            'results': results_summary,
            'total_duration': total_duration,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    print()
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description='Run QualSynth experiments with OpenRouter API (Gemma 3 27B Free)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single dataset with single seed
  python scripts/run_openrouter_experiments.py --datasets german_credit --seeds 42
  
  # Run multiple seeds
  python scripts/run_openrouter_experiments.py --datasets german_credit --seeds 42 123 456
  
  # Run multiple datasets
  python scripts/run_openrouter_experiments.py --datasets german_credit breast_cancer --seeds 42
  
  # Run all datasets with all seeds
  python scripts/run_openrouter_experiments.py --all
  
  # Custom max iterations
  python scripts/run_openrouter_experiments.py --datasets german_credit --seeds 42 --max-iterations 20
"""
    )
    
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        default=['german_credit'],
        help='Datasets to run (default: german_credit)'
    )
    parser.add_argument(
        '--seeds', 
        nargs='+', 
        type=int, 
        default=[42],
        help='Random seeds (default: 42)'
    )
    parser.add_argument(
        '--max-iterations', 
        type=int, 
        default=None,
        help='Maximum iterations per experiment (default: auto-calculate based on dataset)'
    )
    parser.add_argument(
        '--no-auto-iterations',
        action='store_true',
        help='Disable auto-calculation of iterations (use --max-iterations or default 30)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=150,
        help='Samples per iteration (default: 150, optimized for 27B cloud model)'
    )
    parser.add_argument(
        '--rate-limit-delay', 
        type=float, 
        default=None,
        help='Seconds between API calls (default: auto based on tier - 0.5s paid, 3.0s free)'
    )
    parser.add_argument(
        '--no-rate-limit', 
        action='store_true',
        help='Disable rate limiting (not recommended for free tier)'
    )
    parser.add_argument(
        '--paid-tier',
        action='store_true',
        help='Use paid tier limits (1000 req/day instead of 50). Requires any credit purchase on OpenRouter (e.g. $1).'
    )
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run all datasets with all 10 seeds'
    )
    parser.add_argument(
        '--no-resume', 
        action='store_true',
        help='Re-run even if results exist'
    )
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Handle --all flag
    if args.all:
        datasets = ['german_credit', 'breast_cancer', 'pima_diabetes', 
                    'wine_quality', 'yeast', 'haberman', 'thyroid', 'htru2']
        seeds = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]
    else:
        datasets = args.datasets
        seeds = args.seeds
    
    # Determine rate limit
    rate_limit = 0.0 if args.no_rate_limit else args.rate_limit_delay
    
    # Determine auto iterations
    auto_iterations = not args.no_auto_iterations
    
    # If max_iterations provided, disable auto
    if args.max_iterations is not None:
        auto_iterations = False
    
    # Run experiments
    run_openrouter_experiments(
        datasets=datasets,
        seeds=seeds,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        rate_limit_delay=rate_limit,
        auto_iterations=auto_iterations,
        paid_tier=args.paid_tier,
        resume=not args.no_resume,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()

