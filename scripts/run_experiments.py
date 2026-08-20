"""
Run Multiple Experiments

This script runs a batch of experiments based on an experiment configuration.
"""

# CRITICAL: Set environment variables BEFORE any imports
# This prevents PyTorch from using MPS (Apple GPU) which causes segfaults
import os
os.environ['PYTORCH_MPS_METAL'] = '0'  # KEY FIX: Disable MPS Metal backend
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
os.environ['MPS_DISABLE'] = '1'  # Additional MPS disable

# CRITICAL: Disable ALL parallel processing to prevent process explosion
# Setting n_jobs=1 in classifiers + these env vars = no worker processes
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Force single worker (was 4)
os.environ['OMP_NUM_THREADS'] = '1'  # Fix OpenMP pthread_mutex_init error
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Limit OpenBLAS threads
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # Limit NumExpr threads
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Allow duplicate OpenMP libraries

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.qualsynth.experiments.experiment_runner import ExperimentRunner, ExperimentResult
from src.qualsynth.utils.config_loader import ConfigLoader


def run_experiments(
    experiment_name: str,
    methods: List[str] = None,
    datasets: List[str] = None,
    seeds: List[int] = None,
    output_dir: str = None,
    resume: bool = True,
    verbose: bool = True,
    max_iterations: Optional[int] = None,
    batch_size: Optional[int] = None,
    model_name: Optional[str] = None,
    enable_universal_validation: bool = False,
    validation_mode: Optional[str] = None,
    classifier_imbalance_policy: Optional[str] = None,
):
    """
    Run multiple experiments.
    
    Args:
        experiment_name: Name of experiment configuration
        methods: List of methods to run (None = all from config)
        datasets: List of datasets to run (None = all from config)
        seeds: List of seeds to run (None = all from config)
        output_dir: Output directory for results
        resume: Whether to skip already completed experiments
        verbose: Whether to print progress
        max_iterations: Override max_iterations for Qualsynth methods (None = use config)
        model_name: Override LLM model name (None = use config or env var)
    """
    # Load experiment configuration
    config_loader = ConfigLoader()
    exp_config = config_loader.load_experiment_config(experiment_name)
    config_loader.validate_experiment_config(exp_config)
    execution_config = exp_config.execution or {}
    if not enable_universal_validation:
        enable_universal_validation = bool(execution_config.get("enable_universal_validation", False))
    if validation_mode is None:
        validation_mode = str(execution_config.get("validation_mode", "standard"))
    if classifier_imbalance_policy is None:
        classifier_imbalance_policy = str(
            execution_config.get("classifier_imbalance_policy", "balanced")
        )
    
    # Override with command-line arguments
    if datasets is None:
        datasets = exp_config.datasets
    if seeds is None:
        seeds = exp_config.seeds
    if methods is None:
        methods = []
        for method_list in exp_config.methods.values():
            methods.extend(method_list)
    methods = list(dict.fromkeys(methods))

    if output_dir is None and exp_config.output and exp_config.output.get('base_dir'):
        output_dir = exp_config.output['base_dir']
    
    # Generate experiment matrix
    experiments = []
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                exp_dict = {
                    'dataset': dataset,
                    'method': method,
                    'seed': seed,
                    'experiment_id': f"{dataset}_{method}_seed{seed}"
                }
                # Add max_iterations override if provided
                if max_iterations is not None:
                    exp_dict['max_iterations_override'] = max_iterations
                # Add model_name override if provided
                if model_name is not None:
                    exp_dict['model_name_override'] = model_name
                experiments.append(exp_dict)
    
    print("="*80)
    print(f"Qualsynth Experiment Runner")
    print("="*80)
    print(f"\nExperiment: {experiment_name}")
    print(f"Datasets: {datasets}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    if max_iterations is not None:
        print(f"Max iterations override: {max_iterations}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Resume mode: {resume}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize runner
    runner = ExperimentRunner(
        output_dir=output_dir,
        verbose=verbose,
        enable_universal_validation=enable_universal_validation,
        validation_mode=validation_mode,
        classifier_imbalance_policy=classifier_imbalance_policy,
        run_context={"experiment_name": experiment_name},
    )
    
    # Track results
    results = []
    completed = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    # Run experiments
    for i, exp in enumerate(experiments, 1):
        print(f"\n\n{'='*80}")
        print(f"Experiment {i}/{len(experiments)}")
        print(f"{'='*80}")
        
        # Check if already completed (resume mode)
        if resume:
            result_file = (
                runner.output_dir / exp['dataset'] / exp['method'] / 
                f"seed{exp['seed']}.json"
            )
            if result_file.exists():
                # Check if the experiment was actually successful
                try:
                    with open(result_file, 'r') as f:
                        existing_result = json.load(f)
                    # Only skip if experiment was successful
                    if existing_result.get('success', False):
                        print(f"⏭️  Skipping (already completed successfully): {exp['experiment_id']}")
                        skipped += 1
                        continue
                    else:
                        print(f"🔄 Re-running failed experiment: {exp['experiment_id']}")
                except (json.JSONDecodeError, KeyError, IOError):
                    print(f"🔄 Re-running (invalid result file): {exp['experiment_id']}")
        
        # Run experiment
        try:
            result = runner.run_experiment(
                dataset_name=exp['dataset'],
                method_name=exp['method'],
                seed=exp['seed'],
                save_results=True,
                max_iterations_override=exp.get('max_iterations_override'),
                model_name_override=exp.get('model_name_override'),
                batch_size_override=batch_size
            )
            
            results.append(result)
            
            if result.success:
                completed += 1
            else:
                failed += 1
        
        except Exception as e:
            print(f"\n❌ Experiment failed with exception: {str(e)}")
            failed += 1
        
        # Print progress
        elapsed = time.time() - start_time
        avg_time = elapsed / (i - skipped) if (i - skipped) > 0 else 0
        remaining = avg_time * (len(experiments) - i)
        
        print(f"\nProgress: {i}/{len(experiments)} experiments")
        print(f"  Completed: {completed}")
        print(f"  Failed: {failed}")
        print(f"  Skipped: {skipped}")
        print(f"  Elapsed: {elapsed/60:.1f} min")
        print(f"  Estimated remaining: {remaining/60:.1f} min")
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n\n" + "="*80)
    print("EXPERIMENT BATCH COMPLETE")
    print("="*80)
    print(f"\nTotal experiments: {len(experiments)}")
    print(f"  ✅ Completed: {completed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    if completed + failed > 0:
        print(f"Average time per experiment: {total_time/(completed+failed):.1f} seconds")
    print(f"\nResults saved to: {runner.output_dir}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Save summary
    summary = {
        'experiment_name': experiment_name,
        'datasets': datasets,
        'methods': methods,
        'seeds': seeds,
        'classifier_imbalance_policy_default': classifier_imbalance_policy,
        'total_experiments': len(experiments),
        'completed': completed,
        'failed': failed,
        'skipped': skipped,
        'total_time_seconds': total_time,
        'avg_time_per_experiment': total_time / (completed + failed) if (completed + failed) > 0 else 0,
        'completed_result_totals': {
            'generation_time_seconds': sum(float(getattr(r, 'generation_time', 0.0) or 0.0) for r in results if r.success),
            'training_time_seconds': sum(float(getattr(r, 'training_time', 0.0) or 0.0) for r in results if r.success),
            'evaluation_time_seconds': sum(float(getattr(r, 'evaluation_time', 0.0) or 0.0) for r in results if r.success),
            'generation_cost_usd': sum(float(getattr(r, 'generation_cost', 0.0) or 0.0) for r in results if r.success),
            'llm_calls': sum(int(getattr(r, 'llm_calls', 0) or 0) for r in results if r.success),
            'prompt_tokens': sum(int(getattr(r, 'prompt_tokens', 0) or 0) for r in results if r.success),
            'completion_tokens': sum(int(getattr(r, 'completion_tokens', 0) or 0) for r in results if r.success),
            'total_tokens': sum(int(getattr(r, 'total_tokens', 0) or 0) for r in results if r.success),
            'generated_samples': sum(int(getattr(r, 'n_generated', 0) or 0) for r in results if r.success),
        },
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = runner.output_dir / f"{experiment_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run Qualsynth experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments from main_experiments config
  python scripts/run_experiments.py main_experiments
  
  # Run specific methods
  python scripts/run_experiments.py main_experiments --methods smote ctgan qualsynth
  
  # Run specific dataset
  python scripts/run_experiments.py main_experiments --datasets thyroid
  
  # Run specific seed
  python scripts/run_experiments.py main_experiments --seeds 42
  
  # Override max iterations (use predictor's recommendation)
  python scripts/run_experiments.py main_experiments --max-iterations 12
  
  # Override model (use M4-optimized model)
  python scripts/run_experiments.py main_experiments --model gemma3-m4
  
  # Don't resume (rerun all)
  python scripts/run_experiments.py main_experiments --no-resume
        """
    )
    
    parser.add_argument(
        'experiment',
        type=str,
        help='Name of experiment configuration (e.g., main_experiments, ablation_study)'
    )
    
    parser.add_argument(
        '--methods',
        nargs='+',
        default=None,
        help='Methods to run (default: all from config)'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='Datasets to run (default: all from config)'
    )
    
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=None,
        help='Seeds to run (default: all from config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Rerun all experiments (don\'t skip completed ones)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='Override max_iterations for Qualsynth methods (e.g., --max-iterations 12)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size for Qualsynth methods'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Override LLM model name (e.g., --model gemma3-m4, --model gpt-4)'
    )
    
    parser.add_argument(
        '--enable-universal-validation',
        action='store_true',
        help='Enable universal validation for all methods (fair comparison)'
    )

    parser.add_argument(
        '--validation-mode',
        type=str,
        default=None,
        choices=['standard', 'auto', 'high_dimensional'],
        help='Universal validation mode (default: experiment config or standard)'
    )
    parser.add_argument(
        '--classifier-imbalance-policy',
        choices=['balanced', 'none'],
        default=None,
        help='Downstream classifier weighting (default: experiment config or balanced)'
    )
    
    args = parser.parse_args()
    
    # Run experiments
    run_experiments(
        experiment_name=args.experiment,
        methods=args.methods,
        datasets=args.datasets,
        seeds=args.seeds,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        verbose=not args.quiet,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        model_name=args.model,
        enable_universal_validation=args.enable_universal_validation,
        validation_mode=args.validation_mode,
        classifier_imbalance_policy=args.classifier_imbalance_policy,
    )


if __name__ == "__main__":
    main()
