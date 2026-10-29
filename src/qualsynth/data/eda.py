"""
Exploratory Data Analysis for Qualsynth datasets.

Analyzes:
- Summary statistics
- Class imbalance
- Feature distributions
- Protected attribute analysis
- Overlap regions between classes
- Underrepresented subgroups
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from .preprocessing import load_dataset
    from .splitting import load_split
except ImportError:
    from preprocessing import load_dataset
    from splitting import load_split


def analyze_dataset(dataset_name: str, data_dir: str = "data/raw") -> Dict:
    """
    Perform comprehensive EDA on a dataset.
    
    Args:
        dataset_name: Name of dataset
        data_dir: Directory containing raw data
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*70}")
    print(f"EDA: {dataset_name.upper()}")
    print('='*70)
    
    # Load data
    X, y, info = load_dataset(dataset_name, data_dir)
    
    results = {
        'dataset': dataset_name,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_info': info
    }
    
    # 1. Class distribution
    print("\n1. CLASS DISTRIBUTION")
    print("-" * 70)
    class_counts = y.value_counts()
    results['class_distribution'] = class_counts.to_dict()
    results['imbalance_ratio'] = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
    
    print(f"Class 0 (majority): {class_counts[0]:,} samples ({class_counts[0]/len(y)*100:.1f}%)")
    print(f"Class 1 (minority): {class_counts[1]:,} samples ({class_counts[1]/len(y)*100:.1f}%)")
    print(f"Imbalance Ratio: {results['imbalance_ratio']:.2f}:1")
    
    # 2. Feature statistics
    print("\n2. FEATURE STATISTICS")
    print("-" * 70)
    
    # Numerical features
    if info['numerical_features']:
        print(f"\nNumerical features ({len(info['numerical_features'])}):")
        num_stats = X[info['numerical_features']].describe()
        print(num_stats.round(3))
        results['numerical_stats'] = num_stats.to_dict()
    
    # Categorical features
    if info['categorical_features']:
        print(f"\nCategorical features ({len(info['categorical_features'])}):")
        cat_stats = {}
        for col in info['categorical_features'][:5]:  # Show first 5
            n_unique = X[col].nunique()
            cat_stats[col] = {
                'n_unique': n_unique,
                'top_values': X[col].value_counts().head(3).to_dict()
            }
            print(f"  {col}: {n_unique} unique values")
        results['categorical_stats'] = cat_stats
    
    # 3. Protected attribute analysis
    print("\n3. PROTECTED ATTRIBUTE ANALYSIS")
    print("-" * 70)
    
    protected_analysis = {}
    for attr in info['protected_attributes']:
        if attr in X.columns:
            print(f"\n{attr}:")
            
            # Distribution
            attr_dist = X[attr].value_counts()
            print(f"  Distribution: {attr_dist.to_dict()}")
            
            # Class distribution by protected attribute
            combined = pd.DataFrame({'attr': X[attr], 'target': y})
            cross_tab = pd.crosstab(combined['attr'], combined['target'], normalize='index')
            print(f"  Class 1 rate by {attr}:")
            for idx in cross_tab.index:
                rate = cross_tab.loc[idx, 1] if 1 in cross_tab.columns else 0
                print(f"    {attr}={idx}: {rate:.2%}")
            
            protected_analysis[attr] = {
                'distribution': attr_dist.to_dict(),
                'class_rates': cross_tab.to_dict() if not cross_tab.empty else {}
            }
    
    results['protected_analysis'] = protected_analysis
    
    # 4. Overlap analysis
    print("\n4. OVERLAP ANALYSIS")
    print("-" * 70)
    
    overlap_scores = {}
    for col in X.columns[:10]:  # Analyze first 10 features
        class0_vals = X[y == 0][col]
        class1_vals = X[y == 1][col]
        
        # Overlap as intersection of ranges
        if X[col].dtype in ['float64', 'int64']:
            min0, max0 = class0_vals.min(), class0_vals.max()
            min1, max1 = class1_vals.min(), class1_vals.max()
            
            overlap_min = max(min0, min1)
            overlap_max = min(max0, max1)
            
            if overlap_max > overlap_min:
                overlap_range = overlap_max - overlap_min
                total_range = max(max0, max1) - min(min0, min1)
                overlap_pct = (overlap_range / total_range) * 100 if total_range > 0 else 0
                overlap_scores[col] = overlap_pct
    
    if overlap_scores:
        avg_overlap = np.mean(list(overlap_scores.values()))
        print(f"Average feature overlap: {avg_overlap:.1f}%")
        print(f"High overlap features (>80%):")
        for col, score in sorted(overlap_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {col}: {score:.1f}%")
        results['overlap_scores'] = overlap_scores
        results['avg_overlap'] = avg_overlap
    
    # 5. Underrepresented subgroups
    print("\n5. UNDERREPRESENTED SUBGROUPS")
    print("-" * 70)
    
    minority_samples = X[y == 1]
    from sklearn.neighbors import NearestNeighbors
    
    if len(minority_samples) > 10:
        # Only use numerical features for distance calculation
        numerical_cols = [col for col in info['numerical_features'] if col in minority_samples.columns]
        if numerical_cols:
            minority_numerical = minority_samples[numerical_cols]
        else:
            minority_numerical = minority_samples
        
        nbrs = NearestNeighbors(n_neighbors=min(10, len(minority_samples)), metric='euclidean')
        nbrs.fit(minority_numerical)
        distances, _ = nbrs.kneighbors(minority_numerical)
        
        # Average distance to 10 nearest neighbors
        avg_distances = distances.mean(axis=1)
        
        # Identify sparse regions (top 20% by distance)
        threshold = np.percentile(avg_distances, 80)
        sparse_samples = (avg_distances > threshold).sum()
        
        print(f"Minority class samples: {len(minority_samples):,}")
        print(f"Samples in sparse regions (>80th percentile): {sparse_samples:,} ({sparse_samples/len(minority_samples)*100:.1f}%)")
        print(f"Average neighbor distance: {avg_distances.mean():.3f} ± {avg_distances.std():.3f}")
        
        results['sparse_analysis'] = {
            'n_minority': len(minority_samples),
            'n_sparse': int(sparse_samples),
            'sparse_pct': float(sparse_samples/len(minority_samples)*100),
            'avg_neighbor_dist': float(avg_distances.mean())
        }
    
    return results


def create_visualizations(dataset_name: str, data_dir: str = "data/raw", output_dir: str = "figures"):
    """
    Create EDA visualizations.
    
    Args:
        dataset_name: Name of dataset
        data_dir: Directory containing raw data
        output_dir: Directory to save figures
    """
    output_path = Path(output_dir) / "eda"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y, info = load_dataset(dataset_name, data_dir)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    class_counts = y.value_counts()
    ax.bar(['Class 0\n(Majority)', 'Class 1\n(Minority)'], class_counts.values, color=['#3498db', '#e74c3c'])
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'{dataset_name.upper()}: Class Distribution')
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + max(class_counts.values)*0.02, f'{v:,}\n({v/len(y)*100:.1f}%)', 
                ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path / f"{dataset_name}_class_dist.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Feature distributions (first 6 numerical features)
    if info['numerical_features']:
        n_features = min(6, len(info['numerical_features']))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(info['numerical_features'][:n_features]):
            ax = axes[i]
            
            # Plot distributions for both classes
            X[y == 0][col].hist(ax=ax, alpha=0.5, bins=30, label='Class 0', color='#3498db')
            X[y == 1][col].hist(ax=ax, alpha=0.5, bins=30, label='Class 1', color='#e74c3c')
            
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.set_title(f'Distribution: {col}')
        
        # Hide unused subplots
        for i in range(n_features, 6):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / f"{dataset_name}_feature_dist.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n✓ Visualizations saved to: {output_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "figures"
    
    datasets = ['german_credit', 'thyroid']
    
    print("="*70)
    print("Qualsynth: Exploratory Data Analysis")
    print("="*70)
    
    all_results = {}
    
    for dataset in datasets:
        results = analyze_dataset(dataset, str(data_dir))
        all_results[dataset] = results
        
        # Create visualizations
        create_visualizations(dataset, str(data_dir), str(output_dir))
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    summary_df = pd.DataFrame({
        'Dataset': datasets,
        'Samples': [all_results[d]['n_samples'] for d in datasets],
        'Features': [all_results[d]['n_features'] for d in datasets],
        'Imbalance Ratio': [f"{all_results[d]['imbalance_ratio']:.2f}:1" for d in datasets],
        'Minority %': [f"{all_results[d]['class_distribution'][1]/all_results[d]['n_samples']*100:.1f}%" for d in datasets]
    })
    
    print("\n", summary_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ EDA Complete!")
    print("="*70)

