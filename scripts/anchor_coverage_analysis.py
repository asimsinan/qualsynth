#!/usr/bin/env python3
"""Anchor coverage under three anchor-selection rules (Reviewer 5, point 7).

For each dataset, seed and generation iteration, the anchor set is built from the
training-fold minority rows (encoded representation, as passed to the generator) by
  first_feature  the implemented rule: quantile strata of the first numerical column
  pc1            the same strata and weights on the first principal component
  random         a uniform random sample of the same size
Coverage is measured in the minority-standardised feature space as the mean distance from
each minority row to its nearest anchor, per iteration and cumulatively over iterations,
and as the share of the minority PC1 range spanned by the anchors.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.qualsynth.data.splitting import load_split  # noqa: E402
from src.qualsynth.utils.diversity_maximizer import DiversityMaximizer  # noqa: E402

DATASETS = ["breast_cancer", "german_credit", "haberman", "htru2", "pima_diabetes",
            "thyroid", "wine_quality", "yeast"]
SEEDS = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]
N_ANCHORS, ITERATIONS = 12, 20
WEIGHTS = {1: 0.10, 2: 0.15, 3: 0.50, 4: 0.15, 5: 0.10}


def standardise(frame, ref):
    sd = ref.std(ddof=0).replace(0, 1.0)
    return ((frame - ref.mean()) / sd).to_numpy(float)


def pc1_scores(Z):
    centred = Z - Z.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[0]


def strata_anchors(key, n, iteration):
    """Replicates the implemented stratified allocation on an arbitrary key."""
    strata = pd.qcut(pd.Series(key), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    rng = np.random.RandomState(42 + iteration * 7)
    chosen = []
    for stratum, weight in WEIGHTS.items():
        members = np.flatnonzero(strata.to_numpy() == stratum)
        if len(members):
            k = min(max(1, int(n * weight)), len(members))
            chosen.extend(pd.Series(members).sample(n=k, random_state=rng.randint(0, 10000)).tolist())
    return np.array(chosen)


def nn_distance(Z, anchors_idx):
    A = Z[anchors_idx]
    d = np.sqrt(((Z[:, None, :] - A[None, :, :]) ** 2).sum(axis=2))
    return d.min(axis=1).mean()


def main():
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            split = load_split(dataset, seed=seed, split_dir=str(ROOT / "data/splits"))
            X = split["X_train"].reset_index(drop=True)
            y = pd.Series(split["y_train"]).reset_index(drop=True)
            minority = X[y == y.value_counts().idxmin()].reset_index(drop=True)
            Z = standardise(minority, minority)
            pc1 = pc1_scores(Z)
            dm = DiversityMaximizer()
            dm.config.anchor_rotation_strategy = "stratified"
            dm.fit(minority, categorical_features=minority.select_dtypes(include=["object", "category"]).columns.tolist())
            seen = {"first_feature": set(), "pc1": set(), "random": set()}
            for it in range(1, ITERATIONS + 1):
                ff = dm.select_diverse_anchors(minority, n_anchors=N_ANCHORS, iteration=it)
                ff_idx = ff.index.to_numpy()
                sets = {"first_feature": ff_idx,
                        "pc1": strata_anchors(pc1, N_ANCHORS, it),
                        "random": np.random.RandomState(seed * 1000 + it).choice(len(minority), len(ff_idx), replace=False)}
                for rule, idx in sets.items():
                    seen[rule].update(idx.tolist())
                    span = (pc1[idx].max() - pc1[idx].min()) / (pc1.max() - pc1.min())
                    rows.append({"dataset": dataset, "seed": seed, "iteration": it, "rule": rule,
                                 "n_anchors": len(idx), "nn_distance": nn_distance(Z, idx), "pc1_span": span,
                                 "cumulative_nn_distance": nn_distance(Z, np.array(sorted(seen[rule]))),
                                 "cumulative_anchors": len(seen[rule])})
    frame = pd.DataFrame(rows)
    out = ROOT / "results/analysis/anchor_coverage"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "anchor_coverage_per_iteration.csv", index=False)
    last = frame[frame.iteration == ITERATIONS]
    summary = frame.groupby(["dataset", "rule"]).agg(nn_distance=("nn_distance", "mean"), pc1_span=("pc1_span", "mean"))
    summary = summary.join(last.groupby(["dataset", "rule"]).cumulative_nn_distance.mean())
    summary.to_csv(out / "anchor_coverage_summary.csv")
    print(summary.round(3).unstack("rule").to_string())


if __name__ == "__main__":
    main()
