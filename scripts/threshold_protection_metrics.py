#!/usr/bin/env python3
"""Protection metrics for the validation-threshold study (Reviewer 5, point 4).

For each admission condition and candidate pool, the selected rows are assessed for
  class overlap        share whose nearest training-fold row (fold-standardised Euclidean
                       distance) belongs to the majority class
  joint implausibility share whose squared Mahalanobis distance to the training-fold
                       minority class (Ledoit-Wolf covariance) exceeds the chi-square 0.999
                       quantile, i.e. implausible combinations of individually plausible values
The same quantities for the real minority training rows (leave-one-out for the
nearest-neighbour measure) are reported as a reference.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.qualsynth.data.splitting import load_split  # noqa: E402

T = ROOT / "results/threshold_sensitivity"
CONDITIONS = ["std_3", "std_4", "std_5", "historical_preclip_std4", "percentile_0_99",
              "percentile_0_995", "percentile_0_999", "calibrated_z95", "no_statistical_filter"]


def main():
    manifest = pd.read_csv(T / "per_run_metrics.csv")[["dataset", "seed"]].drop_duplicates()
    rows = []
    for dataset, seed in manifest.itertuples(index=False):
        split = load_split(dataset, seed=int(seed), split_dir=str(ROOT / "data/splits"))
        X = split["X_train"].reset_index(drop=True).astype(float)
        y = pd.Series(split["y_train"]).reset_index(drop=True)
        is_min = (y == y.value_counts().idxmin()).to_numpy()
        mu, sd = X.mean(), X.std(ddof=0).replace(0, 1.0)
        Zx = ((X - mu) / sd).to_numpy()
        nn = NearestNeighbors(n_neighbors=2).fit(Zx)
        minority = X[is_min]
        active = minority.columns[minority.std(ddof=0) > 0]
        lw = LedoitWolf().fit(minority[active].to_numpy())
        prec, centre = lw.precision_, lw.location_
        cutoff = chi2.ppf(0.999, df=len(active))

        def implausible(frame):
            d = frame[active].to_numpy(float) - centre
            return float(np.mean(np.einsum("ij,jk,ik->i", d, prec, d) > cutoff))

        _, idx = nn.kneighbors(Zx[is_min], n_neighbors=2)
        rows.append({"dataset": dataset, "seed": seed, "condition": "real_minority",
                     "rows": int(is_min.sum()), "majority_neighbour": float(np.mean(~is_min[idx[:, 1]])),
                     "implausible": implausible(minority)})
        for condition in CONDITIONS:
            path = T / "selected_candidates" / f"{dataset}_seed{seed}_{condition}.csv"
            sel = pd.read_csv(path)
            if sel.empty:
                continue
            sel = sel[X.columns].astype(float)
            Zs = ((sel - mu) / sd).to_numpy()
            _, near = nn.kneighbors(Zs, n_neighbors=1)
            rows.append({"dataset": dataset, "seed": seed, "condition": condition, "rows": len(sel),
                         "majority_neighbour": float(np.mean(~is_min[near[:, 0]])),
                         "implausible": implausible(sel)})
    frame = pd.DataFrame(rows)
    frame.to_csv(T / "protection_metrics_per_pool.csv", index=False)
    summary = frame.groupby("condition")[["majority_neighbour", "implausible"]].mean()
    summary.to_csv(T / "protection_metrics_summary.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
