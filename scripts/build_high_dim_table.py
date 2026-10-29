"""High-dimensional table with the same five methods at every setting.

Metrics use the minority training rows as reference:
standardised Wasserstein, Frobenius correlation distance, duplicates on rows rounded
to six decimals, and range violations. CTGAN and TabDDPM at every setting, and SMOTE
and TabFairGDT at Golub k=200/500, come from results/high_dim_baselines_rerun
(one PyTorch environment); QualSynth and the remaining SMOTE and TabFairGDT runs
come from results/high_dim_extension.
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # split pickles reference src.qualsynth classes
HD = ROOT / "results/high_dim_extension"
NEW = ROOT / "results/high_dim_baselines_rerun"
COMP = ROOT / "results/ablations/component_3seed"
SEEDS = [42, 123, 456]
SETTINGS = ["alon_colon", "alon_colon_k200", "alon_colon_k500", "golub_leukemia", "golub_leukemia_k200", "golub_leukemia_k500"]
METHODS = ["qualsynth", "smote", "ctgan", "tabfairgdt", "tabddpm"]


def paths(ds, m, seed):
    if m == "qualsynth":
        j = HD / ds / "qualsynth" / f"seed{seed}.json"
        c = HD / "logs" / f"{ds}_qualsynth_seed{seed}_generated_samples.csv"
        if not j.exists():
            j = COMP / ds / "qualsynth_component_full" / f"seed{seed}.json"
            c = COMP / "logs" / f"{ds}_qualsynth_component_full_seed{seed}_generated_samples.csv"
        return j, c
    base = NEW if m in ("ctgan", "tabddpm") or ds in ("golub_leukemia_k200", "golub_leukemia_k500") else HD
    return base / ds / m / f"seed{seed}.json", base / "logs" / f"{ds}_{m}_seed{seed}_generated_samples.csv"


def row(ds, m, seed):
    j, c = paths(ds, m, seed)
    d = json.load(open(j))
    if d.get("success") is False:
        raise RuntimeError(f"failed run {j}")
    sp = pickle.load(open(ROOT / "data/splits" / ds / f"split_seed{seed}.pkl", "rb"))
    Xtr, ytr = sp["X_train"], np.asarray(sp["y_train"])
    real = Xtr[ytr == 1].reset_index(drop=True)
    syn = pd.read_csv(c)
    syn = syn.drop(columns=[x for x in ["target"] if x in syn.columns]).select_dtypes(include=[np.number])
    cols = [x for x in real.columns if x in syn.columns]
    real, syn = real[cols], syn[cols]
    W = np.mean([stats.wasserstein_distance(real[x], syn[x]) / (np.std(real[x]) + 1e-12) for x in cols])
    C = np.linalg.norm(real.corr().fillna(0).to_numpy() - syn.corr().fillna(0).to_numpy(), "fro")
    dup = syn.round(6).duplicated().sum() / len(syn)
    lo, hi = real.min(), real.max()
    rv = ((syn.lt(lo - (1e-12 + 1e-9 * lo.abs()), axis=1)) | (syn.gt(hi + (1e-12 + 1e-9 * hi.abs()), axis=1))).any(axis=1).mean()
    return dict(dataset=ds, method=m, seed=seed, W=W, C=C, dup=dup, rv=rv,
                f1=d["avg_performance"]["f1"], auc=d["avg_performance"]["roc_auc"], n_syn=len(syn))


def main():
    rows = [row(ds, m, s) for ds in SETTINGS for m in METHODS for s in SEEDS]
    df = pd.DataFrame(rows)
    out = NEW
    df.to_csv(out / "highdim_all_methods_per_run.csv", index=False)
    g = df.groupby(["dataset", "method"], sort=False)[["W", "C", "dup", "rv", "f1", "auc"]].mean()
    g.to_csv(out / "highdim_all_methods_summary.csv")
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
