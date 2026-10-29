#!/usr/bin/env python3
"""Re-apply the runner's final validation pass to the component-ablation runs.

The original pass used a row hash that treated numerically equal rows as distinct
(signed zeros, integer versus float). For every arm that went through the pass, this
script re-runs it on the stored workflow output exactly as ExperimentRunner._maybe_validate
does, first with the original hash (which must reproduce the archived row counts) and
then with the canonical hash. In --apply mode, runs whose admitted rows change are
rewritten: the admitted rows are stored, classifiers are retrained, and the run JSON is
updated.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.qualsynth.data.splitting import encode_features, load_split  # noqa: E402
from src.qualsynth.evaluation.classifiers import ClassifierPipeline  # noqa: E402
from src.qualsynth.experiments.experiment_runner import ExperimentRunner  # noqa: E402
from src.qualsynth.validation.universal_validator import UniversalValidator  # noqa: E402

ABL = ROOT / "results/ablations/component_3seed"
CANONICAL = UniversalValidator._canonical_row_hashes


def original_hash(frame):
    canonical = frame.copy()
    numeric = canonical.select_dtypes(include=[np.number]).columns
    if len(numeric):
        canonical.loc[:, numeric] = canonical.loc[:, numeric].round(12)
    return pd.util.hash_pandas_object(canonical, index=False).astype(str)


def final_pass(X_gen, X_train, y_train, mode, hasher):
    UniversalValidator._canonical_row_hashes = staticmethod(hasher)
    try:
        validator = UniversalValidator(max_samples=len(X_gen), verbose=False, validation_mode=mode)
        with contextlib.redirect_stdout(io.StringIO()):
            result = validator.validate_and_select(
                X_gen, pd.Series([1] * len(X_gen)), X_train, y_train, method_name="ablation_final_pass")
    finally:
        UniversalValidator._canonical_row_hashes = staticmethod(CANONICAL)
    return result.X_validated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    rows = []
    for path in sorted(ABL.glob("*/qualsynth_component_*/seed*.json")):
        payload = json.loads(path.read_text())
        meta = payload["metadata"]
        if not payload.get("success") or not meta.get("validation_applied"):
            continue
        dataset, variant, seed = path.parts[-3], path.parts[-2], int(path.stem[4:])
        stem = ABL / "logs" / f"{dataset}_{variant}_seed{seed}"
        generated = pd.read_csv(f"{stem}_generated_samples.csv", float_precision="round_trip")
        split = load_split(dataset, seed=seed, split_dir=str(ROOT / "data/splits"))
        X_train = split["X_train"].reset_index(drop=True)
        y_train, _, y_test, _ = ExperimentRunner._normalize_binary_labels(
            pd.Series(split["y_train"]).reset_index(drop=True),
            pd.Series(split["y_val"]).reset_index(drop=True),
            pd.Series(split["y_test"]).reset_index(drop=True))
        columns = list(X_train.columns)
        X_gen = generated.drop(columns=["target"], errors="ignore")[columns]
        if split.get("preprocessor") is not None and (X_gen.dtypes == object).any():
            X_gen = encode_features(X_gen, split["preprocessor"])[columns]
        X_gen = X_gen.reset_index(drop=True)
        mode = meta.get("run_provenance", {}).get("policies", {}).get("validation_mode")
        if mode in (None, "not_applicable"):
            mode = "high_dimensional" if dataset.startswith(("alon", "golub")) else "standard"
        old = final_pass(X_gen, X_train, y_train, mode, original_hash)
        new = final_pass(X_gen, X_train, y_train, mode, CANONICAL)
        row = {"dataset": dataset, "variant": variant, "seed": seed, "generated": len(X_gen),
               "archived_validated": meta.get("n_validated"), "old_pass": len(old), "new_pass": len(new),
               "new_repeats": int(CANONICAL(new).duplicated().sum()),
               "new_train_matches": int(CANONICAL(new).isin(set(CANONICAL(X_train))).sum())}
        # The validator renumbers rows; map admitted rows back to their first stored occurrence.
        gen_hash = CANONICAL(X_gen)
        first = {h: i for i, h in reversed(list(enumerate(gen_hash)))}
        keep = np.array([first[h] for h in CANONICAL(new.reset_index(drop=True))])
        row["index_preserved"] = bool(np.allclose(X_gen.iloc[keep].to_numpy(float), new.to_numpy(float), equal_nan=True))
        rows.append(row)
        if args.apply and len(new) != len(old):
            generated.iloc[keep].to_csv(f"{stem}_validated_samples.csv", index=False)
            X_aug = pd.concat([X_train, new.reset_index(drop=True)], ignore_index=True)
            y_aug = pd.concat([y_train, pd.Series([1] * len(new), name=y_train.name)], ignore_index=True)
            pipeline = ClassifierPipeline(random_state=seed, imbalance_policy="balanced")
            pipeline.train(X_aug, y_aug, verbose=False)
            metrics = pipeline.evaluate(split["X_test"].reset_index(drop=True), y_test,
                                        compute_fairness=False, verbose=False)
            payload["performance_metrics"] = metrics
            payload["avg_performance"] = ExperimentRunner._average_metrics(metrics)
            payload["n_generated"] = len(new)
            meta["n_validated"] = len(new)
            meta["canonical_final_pass"] = {
                "applied": True, "rows_before": len(old), "rows_after": len(new),
                "reason": "final validation pass re-applied with canonical numeric row hashing"}
            path.write_text(json.dumps(payload, indent=2, default=float))
    frame = pd.DataFrame(rows)
    print("runs checked:", len(frame))
    print("old pass reproduces archived counts:", bool((frame.old_pass == frame.archived_validated).all()))
    print("admitted rows map back to stored rows:", bool(frame.index_preserved.all()))
    bad = frame[frame.old_pass != frame.archived_validated]
    if len(bad):
        print(bad.to_string(index=False))
    changed = frame[frame.new_pass != frame.old_pass]
    print(changed.to_string(index=False) if len(changed) else "no runs change")
    print("after new pass: repeats", int(frame.new_repeats.sum()), "training matches", int(frame.new_train_matches.sum()))


if __name__ == "__main__":
    main()
