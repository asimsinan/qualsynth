"""
TabDDPM baseline adapter.

This adapter bridges the official TabDDPM implementation into the QualSynth
experiment pipeline by:
1. exporting one existing train/val/test split to the upstream `.npy` format,
2. invoking the official training and sampling entry points, and
3. converting the generated samples back into a pandas DataFrame.

Reference:
Kotelnikov et al. (2023). TabDDPM: Modelling Tabular Data with Diffusion Models.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


CAT_MISSING_VALUE = "__nan__"


@dataclass
class TabDDPMResult:
    """Generated samples and metadata returned by TabDDPM."""

    samples: pd.DataFrame
    labels: pd.Series
    n_requested: int
    n_generated: int
    generation_time: float
    workspace: Optional[str] = None


class TabDDPMBaseline:
    """Baseline wrapper around the official TabDDPM repository."""

    def __init__(
        self,
        repo_dir: Optional[str] = None,
        workspace_root: Optional[str] = None,
        model_type: str = "mlp",
        hidden_dims: Sequence[int] = (256, 256),
        dropout: float = 0.0,
        steps: int = 1000,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        sample_batch_size: int = 512,
        num_timesteps: int = 1000,
        gaussian_loss_type: str = "mse",
        scheduler: str = "cosine",
        normalization: str = "quantile",
        device: str = "cpu",
        verbose: bool = False,
        keep_workspace: bool = False,
        random_state: int = 42,
    ) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.repo_dir = Path(repo_dir or project_root / "third_party" / "tab-ddpm")
        self.workspace_root = Path(
            workspace_root or project_root / "results" / "tabddpm_workspaces"
        )
        self.model_type = model_type
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.steps = steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.sample_batch_size = sample_batch_size
        self.num_timesteps = num_timesteps
        self.gaussian_loss_type = gaussian_loss_type
        self.scheduler = scheduler
        self.normalization = normalization
        self.device = self._resolve_device(device)
        self.verbose = verbose
        self.keep_workspace = keep_workspace
        self.random_state = random_state

    def fit_resample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        numerical_features: Sequence[str],
        categorical_features: Sequence[str],
        n_samples: Optional[int] = None,
        dataset_name: str = "dataset",
    ) -> TabDDPMResult:
        """
        Train TabDDPM on one split and generate balancing samples.

        Args:
            X_train, y_train, X_val, y_val, X_test, y_test:
                Raw train/val/test splits.
            numerical_features: Numerical feature names in raw order.
            categorical_features: Categorical feature names in raw order.
            n_samples: Requested number of synthetic samples. If omitted, fills the
                minority class to match the majority class.
            dataset_name: Name used for temporary workspace naming.
        """
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"TabDDPM repository not found at {self.repo_dir}. "
                "Clone the official repo into third_party/tab-ddpm first."
            )

        class_counts = y_train.astype(int).value_counts()
        if class_counts.empty or len(class_counts) < 2:
            raise ValueError("TabDDPM baseline requires at least two classes.")

        target_samples = int(
            n_samples if n_samples is not None else class_counts.max() - class_counts.min()
        )
        if target_samples <= 0:
            return TabDDPMResult(
                samples=pd.DataFrame(columns=X_train.columns),
                labels=pd.Series(dtype=int, name=y_train.name),
                n_requested=0,
                n_generated=0,
                generation_time=0.0,
            )

        self.workspace_root.mkdir(parents=True, exist_ok=True)
        workspace_cm = (
            tempfile.TemporaryDirectory(
                prefix=f"{dataset_name}_seed{self.random_state}_",
                dir=str(self.workspace_root),
            )
            if not self.keep_workspace
            else None
        )
        workspace_path = (
            Path(workspace_cm.name)
            if workspace_cm is not None
            else self.workspace_root
            / f"{dataset_name}_seed{self.random_state}_{int(time.time())}"
        )
        workspace_path.mkdir(parents=True, exist_ok=True)

        try:
            data_dir = workspace_path / "data"
            output_dir = workspace_path / "output"
            data_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            export_meta = self._export_split_dataset(
                data_dir=data_dir,
                X_parts={"train": X_train, "val": X_val, "test": X_test},
                y_parts={"train": y_train, "val": y_val, "test": y_test},
                numerical_features=numerical_features,
                categorical_features=categorical_features,
            )

            train_fn, sample_fn = self._load_upstream_functions()
            model_params = {
                "is_y_cond": True,
                "num_classes": int(pd.Series(y_train).nunique()),
                "rtdl_params": {
                    "d_layers": list(self.hidden_dims),
                    "dropout": float(self.dropout),
                },
            }
            transform_dict = {
                "seed": int(self.random_state),
                "normalization": self.normalization,
                "num_nan_policy": export_meta["num_nan_policy"],
                "cat_nan_policy": export_meta["cat_nan_policy"],
                "cat_min_frequency": None,
                "cat_encoding": None,
                "y_policy": "default",
            }

            start_time = time.time()
            if self.verbose:
                train_fn(
                    parent_dir=str(output_dir),
                    real_data_path=str(data_dir),
                    steps=int(self.steps),
                    lr=float(self.learning_rate),
                    weight_decay=float(self.weight_decay),
                    batch_size=int(self.batch_size),
                    model_type=self.model_type,
                    model_params=model_params,
                    num_timesteps=int(self.num_timesteps),
                    gaussian_loss_type=self.gaussian_loss_type,
                    scheduler=self.scheduler,
                    T_dict=transform_dict,
                    num_numerical_features=len(numerical_features),
                    device=torch.device(self.device),
                    seed=int(self.random_state),
                    change_val=False,
                )
                sample_fn(
                    parent_dir=str(output_dir),
                    real_data_path=str(data_dir),
                    batch_size=int(self.sample_batch_size),
                    num_samples=int(target_samples),
                    model_type=self.model_type,
                    model_params=model_params,
                    model_path=str(output_dir / "model.pt"),
                    num_timesteps=int(self.num_timesteps),
                    gaussian_loss_type=self.gaussian_loss_type,
                    scheduler=self.scheduler,
                    T_dict=transform_dict,
                    num_numerical_features=len(numerical_features),
                    disbalance="fill",
                    device=torch.device(self.device),
                    seed=int(self.random_state),
                    change_val=False,
                )
            else:
                with open(output_dir / "tabddpm_runtime.log", "w", encoding="utf-8") as log_handle:
                    with redirect_stdout(log_handle), redirect_stderr(log_handle):
                        train_fn(
                            parent_dir=str(output_dir),
                            real_data_path=str(data_dir),
                            steps=int(self.steps),
                            lr=float(self.learning_rate),
                            weight_decay=float(self.weight_decay),
                            batch_size=int(self.batch_size),
                            model_type=self.model_type,
                            model_params=model_params,
                            num_timesteps=int(self.num_timesteps),
                            gaussian_loss_type=self.gaussian_loss_type,
                            scheduler=self.scheduler,
                            T_dict=transform_dict,
                            num_numerical_features=len(numerical_features),
                            device=torch.device(self.device),
                            seed=int(self.random_state),
                            change_val=False,
                        )
                        sample_fn(
                            parent_dir=str(output_dir),
                            real_data_path=str(data_dir),
                            batch_size=int(self.sample_batch_size),
                            num_samples=int(target_samples),
                            model_type=self.model_type,
                            model_params=model_params,
                            model_path=str(output_dir / "model.pt"),
                            num_timesteps=int(self.num_timesteps),
                            gaussian_loss_type=self.gaussian_loss_type,
                            scheduler=self.scheduler,
                            T_dict=transform_dict,
                            num_numerical_features=len(numerical_features),
                            disbalance="fill",
                            device=torch.device(self.device),
                            seed=int(self.random_state),
                            change_val=False,
                        )
            generation_time = time.time() - start_time

            synthetic_X, synthetic_y = self._load_generated_samples(
                output_dir=output_dir,
                feature_order=list(X_train.columns),
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                label_name=y_train.name,
            )
            if len(synthetic_X) > target_samples:
                synthetic_X = synthetic_X.iloc[:target_samples].reset_index(drop=True)
                synthetic_y = synthetic_y.iloc[:target_samples].reset_index(drop=True)

            return TabDDPMResult(
                samples=synthetic_X,
                labels=synthetic_y.astype(int),
                n_requested=target_samples,
                n_generated=len(synthetic_X),
                generation_time=generation_time,
                workspace=str(workspace_path) if self.keep_workspace else None,
            )
        finally:
            if workspace_cm is not None:
                workspace_cm.cleanup()
            elif not self.keep_workspace and workspace_path.exists():
                shutil.rmtree(workspace_path, ignore_errors=True)

    def _resolve_device(self, requested: str) -> str:
        requested = (requested or "cpu").strip().lower()
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _export_split_dataset(
        self,
        data_dir: Path,
        X_parts: Dict[str, pd.DataFrame],
        y_parts: Dict[str, pd.Series],
        numerical_features: Sequence[str],
        categorical_features: Sequence[str],
    ) -> Dict[str, Optional[str]]:
        has_num_nans = False
        has_cat_nans = False

        for split_name, X_part in X_parts.items():
            if numerical_features:
                X_num = X_part[list(numerical_features)].copy()
                X_num = X_num.apply(pd.to_numeric, errors="coerce")
                has_num_nans = has_num_nans or bool(X_num.isna().any().any())
                np.save(
                    data_dir / f"X_num_{split_name}.npy",
                    X_num.to_numpy(dtype=float),
                    allow_pickle=True,
                )

            if categorical_features:
                X_cat = X_part[list(categorical_features)].copy()
                has_cat_nans = has_cat_nans or bool(X_cat.isna().any().any())
                X_cat = X_cat.fillna(CAT_MISSING_VALUE).astype(str)
                np.save(
                    data_dir / f"X_cat_{split_name}.npy",
                    X_cat.to_numpy(dtype=object),
                    allow_pickle=True,
                )

            np.save(
                data_dir / f"y_{split_name}.npy",
                y_parts[split_name].astype(int).to_numpy(),
                allow_pickle=True,
            )

        info = {
            "task_type": "binclass",
            "n_classes": int(pd.Series(y_parts["train"]).nunique()),
        }
        with open(data_dir / "info.json", "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)

        return {
            "num_nan_policy": "mean" if has_num_nans else None,
            "cat_nan_policy": "most_frequent" if has_cat_nans else None,
        }

    @contextmanager
    def _upstream_import_context(self) -> Iterable[None]:
        repo_root = str(self.repo_dir)
        scripts_root = str(self.repo_dir / "scripts")
        injected_paths = [scripts_root, repo_root]
        previous_modules = {
            name: sys.modules.get(name)
            for name in ("train", "sample", "utils_train", "lib", "tab_ddpm")
        }

        for name in previous_modules:
            sys.modules.pop(name, None)
        for path in reversed(injected_paths):
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            yield
        finally:
            for path in injected_paths:
                while path in sys.path:
                    sys.path.remove(path)
            for name, module in previous_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def _load_upstream_functions(self):
        with self._upstream_import_context():
            try:
                train_module = importlib.import_module("train")
                sample_module = importlib.import_module("sample")
            except Exception as exc:  # pragma: no cover - import surface depends on env
                raise ImportError(
                    "Failed to import the upstream TabDDPM runtime. "
                    "Ensure third_party/tab-ddpm is present and its runtime "
                    "dependencies are installed (torch, category-encoders, "
                    "icecream, rtdl, tomli, and the upstream helper packages)."
                ) from exc
            return train_module.train, sample_module.sample

    def _load_generated_samples(
        self,
        output_dir: Path,
        feature_order: Sequence[str],
        numerical_features: Sequence[str],
        categorical_features: Sequence[str],
        label_name: Optional[str],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        frames = []

        if numerical_features:
            X_num = np.load(output_dir / "X_num_train.npy", allow_pickle=True)
            X_num = np.asarray(X_num)
            if X_num.ndim == 1:
                X_num = X_num.reshape(-1, 1)
            frames.append(pd.DataFrame(X_num, columns=list(numerical_features)))

        if categorical_features:
            X_cat = np.load(output_dir / "X_cat_train.npy", allow_pickle=True)
            X_cat = np.asarray(X_cat, dtype=object)
            if X_cat.ndim == 1:
                X_cat = X_cat.reshape(-1, 1)
            frames.append(pd.DataFrame(X_cat, columns=list(categorical_features)))

        if not frames:
            synthetic_X = pd.DataFrame(columns=list(feature_order))
        else:
            synthetic_X = pd.concat(frames, axis=1)
            synthetic_X = synthetic_X.loc[:, list(feature_order)]

        synthetic_y = pd.Series(
            np.load(output_dir / "y_train.npy", allow_pickle=True).astype(int),
            name=label_name,
        )
        return synthetic_X.reset_index(drop=True), synthetic_y.reset_index(drop=True)

    def get_params(self) -> Dict[str, Any]:
        """Return the configured baseline hyperparameters."""
        return {
            "method": "TabDDPM",
            "repo_dir": str(self.repo_dir),
            "workspace_root": str(self.workspace_root),
            "model_type": self.model_type,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "sample_batch_size": self.sample_batch_size,
            "num_timesteps": self.num_timesteps,
            "gaussian_loss_type": self.gaussian_loss_type,
            "scheduler": self.scheduler,
            "normalization": self.normalization,
            "device": self.device,
            "verbose": self.verbose,
            "keep_workspace": self.keep_workspace,
            "random_state": self.random_state,
        }
