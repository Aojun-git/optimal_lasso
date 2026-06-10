"""Data generation, loading, preprocessing, and cache helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DatasetBundle:
    """Common in-memory representation for every experiment dataset."""

    name: str
    A_train: np.ndarray
    b_train: np.ndarray
    A_val: np.ndarray | None = None
    b_val: np.ndarray | None = None
    A_test: np.ndarray | None = None
    b_test: np.ndarray | None = None
    x_star: np.ndarray | None = None
    support: np.ndarray | None = None
    feature_names: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.A_train.ndim != 2:
            raise ValueError("A_train must be a two-dimensional array")
        if self.b_train.shape != (self.A_train.shape[0],):
            raise ValueError("b_train length must match A_train rows")

        if (self.A_val is None) != (self.b_val is None):
            raise ValueError("A_val and b_val must either both exist or both be None")
        if self.A_val is not None:
            if self.A_val.ndim != 2:
                raise ValueError("A_val must be a two-dimensional array")
            if self.A_val.shape[1] != self.A_train.shape[1]:
                raise ValueError("train and validation feature counts must match")
            if self.b_val.shape != (self.A_val.shape[0],):
                raise ValueError("b_val length must match A_val rows")

        if (self.A_test is None) != (self.b_test is None):
            raise ValueError("A_test and b_test must either both exist or both be None")
        if self.A_test is not None:
            if self.A_test.ndim != 2:
                raise ValueError("A_test must be a two-dimensional array")
            if self.A_test.shape[1] != self.A_train.shape[1]:
                raise ValueError("train and test feature counts must match")
            if self.b_test.shape != (self.A_test.shape[0],):
                raise ValueError("b_test length must match A_test rows")

        n_features = self.A_train.shape[1]
        if self.x_star is not None and self.x_star.shape != (n_features,):
            raise ValueError("x_star length must match the feature count")
        if self.support is not None:
            if np.any(self.support < 0) or np.any(self.support >= n_features):
                raise ValueError("support contains an invalid feature index")
        if self.feature_names is not None and len(self.feature_names) != n_features:
            raise ValueError("feature_names length must match the feature count")


def _standardize_design(A: np.ndarray) -> np.ndarray:
    centered = A - A.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    norms[norms == 0.0] = 1.0
    return centered / norms


def make_synthetic_lasso(
    *,
    m: int = 128,
    n: int = 256,
    k: int = 25,
    noise: float = 0.01,
    corr: float = 0.0,
    seed: int = 0,
    name: str = "synthetic",
) -> DatasetBundle:
    """Generate a reproducible sparse linear-regression problem."""
    if not 0 < k <= n:
        raise ValueError("k must satisfy 0 < k <= n")
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if not 0.0 <= corr < 1.0:
        raise ValueError("corr must satisfy 0 <= corr < 1")

    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    if corr > 0.0:
        innovation_scale = np.sqrt(1.0 - corr**2)
        for column in range(1, n):
            A[:, column] = (
                corr * A[:, column - 1] + innovation_scale * A[:, column]
            )
    A = _standardize_design(A)

    support = np.sort(rng.choice(n, size=k, replace=False))
    x_star = np.zeros(n)
    x_star[support] = rng.normal(size=k)

    noise_vector = rng.normal(size=m)
    noise_vector -= noise_vector.mean()
    b = A @ x_star + noise * noise_vector

    dataset = DatasetBundle(
        name=name,
        A_train=A,
        b_train=b,
        x_star=x_star,
        support=support,
        metadata={
            "kind": "synthetic",
            "m": m,
            "n": n,
            "k": k,
            "noise": noise,
            "corr": corr,
            "seed": seed,
        },
    )
    dataset.validate()
    return dataset


def load_diabetes_dataset(
    *,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 0,
    name: str = "diabetes",
) -> DatasetBundle:
    """Load Diabetes from scikit-learn and apply train-only standardization."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    diabetes = load_diabetes()
    if validation_size <= 0.0 or test_size <= 0.0:
        raise ValueError("validation_size and test_size must be positive")
    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1")

    A_train_val, A_test, b_train_val, b_test = train_test_split(
        diabetes.data,
        diabetes.target,
        test_size=test_size,
        random_state=seed,
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    A_train, A_val, b_train, b_val = train_test_split(
        A_train_val,
        b_train_val,
        test_size=relative_validation_size,
        random_state=seed,
    )

    scaler = StandardScaler()
    A_train = scaler.fit_transform(A_train)
    A_val = scaler.transform(A_val)
    A_test = scaler.transform(A_test)
    target_mean = float(b_train.mean())
    b_train = b_train - target_mean
    b_val = b_val - target_mean
    b_test = b_test - target_mean

    dataset = DatasetBundle(
        name=name,
        A_train=A_train,
        b_train=b_train,
        A_val=A_val,
        b_val=b_val,
        A_test=A_test,
        b_test=b_test,
        feature_names=np.asarray(diabetes.feature_names),
        metadata={
            "kind": "diabetes",
            "validation_size": validation_size,
            "test_size": test_size,
            "seed": seed,
            "target_train_mean": target_mean,
        },
    )
    dataset.validate()
    return dataset


def prepare_e2006_subset(
    train_path: str | Path,
    test_path: str | Path,
    *,
    train_samples: int = 3000,
    test_samples: int = 1000,
    top_features: int = 5000,
    validation_fraction: float = 0.2,
    seed: int = 0,
    name: str = "e2006_subset",
) -> DatasetBundle:
    """Load compressed LIBSVM files and create a dense, standardized subset."""
    from sklearn.datasets import load_svmlight_file
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must satisfy 0 < value < 1")

    train_path = Path(train_path)
    test_path = Path(test_path)
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"E2006 file not found: {path}")

    n_features = 150_360
    A_train_sparse, b_train = load_svmlight_file(
        train_path,
        n_features=n_features,
    )
    A_test_sparse, b_test = load_svmlight_file(
        test_path,
        n_features=n_features,
    )
    A_train_val_sparse = A_train_sparse[:train_samples]
    b_train_val = b_train[:train_samples]
    A_test_sparse = A_test_sparse[:test_samples]
    b_test = b_test[:test_samples]

    A_train_sparse, A_val_sparse, b_train, b_val = train_test_split(
        A_train_val_sparse,
        b_train_val,
        test_size=validation_fraction,
        random_state=seed,
    )

    feature_frequency = np.asarray(A_train_sparse.getnnz(axis=0)).ravel()
    available_features = int(np.count_nonzero(feature_frequency))
    selected_count = min(top_features, available_features)
    if selected_count == 0:
        raise ValueError("the selected E2006 training subset contains no features")
    selected = np.argpartition(feature_frequency, -selected_count)[-selected_count:]
    selected = np.sort(selected)

    A_train = A_train_sparse[:, selected].toarray()
    A_val = A_val_sparse[:, selected].toarray()
    A_test = A_test_sparse[:, selected].toarray()
    scaler = StandardScaler()
    A_train = scaler.fit_transform(A_train)
    A_val = scaler.transform(A_val)
    A_test = scaler.transform(A_test)

    target_mean = float(np.mean(b_train))
    b_train = np.asarray(b_train) - target_mean
    b_val = np.asarray(b_val) - target_mean
    b_test = np.asarray(b_test) - target_mean

    dataset = DatasetBundle(
        name=name,
        A_train=A_train,
        b_train=b_train,
        A_val=A_val,
        b_val=b_val,
        A_test=A_test,
        b_test=b_test,
        feature_names=selected.astype(str),
        metadata={
            "kind": "e2006_tfidf_subset",
            "train_samples": len(b_train),
            "validation_samples": len(b_val),
            "test_samples": len(b_test),
            "top_features": selected_count,
            "validation_fraction": validation_fraction,
            "seed": seed,
            "target_train_mean": target_mean,
        },
    )
    dataset.validate()
    return dataset


def save_dataset(dataset: DatasetBundle, path: str | Path) -> Path:
    """Save a dataset bundle as a compressed NumPy archive."""
    dataset.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "name": np.asarray(dataset.name),
        "A_train": dataset.A_train,
        "b_train": dataset.b_train,
        "metadata_json": np.asarray(
            json.dumps(dataset.metadata, ensure_ascii=True, sort_keys=True)
        ),
    }
    for key in (
        "A_val",
        "b_val",
        "A_test",
        "b_test",
        "x_star",
        "support",
        "feature_names",
    ):
        value = getattr(dataset, key)
        if value is not None:
            arrays[key] = np.asarray(value)

    np.savez_compressed(path, **arrays)
    return path


def load_dataset(path: str | Path) -> DatasetBundle:
    """Load a dataset created by save_dataset."""
    with np.load(Path(path), allow_pickle=True) as archive:
        files = set(archive.files)
        dataset = DatasetBundle(
            name=str(archive["name"]),
            A_train=archive["A_train"],
            b_train=archive["b_train"],
            A_val=archive["A_val"] if "A_val" in files else None,
            b_val=archive["b_val"] if "b_val" in files else None,
            A_test=archive["A_test"] if "A_test" in files else None,
            b_test=archive["b_test"] if "b_test" in files else None,
            x_star=archive["x_star"] if "x_star" in files else None,
            support=archive["support"] if "support" in files else None,
            feature_names=(
                archive["feature_names"] if "feature_names" in files else None
            ),
            metadata=json.loads(str(archive["metadata_json"])),
        )
    dataset.validate()
    return dataset
