"""Generate all local datasets required by module A."""

from __future__ import annotations

import argparse
from pathlib import Path

from lasso_demo.data import (
    load_diabetes_dataset,
    make_synthetic_lasso,
    prepare_e2006_subset,
    save_dataset,
)


SYNTHETIC_PRESETS = {
    "small": dict(m=128, n=256, k=25, noise=0.01, corr=0.0, seed=0),
    "demo": dict(m=100, n=300, k=20, noise=0.01, corr=0.0, seed=1),
    "corr_050": dict(m=100, n=300, k=20, noise=0.01, corr=0.5, seed=1),
    "correlated": dict(m=100, n=300, k=20, noise=0.01, corr=0.8, seed=1),
    "corr_095": dict(m=100, n=300, k=20, noise=0.01, corr=0.95, seed=1),
    "noise_free": dict(m=100, n=300, k=20, noise=0.0, corr=0.0, seed=1),
    "noise_050": dict(m=100, n=300, k=20, noise=0.05, corr=0.0, seed=1),
    "noise_100": dict(m=100, n=300, k=20, noise=0.1, corr=0.0, seed=1),
    "stress": dict(m=200, n=1000, k=50, noise=0.01, corr=0.0, seed=2),
}


def prepare_standard_datasets(output_root: Path) -> list[Path]:
    generated: list[Path] = []
    synthetic_root = output_root / "synthetic"
    for preset_name, parameters in SYNTHETIC_PRESETS.items():
        dataset = make_synthetic_lasso(
            name=f"synthetic_{preset_name}",
            **parameters,
        )
        generated.append(
            save_dataset(dataset, synthetic_root / f"{dataset.name}.npz")
        )

    diabetes = load_diabetes_dataset()
    generated.append(save_dataset(diabetes, output_root / "diabetes" / "diabetes.npz"))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--prepare-e2006",
        action="store_true",
        help="also preprocess raw E2006 train/test files",
    )
    parser.add_argument(
        "--e2006-raw-root",
        type=Path,
        default=Path("data/raw/e2006"),
    )
    parser.add_argument("--e2006-train-samples", type=int, default=3000)
    parser.add_argument("--e2006-test-samples", type=int, default=1000)
    parser.add_argument("--e2006-top-features", type=int, default=5000)
    args = parser.parse_args()

    generated = prepare_standard_datasets(args.output_root)
    if args.prepare_e2006:
        dataset = prepare_e2006_subset(
            args.e2006_raw_root / "E2006.train.bz2",
            args.e2006_raw_root / "E2006.test.bz2",
            train_samples=args.e2006_train_samples,
            test_samples=args.e2006_test_samples,
            top_features=args.e2006_top_features,
        )
        generated.append(
            save_dataset(dataset, args.output_root / "e2006" / "e2006_subset.npz")
        )

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
