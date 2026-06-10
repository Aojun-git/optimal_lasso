"""CSV export helpers for histories and summary tables."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path


RESULT_COLUMNS = (
    "dataset",
    "algorithm",
    "mu",
    "objective",
    "relative_objective_gap",
    "relative_error",
    "nonzero_count",
    "support_precision",
    "support_recall",
    "support_f1",
    "validation_mse",
    "test_mse",
    "cpu_time",
    "notes",
)


def build_result_row(
    *,
    dataset: str,
    algorithm: str,
    mu: float,
    metrics: Mapping[str, float | int],
    notes: str = "",
) -> dict[str, str | float | int]:
    row: dict[str, str | float | int] = {
        "dataset": dataset,
        "algorithm": algorithm,
        "mu": mu,
        "notes": notes,
    }
    row.update(metrics)
    return row


def export_result_rows(
    rows: Iterable[Mapping[str, str | float | int]],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})
    return path


def export_history(
    history: Mapping[str, list[float | int]],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(history)
    lengths = {len(history[column]) for column in columns}
    if len(lengths) > 1:
        raise ValueError("history columns must have equal lengths")

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index in range(next(iter(lengths), 0)):
            writer.writerow({column: history[column][index] for column in columns})
    return path
