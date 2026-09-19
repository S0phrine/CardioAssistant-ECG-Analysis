"""Shared experiment splitting, evaluation and artifact saving."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

from .config import AAMI_CLASS_NAMES, DS1_RECORDS, DS2_RECORDS
from .data import PreparedDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class ExperimentSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_groups: np.ndarray


def split_dataset(
    dataset: PreparedDataset,
    mode: str,
    *,
    test_size: float,
    random_state: int,
) -> ExperimentSplit:
    if mode == "intra":
        indices = np.arange(dataset.labels.size)
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=dataset.labels,
        )
    elif mode == "inter":
        train_mask = np.isin(dataset.record_ids, DS1_RECORDS)
        test_mask = np.isin(dataset.record_ids, DS2_RECORDS)
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        if train_indices.size == 0 or test_indices.size == 0:
            raise RuntimeError("The DS1/DS2 split produced an empty subset.")
    else:
        raise ValueError("Mode must be 'intra' or 'inter'.")

    return ExperimentSplit(
        x_train=dataset.features[train_indices],
        x_test=dataset.features[test_indices],
        y_train=dataset.labels[train_indices],
        y_test=dataset.labels[test_indices],
        train_groups=dataset.record_ids[train_indices],
    )


def make_cross_validation(group_cv: bool, random_state: int):
    if group_cv:
        return GroupKFold(n_splits=5)
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_and_save(
    search,
    split: ExperimentSplit,
    output_directory: str | Path,
    experiment_name: str,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    predictions = search.predict(split.x_test)
    matrix = confusion_matrix(split.y_test, predictions, labels=AAMI_CLASS_NAMES)
    report = classification_report(
        split.y_test,
        predictions,
        labels=AAMI_CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "accuracy": accuracy_score(split.y_test, predictions),
        "macro_f1": report["macro avg"]["f1-score"],
        "best_cross_validation_macro_f1": search.best_score_,
        "best_parameters": search.best_params_,
        "classification_report": report,
        "confusion_matrix": matrix,
    }

    model_path = output_directory / f"{experiment_name}.joblib"
    metrics_path = output_directory / f"{experiment_name}_metrics.json"
    matrix_path = output_directory / f"{experiment_name}_confusion_matrix.png"

    joblib.dump(search.best_estimator_, model_path)
    metrics_path.write_text(
        json.dumps(_json_value(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    display = ConfusionMatrixDisplay(matrix, display_labels=AAMI_CLASS_NAMES)
    display.plot(cmap="Blues", colorbar=False)
    plt.title(experiment_name.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=180)
    plt.close()

    print(json.dumps(_json_value(summary), indent=2, ensure_ascii=False))
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved confusion matrix: {matrix_path}")
    return summary
