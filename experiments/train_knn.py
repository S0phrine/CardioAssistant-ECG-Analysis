"""Train and evaluate k-nearest neighbours on MIT-BIH heartbeat features."""

from __future__ import annotations

import argparse
from pathlib import Path

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from cardio_ml.config import DS1_RECORDS, DS2_RECORDS, INTRA_RECORDS
from cardio_ml.data import load_feature_dataset
from cardio_ml.experiment import evaluate_and_save, make_cross_validation, split_dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing MIT-BIH .hea/.dat/.atr files.")
    parser.add_argument("--mode", choices=("intra", "inter"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-size", type=float, default=0.30, help="Used only in intra mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument(
        "--group-cv",
        action="store_true",
        help="Use patient-grouped CV during tuning (recommended for inter mode).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fit the final thesis configuration instead of searching the full grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    records = INTRA_RECORDS if args.mode == "intra" else (*DS1_RECORDS, *DS2_RECORDS)
    dataset = load_feature_dataset(args.data_dir, records)
    split = split_dataset(dataset, args.mode, test_size=args.test_size, random_state=args.seed)

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=args.seed)),
            ("knn", KNeighborsClassifier(n_jobs=args.jobs)),
        ]
    )
    parameter_grid = (
        {
            "knn__n_neighbors": [2],
            "knn__weights": ["distance"],
            "knn__metric": ["manhattan"],
        }
        if args.quick
        else {
            "knn__n_neighbors": [2, 3, 5, 7, 9],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"],
        }
    )
    cross_validation = make_cross_validation(args.group_cv, args.seed)
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=args.jobs,
        refit=True,
        verbose=1,
    )
    fit_arguments = {"groups": split.train_groups} if args.group_cv else {}
    search.fit(split.x_train, split.y_train, **fit_arguments)

    experiment_name = f"knn_{args.mode}"
    evaluate_and_save(search, split, args.output_dir, experiment_name)


if __name__ == "__main__":
    main()
