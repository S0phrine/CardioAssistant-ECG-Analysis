"""Train and evaluate the thesis 1D CNN on MIT-BIH heartbeat segments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from cardio_ml.cnn import ECG1DCNN, HeartbeatDataset, build_rr_feature, resolve_device
from cardio_ml.config import AAMI_CLASS_NAMES, DS1_RECORDS, DS2_RECORDS, INTRA_RECORDS
from cardio_ml.segments import SegmentDataset, load_segment_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("intra", "inter"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=864)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--first-layer-channels", type=int, default=96)
    parser.add_argument("--validation-fraction", type=float, default=0.1111)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--rr-feature", choices=("previous", "next", "ratio"), default="ratio")
    parser.add_argument("--no-rr-channel", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def split_indices(
    dataset: SegmentDataset,
    mode: str,
    test_size: float,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(dataset.labels.size)
    if mode == "intra":
        train_pool, test_indices = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=seed,
            stratify=dataset.labels,
        )
    else:
        train_pool = all_indices[np.isin(dataset.record_ids, DS1_RECORDS)]
        test_indices = all_indices[np.isin(dataset.record_ids, DS2_RECORDS)]
        if train_pool.size == 0 or test_indices.size == 0:
            raise RuntimeError("The DS1/DS2 split produced an empty subset.")

    train_indices, validation_indices = train_test_split(
        train_pool,
        test_size=validation_fraction,
        random_state=seed,
        stratify=dataset.labels[train_pool],
    )
    return train_indices, validation_indices, test_indices


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            batch_probabilities = torch.softmax(logits, dim=1)
            true_labels.extend(labels.numpy().tolist())
            predicted_labels.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            probabilities.append(batch_probabilities.cpu().numpy())

    true_array = np.asarray(true_labels, dtype=np.int64)
    predicted_array = np.asarray(predicted_labels, dtype=np.int64)
    probability_array = np.concatenate(probabilities, axis=0)
    macro_f1 = f1_score(true_array, predicted_array, average="macro", zero_division=0)
    return macro_f1, true_array, predicted_array, probability_array


def save_results(
    model: ECG1DCNN,
    output_directory: Path,
    configuration: dict,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    label_indices = list(range(len(AAMI_CLASS_NAMES)))
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=label_indices,
        target_names=AAMI_CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=label_indices)

    checkpoint = {
        "model_state": model.state_dict(),
        "class_names": AAMI_CLASS_NAMES,
        "config": configuration,
    }
    torch.save(checkpoint, output_directory / "checkpoint.pt")
    metadata = {
        "target_classes": list(AAMI_CLASS_NAMES),
        "in_ch": configuration["input_channels"],
        "fs": configuration["sampling_frequency_hz"],
        "window": configuration["half_window_samples"],
        "rr_norm_mu": [configuration["rr_mean"]] if configuration["input_channels"] == 2 else None,
        "rr_norm_sd": [configuration["rr_standard_deviation"]] if configuration["input_channels"] == 2 else None,
        "rr_feature_kind": configuration["rr_feature"],
        "model_hparams": {"conv1_out": configuration["first_layer_channels"], "p_drop": configuration["dropout"]},
        "seed": configuration["seed"],
    }
    (output_directory / "checkpoint.meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


    summary = {
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "configuration": configuration,
    }
    (output_directory / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (output_directory / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["y_true", "y_pred", *[f"p_{name}" for name in AAMI_CLASS_NAMES]])
        for true_index, predicted_index, probability in zip(
            true_labels,
            predicted_labels,
            probabilities,
        ):
            writer.writerow(
                [
                    AAMI_CLASS_NAMES[int(true_index)],
                    AAMI_CLASS_NAMES[int(predicted_index)],
                    *map(float, probability),
                ]
            )

    display = ConfusionMatrixDisplay(matrix, display_labels=AAMI_CLASS_NAMES)
    display.plot(cmap="Blues", colorbar=False)
    plt.title("1D CNN – test set")
    plt.tight_layout()
    plt.savefig(output_directory / "confusion_matrix.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_arguments()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    records = INTRA_RECORDS if args.mode == "intra" else (*DS1_RECORDS, *DS2_RECORDS)
    dataset = load_segment_dataset(args.data_dir, records)
    train_indices, validation_indices, test_indices = split_indices(
        dataset,
        args.mode,
        args.test_size,
        args.validation_fraction,
        args.seed,
    )

    label_to_index = {label: index for index, label in enumerate(AAMI_CLASS_NAMES)}
    encoded_labels = np.asarray([label_to_index[label] for label in dataset.labels], dtype=np.int64)

    rr_feature = build_rr_feature(dataset.rr_previous, dataset.rr_next, args.rr_feature)
    rr_mean = float(rr_feature[train_indices].mean())
    rr_std = float(rr_feature[train_indices].std() + 1e-8)
    rr_feature = (rr_feature - rr_mean) / rr_std
    use_rr_channel = not args.no_rr_channel

    def make_dataset(indices: np.ndarray) -> HeartbeatDataset:
        return HeartbeatDataset(
            dataset.signals[indices],
            encoded_labels[indices],
            rr_feature[indices] if use_rr_channel else None,
        )

    training_dataset = make_dataset(train_indices)
    validation_dataset = make_dataset(validation_indices)
    test_dataset = make_dataset(test_indices)

    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    evaluation_batch_size = 2 * args.batch_size
    validation_loader = DataLoader(validation_dataset, batch_size=evaluation_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=evaluation_batch_size, shuffle=False)

    input_channels = 2 if use_rr_channel else 1
    model = ECG1DCNN(
        input_channels=input_channels,
        number_of_classes=len(AAMI_CLASS_NAMES),
        first_layer_channels=args.first_layer_channels,
        dropout=args.dropout,
    ).to(device)

    class_counts = Counter(encoded_labels[train_indices].tolist())
    number_of_examples = len(train_indices)
    class_weights = torch.tensor(
        [number_of_examples / class_counts[index] for index in range(len(AAMI_CLASS_NAMES))],
        dtype=torch.float32,
        device=device,
    )
    print(f"Training class distribution: {class_counts}")
    print(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_validation_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, labels in training_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        validation_f1, _, _, _ = evaluate(model, validation_loader, device)
        print(f"Epoch {epoch:02d} | validation macro F1: {validation_f1:.4f}")

        if validation_f1 > best_validation_f1 + 1e-4:
            best_validation_f1 = validation_f1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)

    test_f1, true_labels, predicted_labels, probabilities = evaluate(model, test_loader, device)
    print(f"Test macro F1: {test_f1:.4f}")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=list(range(len(AAMI_CLASS_NAMES))),
            target_names=AAMI_CLASS_NAMES,
            zero_division=0,
        )
    )

    configuration = {
        "mode": args.mode,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "first_layer_channels": args.first_layer_channels,
        "input_channels": input_channels,
        "rr_feature": args.rr_feature if use_rr_channel else None,
        "rr_mean": rr_mean if use_rr_channel else None,
        "rr_standard_deviation": rr_std if use_rr_channel else None,
        "half_window_samples": 90,
        "sampling_frequency_hz": 360,
        "best_validation_macro_f1": best_validation_f1,
    }
    run_name = f"cnn_{args.mode}_{'rr-' + args.rr_feature if use_rr_channel else 'signal-only'}_seed-{args.seed}"
    save_results(
        model,
        args.output_dir / run_name,
        configuration,
        true_labels,
        predicted_labels,
        probabilities,
    )
    print(f"Saved results to: {args.output_dir / run_name}")


if __name__ == "__main__":
    main()
