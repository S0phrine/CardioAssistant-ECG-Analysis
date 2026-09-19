"""MIT-BIH loading, filtering, segmentation and feature preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, medfilt

from .config import AAMI_SYMBOL_TO_CLASS, FEATURE_NAMES
from .features import extract_feature_vector


@dataclass(frozen=True)
class PreparedDataset:
    """Features, labels and record identifiers for individual heartbeats."""

    features: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray


def _odd_kernel_size(seconds: float, sampling_frequency: float) -> int:
    size = max(3, int(seconds * sampling_frequency))
    return size if size % 2 else size + 1


def filter_ecg(signal: np.ndarray, sampling_frequency: float) -> np.ndarray:
    """Apply baseline correction, 50 Hz notch and 0.5-40 Hz band-pass."""

    signal = np.asarray(signal, dtype=np.float64)
    baseline = medfilt(
        signal,
        kernel_size=_odd_kernel_size(0.2, sampling_frequency),
    )
    corrected = signal - baseline

    notch_b, notch_a = iirnotch(50.0, 30.0, sampling_frequency)
    notch_filtered = filtfilt(notch_b, notch_a, corrected)

    nyquist = sampling_frequency / 2.0
    band_b, band_a = butter(
        4,
        [0.5 / nyquist, 40.0 / nyquist],
        btype="bandpass",
    )
    filtered = filtfilt(band_b, band_a, notch_filtered)

    maximum = np.max(np.abs(filtered))
    if maximum == 0:
        return filtered
    return filtered / maximum


def _record_base_path(data_directory: Path, record_id: str) -> Path:
    base_path = data_directory / record_id
    missing = [suffix for suffix in (".hea", ".dat", ".atr") if not base_path.with_suffix(suffix).exists()]
    if missing:
        raise FileNotFoundError(
            f"Record {record_id} is incomplete in {data_directory}. "
            f"Missing: {', '.join(missing)}"
        )
    return base_path


def _extract_record_examples(
    data_directory: Path,
    record_id: str,
    half_window_samples: int,
    minimum_segment_std: float,
) -> tuple[list[np.ndarray], list[str]]:
    base_path = _record_base_path(data_directory, record_id)
    record = wfdb.rdrecord(str(base_path))
    annotation = wfdb.rdann(str(base_path), "atr")

    sampling_frequency = float(record.fs)
    if not np.isclose(sampling_frequency, 360.0):
        raise ValueError(f"Record {record_id}: expected 360 Hz, got {sampling_frequency} Hz.")
    signal = filter_ecg(record.p_signal[:, 0], sampling_frequency)

    # Match the notebooks: RR is measured between retained N/S/V windows.
    retained = []
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        class_name = AAMI_SYMBOL_TO_CLASS.get(symbol)
        sample = int(sample)
        if class_name is None or not (half_window_samples < sample < signal.size - half_window_samples):
            continue
        segment = signal[sample - half_window_samples:sample + half_window_samples]
        if np.std(segment) < minimum_segment_std:
            continue
        retained.append((sample, class_name, segment))
    if not retained:
        return [], []
    intervals = np.diff([item[0] for item in retained]) / sampling_frequency
    # Per-record imputation deliberately avoids pooling test-record statistics.
    median_rr = float(np.median(intervals)) if intervals.size else 1.0
    previous_rr = np.concatenate(([median_rr], intervals))
    feature_rows = [extract_feature_vector(
        segment, previous_rr_seconds=float(rr), sampling_frequency=sampling_frequency
    ) for (_, _, segment), rr in zip(retained, previous_rr)]
    labels = [label for _, label, _ in retained]

    return feature_rows, labels


def load_feature_dataset(
    data_directory: str | Path,
    record_ids: Iterable[str],
    *,
    half_window_samples: int = 90,
    minimum_segment_std: float = 0.05,
) -> PreparedDataset:
    """Build the 12-feature dataset for the requested MIT-BIH records."""

    data_directory = Path(data_directory).expanduser().resolve()
    all_features: list[np.ndarray] = []
    all_labels: list[str] = []
    all_record_ids: list[str] = []

    for record_id in record_ids:
        print(f"Preparing record {record_id}...")
        features, labels = _extract_record_examples(
            data_directory,
            str(record_id),
            half_window_samples,
            minimum_segment_std,
        )
        all_features.extend(features)
        all_labels.extend(labels)
        all_record_ids.extend([str(record_id)] * len(labels))

    if not all_features:
        raise RuntimeError("No heartbeat examples were extracted from the selected records.")

    feature_matrix = np.vstack(all_features)
    if feature_matrix.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("The feature matrix does not contain the expected 12 columns.")

    return PreparedDataset(
        features=feature_matrix,
        labels=np.asarray(all_labels),
        record_ids=np.asarray(all_record_ids),
    )
