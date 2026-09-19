"""Preparation of fixed-length ECG segments for the 1D CNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import wfdb

from .config import AAMI_SYMBOL_TO_CLASS
from .data import _record_base_path, filter_ecg


@dataclass(frozen=True)
class SegmentDataset:
    """Filtered heartbeat segments with rhythm information."""

    signals: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    rr_previous: np.ndarray
    rr_next: np.ndarray


def _record_segments(
    data_directory: Path,
    record_id: str,
    half_window_samples: int,
    minimum_segment_std: float,
    expected_sampling_frequency: float,
) -> tuple[list[np.ndarray], list[str], list[float], list[float]]:
    base_path = _record_base_path(data_directory, record_id)
    record = wfdb.rdrecord(str(base_path))
    annotation = wfdb.rdann(str(base_path), "atr")

    sampling_frequency = float(record.fs)
    if not np.isclose(sampling_frequency, expected_sampling_frequency):
        raise ValueError(f"Record {record_id}: expected {expected_sampling_frequency} Hz, got {sampling_frequency} Hz.")
    signal = filter_ecg(record.p_signal[:, 0].astype(np.float32), sampling_frequency)

    segments: list[np.ndarray] = []
    labels: list[str] = []
    accepted_samples: list[int] = []

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        class_name = AAMI_SYMBOL_TO_CLASS.get(symbol)
        if class_name is None:
            continue

        sample = int(sample)
        start = sample - half_window_samples
        stop = sample + half_window_samples
        if start <= 0 or stop >= signal.size:
            continue

        segment = signal[start:stop]
        if segment.size != 2 * half_window_samples:
            continue
        segment_std = float(np.std(segment))
        if segment_std < minimum_segment_std:
            continue

        # Per-beat z-score used by the final CNN experiments.
        segment = (segment - np.mean(segment)) / (segment_std + 1e-8)
        segments.append(segment.astype(np.float32))
        labels.append(class_name)
        accepted_samples.append(sample)

    if not segments:
        return [], [], [], []

    sample_array = np.asarray(accepted_samples, dtype=np.int64)
    intervals = np.diff(sample_array) / sampling_frequency
    valid_intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    median_interval = float(np.median(valid_intervals)) if valid_intervals.size else 1.0

    rr_previous = np.full(sample_array.size, median_interval, dtype=np.float32)
    rr_next = np.full(sample_array.size, median_interval, dtype=np.float32)
    if intervals.size:
        rr_previous[1:] = intervals
        rr_next[:-1] = intervals

    return segments, labels, rr_previous.tolist(), rr_next.tolist()


def load_segment_dataset(
    data_directory: str | Path,
    record_ids: Iterable[str],
    *,
    half_window_samples: int = 90,
    minimum_segment_std: float = 0.05,
    expected_sampling_frequency: float = 360.0,
) -> SegmentDataset:
    """Build the fixed-length segment dataset used by the 1D CNN."""

    data_directory = Path(data_directory).expanduser().resolve()
    all_signals: list[np.ndarray] = []
    all_labels: list[str] = []
    all_record_ids: list[str] = []
    all_rr_previous: list[float] = []
    all_rr_next: list[float] = []

    for record_id in record_ids:
        record_id = str(record_id)
        print(f"Preparing CNN segments from record {record_id}...")
        signals, labels, rr_previous, rr_next = _record_segments(
            data_directory,
            record_id,
            half_window_samples,
            minimum_segment_std,
            expected_sampling_frequency,
        )
        all_signals.extend(signals)
        all_labels.extend(labels)
        all_record_ids.extend([record_id] * len(labels))
        all_rr_previous.extend(rr_previous)
        all_rr_next.extend(rr_next)

    if not all_signals:
        raise RuntimeError("No heartbeat segments were extracted from the selected records.")

    return SegmentDataset(
        signals=np.stack(all_signals).astype(np.float32),
        labels=np.asarray(all_labels),
        record_ids=np.asarray(all_record_ids),
        rr_previous=np.asarray(all_rr_previous, dtype=np.float32),
        rr_next=np.asarray(all_rr_next, dtype=np.float32),
    )
