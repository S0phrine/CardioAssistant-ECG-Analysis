"""Extraction of the 12 handcrafted heartbeat features."""

from __future__ import annotations

import numpy as np
import pywt

from .config import FEATURE_NAMES


def _spectral_magnitude_sum(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    return float(np.abs(spectrum[mask]).sum())


def extract_feature_vector(
    segment: np.ndarray,
    previous_rr_seconds: float,
    sampling_frequency: float,
) -> np.ndarray:
    """Return the 12 features used by the classical classifiers.

    The spectral features intentionally use sums of FFT magnitudes, matching
    the implementation used in the thesis experiments.
    """

    segment = np.asarray(segment, dtype=np.float64)
    if segment.ndim != 1:
        raise ValueError("A heartbeat segment must be one-dimensional.")

    spectrum = np.fft.fft(segment)
    frequencies = np.fft.fftfreq(segment.size, d=1.0 / sampling_frequency)

    wavelet_coefficients = pywt.wavedec(segment, "db4", level=2)
    wavelet_energies = [float(np.square(coefficients).sum()) for coefficients in wavelet_coefficients]

    features = np.asarray(
        [
            np.max(segment),
            np.min(segment),
            np.mean(segment),
            np.std(segment),
            np.abs(np.diff(segment)).sum(),
            _spectral_magnitude_sum(spectrum, frequencies, 0.0, 10.0),
            _spectral_magnitude_sum(spectrum, frequencies, 10.0, 40.0),
            _spectral_magnitude_sum(spectrum, frequencies, 40.0, 100.0),
            *wavelet_energies,
            previous_rr_seconds,
        ],
        dtype=np.float64,
    )

    if features.size != len(FEATURE_NAMES):
        raise RuntimeError("Unexpected number of extracted features.")
    return features
