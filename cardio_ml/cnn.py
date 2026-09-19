"""PyTorch dataset and 1D CNN architecture used in the thesis."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import Dataset


class HeartbeatDataset(Dataset):
    """Expose one- or two-channel heartbeat segments to PyTorch."""

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        rr_feature: np.ndarray | None = None,
    ) -> None:
        signals = np.asarray(signals, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        if rr_feature is None:
            model_input = signals[:, None, :]
        else:
            rr_feature = np.asarray(rr_feature, dtype=np.float32)
            rhythm_channel = np.repeat(
                rr_feature[:, None],
                signals.shape[1],
                axis=1,
            )
            model_input = np.stack((signals, rhythm_channel), axis=1)

        self.signals = torch.from_numpy(model_input)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signals[index], self.labels[index]


class ECG1DCNN(nn.Module):
    """Three-layer 1D CNN followed by global average pooling."""

    def __init__(
        self,
        input_channels: int = 2,
        number_of_classes: int = 3,
        first_layer_channels: int = 96,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            input_channels,
            first_layer_channels,
            kernel_size=15,
            stride=2,
            padding=7,
        )
        # Short layer names preserve compatibility with the checkpoint saved
        # by the original thesis application.
        self.bn1 = nn.BatchNorm1d(first_layer_channels)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(
            first_layer_channels,
            64,
            kernel_size=7,
            padding=3,
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 32, kernel_size=7, padding=3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(32, number_of_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool1(
            functional.relu(self.bn1(self.conv1(inputs)))
        )
        features = self.pool2(
            functional.relu(self.bn2(self.conv2(features)))
        )
        features = functional.relu(self.conv3(features))
        features = self.gap(features).squeeze(-1)
        return self.fc(self.drop(features))


def build_rr_feature(
    rr_previous: np.ndarray,
    rr_next: np.ndarray,
    kind: str,
) -> np.ndarray:
    """Create the rhythm feature added as the constant second input channel."""

    if kind == "previous":
        feature = rr_previous
    elif kind == "next":
        feature = rr_next
    elif kind == "ratio":
        feature = (rr_previous + 1e-6) / (rr_next + 1e-6)
    else:
        raise ValueError("RR feature must be 'previous', 'next' or 'ratio'.")
    return np.asarray(feature, dtype=np.float32)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve auto/cpu/cuda/mps and fail clearly for unavailable hardware."""

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested_device)
