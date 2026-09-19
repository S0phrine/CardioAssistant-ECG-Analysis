"""SQLite persistence for CardioAssistant analysis results."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from contextlib import closing
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


DEFAULT_DATABASE_PATH = Path("data/cardioassistant.db")


def get_connection(database_path: str | Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Open the SQLite database and enable foreign-key validation."""

    path = Path(database_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
    """Create the analysis tables when they do not exist yet."""

    with closing(get_connection(database_path)) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                records TEXT NOT NULL,
                first_beat INTEGER NOT NULL,
                last_beat INTEGER NOT NULL,
                beat_count INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                dominant_class TEXT NOT NULL,
                mean_confidence REAL NOT NULL,
                analyzed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS heartbeat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                beat_index INTEGER NOT NULL,
                predicted_class TEXT NOT NULL,
                probability_n REAL,
                probability_s REAL,
                probability_v REAL,
                FOREIGN KEY (analysis_id) REFERENCES analysis(id) ON DELETE CASCADE
            );
            """
        )


def save_analysis(
    segment: Mapping[str, object],
    class_names: Sequence[str],
    model_name: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Save one analyzed segment and its per-heartbeat predictions atomically."""

    probabilities = np.asarray(segment["probabilities"], dtype=float)
    predictions = np.asarray(segment["predictions"], dtype=int)
    beat_count = int(segment["beat_count"])
    if probabilities.shape != (beat_count, len(class_names)):
        raise ValueError("Probability matrix has an unexpected shape.")
    if predictions.shape != (beat_count,):
        raise ValueError("Prediction vector has an unexpected shape.")

    if beat_count <= 0 or int(segment["end"]) - int(segment["start"]) + 1 != beat_count:
        raise ValueError("Invalid beat range.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid class probabilities.")
    if np.any(predictions < 0) or np.any(predictions >= len(class_names)):
        raise ValueError("Invalid class indices.")
    counts = dict(segment.get("counts", {}))
    dominant_class = max(counts, key=counts.get) if counts else ""
    class_to_index = {name: index for index, name in enumerate(class_names)}

    with closing(get_connection(database_path)) as connection, connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis (
                records, first_beat, last_beat, beat_count, model_name,
                dominant_class, mean_confidence, analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(segment.get("records", "")),
                int(segment["start"]),
                int(segment["end"]),
                beat_count,
                model_name,
                dominant_class,
                float(segment["mean_confidence"]),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        analysis_id = int(cursor.lastrowid)

        rows = []
        for offset, predicted_index in enumerate(predictions):
            probability = probabilities[offset]
            rows.append(
                (
                    analysis_id,
                    int(segment["start"]) + offset,
                    class_names[int(predicted_index)],
                    float(probability[class_to_index["N"]]) if "N" in class_to_index else None,
                    float(probability[class_to_index["S"]]) if "S" in class_to_index else None,
                    float(probability[class_to_index["V"]]) if "V" in class_to_index else None,
                )
            )

        connection.executemany(
            """
            INSERT INTO heartbeat (
                analysis_id, beat_index, predicted_class,
                probability_n, probability_s, probability_v
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    return analysis_id
