"""Tkinter desktop interface for the CardioAssistant research prototype."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

from cardio_ml.cnn import ECG1DCNN, build_rr_feature, resolve_device
from cardio_ml.database import initialize_database, save_analysis
from cardio_ml.segments import load_segment_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIRECTORY = PROJECT_ROOT / "data" / "mitdb"
DEFAULT_MODEL_PATH = Path("models") / "best_model.pt"
DEFAULT_METADATA_PATH = Path("models") / "best_model.meta.json"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "cardioassistant.db"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve repository-relative paths without depending on the working directory."""

    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def load_metadata(path: str | Path) -> dict:
    """Load and validate the metadata required to reproduce preprocessing."""

    metadata_path = Path(path).expanduser()

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono metadanych: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    required_keys = {
        "target_classes",
        "in_ch",
        "fs",
        "window",
        "rr_norm_mu",
        "rr_norm_sd",
        "model_hparams",
        "rr_feature_kind",
    }

    missing = sorted(required_keys.difference(metadata))

    if missing:
        raise ValueError(f"W metadanych brakuje pól: {', '.join(missing)}")

    if metadata["target_classes"] != ["N", "S", "V"]:
        raise ValueError("Oczekiwano klas N, S, V w tej kolejności.")

    if (
        metadata["in_ch"] not in (1, 2)
        or metadata["fs"] != 360
        or metadata["window"] != 90
    ):
        raise ValueError(
            "Oczekiwano modelu 1/2-kanałowego, " "360 Hz i okna ±90 próbek."
        )

    if metadata["in_ch"] == 2:
        if metadata["rr_feature_kind"] not in (
            "previous",
            "next",
            "ratio",
        ):
            raise ValueError("Nieznana cecha RR.")

        mean = np.asarray(
            metadata["rr_norm_mu"],
            dtype=float,
        )

        std = np.asarray(
            metadata["rr_norm_sd"],
            dtype=float,
        )

        if (
            mean.shape != (1,)
            or std.shape != (1,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or std[0] <= 0
        ):
            raise ValueError("Niepoprawna normalizacja RR w metadanych.")

    return metadata


def load_state_dict(
    path: str | Path,
    device: torch.device,
) -> dict:
    """Load either the original state dict or a cleaned training checkpoint."""

    model_path = Path(path).expanduser()

    if not model_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono wag modelu: {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]

    if not isinstance(checkpoint, dict):
        raise ValueError("Plik wag nie zawiera słownika stanu modelu.")

    return checkpoint


@torch.inference_mode()
def predict_probabilities(
    model: ECG1DCNN,
    signals: np.ndarray,
    rr_feature: np.ndarray | None,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict class probabilities for normalized heartbeat segments."""

    model.eval()

    signal_array = np.asarray(
        signals,
        dtype=np.float32,
    )

    if (
        signal_array.ndim != 2
        or not len(signal_array)
        or not np.isfinite(signal_array).all()
    ):
        raise ValueError("Niepoprawne segmenty sygnału.")

    if batch_size <= 0:
        raise ValueError("Rozmiar partii musi być dodatni.")

    if rr_feature is not None:
        if (
            np.shape(rr_feature) != (len(signal_array),)
            or not np.isfinite(rr_feature).all()
        ):
            raise ValueError("Niepoprawna cecha RR.")

    if rr_feature is None:
        model_input = signal_array[:, None, :]
    else:
        rhythm_channel = np.repeat(
            np.asarray(
                rr_feature,
                dtype=np.float32,
            )[:, None],
            signal_array.shape[1],
            axis=1,
        )

        model_input = np.stack(
            (signal_array, rhythm_channel),
            axis=1,
        )

    probability_batches = []

    for start in range(
        0,
        model_input.shape[0],
        batch_size,
    ):
        batch = torch.from_numpy(model_input[start : start + batch_size]).to(device)

        probability_batches.append(
            torch.softmax(
                model(batch),
                dim=1,
            )
            .cpu()
            .numpy()
        )

    probabilities = np.concatenate(
        probability_batches,
        axis=0,
    )

    predictions = probabilities.argmax(axis=1)

    return probabilities, predictions


class CardioAssistantApp(tk.Tk):
    """Desktop interface for loading, analyzing and exporting ECG segments."""

    def __init__(self) -> None:
        super().__init__()

        self.title("CardioAssistant")
        self.geometry("1360x650")
        self.protocol(
            "WM_DELETE_WINDOW",
            self.on_close,
        )

        self.device = resolve_device("auto")

        self.model: ECG1DCNN | None = None
        self.class_names: tuple[str, ...] = (
            "N",
            "S",
            "V",
        )

        self.metadata: dict = {}
        self.loaded_records = ""
        self.loaded_model_name = ""
        self.record_ids = None
        self.signals: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.rr_feature: np.ndarray | None = None
        self.last_analysis: dict | None = None

        self._build_controls()
        self._build_chart_and_results()
        self.after_idle(self.freeze_window_size)

    def _build_controls(self) -> None:
        top = ttk.Frame(self)

        top.pack(
            side=tk.TOP,
            fill=tk.X,
            padx=10,
            pady=6,
        )

        ttk.Label(
            top,
            text="Folder",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        self.data_directory_var = tk.StringVar(value=str(DEFAULT_DATA_DIRECTORY))

        ttk.Entry(
            top,
            textvariable=self.data_directory_var,
        ).grid(
            row=0,
            column=1,
            padx=6,
            pady=2,
            sticky="ew",
        )

        ttk.Button(
            top,
            text="Przeglądaj",
            command=self.browse_data_directory,
        ).grid(
            row=0,
            column=2,
            padx=(0, 6),
        )

        ttk.Label(
            top,
            text="Wagi",
        ).grid(
            row=1,
            column=0,
            sticky="w",
        )

        self.model_path_var = tk.StringVar(value=str(DEFAULT_MODEL_PATH))

        ttk.Entry(
            top,
            textvariable=self.model_path_var,
        ).grid(
            row=1,
            column=1,
            padx=6,
            pady=2,
            sticky="ew",
        )

        ttk.Label(
            top,
            text="Meta",
        ).grid(
            row=2,
            column=0,
            sticky="w",
        )

        self.metadata_path_var = tk.StringVar(value=str(DEFAULT_METADATA_PATH))

        ttk.Entry(
            top,
            textvariable=self.metadata_path_var,
        ).grid(
            row=2,
            column=1,
            padx=6,
            pady=2,
            sticky="ew",
        )

        ttk.Button(
            top,
            text="Inicjalizuj model",
            command=self.initialize_model,
        ).grid(
            row=1,
            column=3,
            rowspan=2,
            padx=(8, 0),
            sticky="nsew",
        )

        top.columnconfigure(
            1,
            weight=1,
        )

        bottom = ttk.Frame(self)

        bottom.pack(
            side=tk.BOTTOM,
            fill=tk.X,
            padx=10,
            pady=8,
        )

        ttk.Label(
            bottom,
            text="Wpisz rekord/y",
        ).pack(side=tk.LEFT)

        self.records_var = tk.StringVar(value="100,103,105,106,108")

        ttk.Entry(
            bottom,
            textvariable=self.records_var,
            width=52,
        ).pack(
            side=tk.LEFT,
            padx=8,
        )

        ttk.Button(
            bottom,
            text="Wczytaj",
            command=self.load_records,
        ).pack(
            side=tk.LEFT,
            padx=(0, 6),
        )

        ttk.Button(
            bottom,
            text="Zapisz",
            command=self.save_results,
        ).pack(side=tk.RIGHT)

    def _build_chart_and_results(self) -> None:
        main = ttk.Frame(self)

        main.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
        )

        self.figure, self.axis = plt.subplots(figsize=(9.5, 3.9))

        self.axis.set_title("Brak danych do analizy")

        self.axis.set_xlabel("Próbki sygnału")

        self.axis.set_ylabel("Znormalizowana amplituda")

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=main,
        )

        self.canvas.get_tk_widget().pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
        )

        side = ttk.Frame(main, width=560)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side.pack_propagate(False)

        ttk.Label(
            side,
            text=f"Urządzenie: {str(self.device).upper()}",
        ).pack(anchor="w")

        status_box = ttk.Frame(side, height=45)
        status_box.pack(fill=tk.X)
        status_box.pack_propagate(False)

        self.status_var = tk.StringVar(value="Najpierw zainicjalizuj model.")
        self.status_label = ttk.Label(
            status_box,
            textvariable=self.status_var,
            foreground="orange",
            anchor="w",
            justify=tk.LEFT,
            wraplength=535,
        )
        self.status_label.pack(fill=tk.X, anchor="w")

        ttk.Separator(
            side,
            orient=tk.HORIZONTAL,
        ).pack(
            fill=tk.X,
            pady=6,
        )

        range_frame = ttk.LabelFrame(
            side,
            text="Wybierz zakres do analizy",
        )

        range_frame.pack(
            fill=tk.X,
            pady=(0, 6),
        )

        ttk.Button(
            range_frame,
            text="ⓘ",
            width=3,
            command=self.show_range_info,
        ).pack(
            anchor="e",
            padx=4,
            pady=(4, 0),
        )

        self.start_var = tk.IntVar(value=0)
        self.end_var = tk.IntVar(value=0)

        self.start_scale = ttk.Scale(
            range_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.start_var,
            command=self.on_range_change,
        )

        self.start_scale.pack(
            fill=tk.X,
            pady=(6, 2),
        )

        self.end_scale = ttk.Scale(
            range_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.end_var,
            command=self.on_range_change,
        )

        self.end_scale.pack(
            fill=tk.X,
            pady=(0, 4),
        )

        entry_row = ttk.Frame(range_frame)

        entry_row.pack(
            fill=tk.X,
            pady=(0, 4),
        )

        ttk.Label(
            entry_row,
            text="Zakres:",
        ).pack(side=tk.LEFT)

        self.start_entry = ttk.Entry(
            entry_row,
            width=7,
        )

        self.start_entry.pack(
            side=tk.LEFT,
            padx=4,
        )

        ttk.Label(
            entry_row,
            text="–",
        ).pack(side=tk.LEFT)

        self.end_entry = ttk.Entry(
            entry_row,
            width=7,
        )

        self.end_entry.pack(
            side=tk.LEFT,
            padx=4,
        )

        self.start_entry.bind(
            "<Return>",
            self.on_range_entry,
        )

        self.end_entry.bind(
            "<Return>",
            self.on_range_entry,
        )

        self.beat_count_var = tk.StringVar(value="Liczba uderzeń: –")

        ttk.Label(
            range_frame,
            textvariable=self.beat_count_var,
        ).pack(anchor="w")

        ttk.Button(
            range_frame,
            text="Analizuj segment",
            command=self.analyze_segment,
        ).pack(
            anchor="e",
            padx=4,
            pady=6,
        )

        self.probability_frame = ttk.LabelFrame(
            side,
            text="Prawdopodobieństwo przynależności do klas",
        )

        self.probability_frame.pack(
            fill=tk.X,
            pady=(0, 6),
        )

        self.probability_info_button = ttk.Button(
            self.probability_frame,
            text="ⓘ",
            width=3,
            command=self.show_result_info,
        )

        self.probability_info_button.pack(
            anchor="e",
            padx=4,
            pady=(4, 0),
        )

        self.class_result_vars: dict[
            str,
            tk.StringVar,
        ] = {}

        self.refresh_class_labels()

        self.confidence_var = tk.StringVar(value="Średnia pewność decyzji modelu: –")

        ttk.Label(
            side,
            textvariable=self.confidence_var,
            wraplength=280,
        ).pack(anchor="w")

    def browse_data_directory(self) -> None:
        selected = filedialog.askdirectory()

        if selected:
            self.data_directory_var.set(selected)

    def set_status(
        self,
        text: str,
        color: str,
    ) -> None:
        self.status_var.set(text)

        self.status_label.configure(foreground=color)

    def discard_analysis(self) -> bool:
        if self.last_analysis is not None and not messagebox.askyesno(
            "Niezapisane wyniki",
            "Odrzucić wyniki ostatniej analizy?",
        ):
            return False

        return True

    def clear_results(self) -> None:
        self.last_analysis = None

        self.probability_frame.configure(
            text="Prawdopodobieństwo przynależności do klas"
        )

        self.refresh_class_labels()

        self.confidence_var.set("Średnia pewność decyzji modelu: –")

    def initialize_model(self) -> None:
        if not self.discard_analysis():
            return

        try:
            metadata_path = resolve_project_path(self.metadata_path_var.get())

            model_path = resolve_project_path(self.model_path_var.get())

            metadata = load_metadata(metadata_path)

            class_names = tuple(metadata["target_classes"])

            input_channels = int(metadata["in_ch"])

            if input_channels not in (1, 2):
                raise ValueError("Aplikacja obsługuje model " "jedno- lub dwukanałowy.")

            parameters = metadata.get(
                "model_hparams",
                {},
            )

            model = ECG1DCNN(
                input_channels=input_channels,
                number_of_classes=len(class_names),
                first_layer_channels=int(
                    parameters.get(
                        "conv1_out",
                        96,
                    )
                ),
                dropout=float(
                    parameters.get(
                        "p_drop",
                        0.20,
                    )
                ),
            ).to(self.device)

            state_dict = load_state_dict(
                model_path,
                self.device,
            )

            model.load_state_dict(
                state_dict,
                strict=True,
            )

            model.eval()

            self.metadata = metadata
            self.class_names = class_names
            self.model = model
            self.loaded_model_name = model_path.name
            self.signals = None
            self.rr_feature = None
            self.last_analysis = None

            self.clear_results()

            self.axis.clear()

            self.axis.set_title("Wczytaj rekordy dla nowego modelu")

            self.canvas.draw_idle()
            self.update_range_labels()

            self.set_status(
                "Model gotowy. Wczytaj rekordy MIT-BIH.",
                "green",
            )

        except Exception as error:
            self.set_status(
                "Nie udało się zainicjalizować modelu.",
                "red",
            )

            messagebox.showerror(
                "Inicjalizacja modelu",
                str(error),
            )

    def load_records(self) -> None:
        if self.model is None:
            self.set_status(
                "Najpierw zainicjalizuj model.",
                "orange",
            )
            return

        if not self.discard_analysis():
            return

        records = [
            value.strip()
            for value in self.records_var.get().split(",")
            if value.strip()
        ]

        if not records:
            messagebox.showwarning(
                "Rekordy",
                "Podaj co najmniej jeden numer rekordu.",
            )
            return

        try:
            dataset = load_segment_dataset(
                self.data_directory_var.get(),
                records,
                half_window_samples=int(self.metadata["window"]),
                expected_sampling_frequency=float(self.metadata["fs"]),
            )

            if int(self.metadata["in_ch"]) == 2:
                rr_feature = build_rr_feature(
                    dataset.rr_previous,
                    dataset.rr_next,
                    str(self.metadata["rr_feature_kind"]),
                )

                rr_mean = float(np.asarray(self.metadata["rr_norm_mu"]).reshape(-1)[0])

                rr_std = float(np.asarray(self.metadata["rr_norm_sd"]).reshape(-1)[0])

                if rr_std <= 0:
                    raise ValueError(
                        "Odchylenie standardowe cechy RR " "musi być dodatnie."
                    )

                normalized_rr = ((rr_feature - rr_mean) / rr_std).astype(np.float32)

            else:
                normalized_rr = None

            self.rr_feature = normalized_rr
            self.loaded_records = ",".join(records)
            self.record_ids = dataset.record_ids
            self.signals = dataset.signals
            self.labels = dataset.labels

            self.clear_results()

            last_index = len(dataset.signals) - 1

            self.start_scale.configure(
                from_=0,
                to=last_index,
            )

            self.end_scale.configure(
                from_=0,
                to=last_index,
            )

            self.start_var.set(0)
            self.end_var.set(last_index)

            self.update_range_labels()
            self.plot_segment(0, last_index)

            self.set_status(
                f"Wczytano {len(dataset.signals)} uderzeń.",
                "green",
            )

        except Exception as error:
            self.set_status(
                "Nie udało się wczytać rekordów.",
                "red",
            )

            messagebox.showerror(
                "Wczytywanie danych",
                str(error),
            )

    def clamp_range(
        self,
        start: int,
        end: int,
    ) -> tuple[int, int]:
        if self.signals is None:
            return 0, 0

        last_index = len(self.signals) - 1

        start = max(
            0,
            min(int(start), last_index),
        )

        end = max(
            0,
            min(int(end), last_index),
        )

        if end < start:
            return end, start

        return start, end

    def update_range_labels(self) -> None:
        start, end = self.clamp_range(
            self.start_var.get(),
            self.end_var.get(),
        )

        self.start_var.set(start)
        self.end_var.set(end)

        self.start_entry.delete(
            0,
            tk.END,
        )

        self.start_entry.insert(
            0,
            str(start),
        )

        self.end_entry.delete(
            0,
            tk.END,
        )

        self.end_entry.insert(
            0,
            str(end),
        )

        if self.signals is not None:
            count = end - start + 1
        else:
            count = 0

        self.beat_count_var.set(f"Liczba uderzeń: " f"{count if count else '–'}")

    def on_range_change(
        self,
        _event=None,
    ) -> None:
        start, end = self.clamp_range(
            self.start_var.get(),
            self.end_var.get(),
        )

        self.start_var.set(start)
        self.end_var.set(end)

        self.update_range_labels()
        self.plot_segment(start, end)

    def on_range_entry(
        self,
        _event=None,
    ) -> None:
        try:
            start = int(self.start_entry.get())

            end = int(self.end_entry.get())

        except ValueError:
            messagebox.showwarning(
                "Zakres",
                "Zakres musi zawierać liczby całkowite.",
            )
            return

        start, end = self.clamp_range(
            start,
            end,
        )

        self.start_var.set(start)
        self.end_var.set(end)

        self.update_range_labels()
        self.plot_segment(start, end)

    def plot_segment(
        self,
        start: int,
        end: int,
    ) -> None:
        if self.signals is None:
            return

        displayed = self.signals[start : end + 1].reshape(-1)

        self.axis.clear()

        self.axis.plot(
            displayed,
            linewidth=0.9,
        )

        self.axis.set_title(f"Segment uderzeń {start}–{end}")

        self.axis.set_xlabel("próbki sklejonych okien")

        self.axis.set_ylabel("znormalizowana amplituda")

        self.canvas.draw_idle()

        if self.last_analysis is not None:
            previous = self.last_analysis

            if (
                start,
                end,
            ) == (
                previous["start"],
                previous["end"],
            ):
                self.probability_frame.configure(
                    text=("Prawdopodobieństwo " "przynależności do klas")
                )

                self.set_status(
                    (
                        f"Analiza segmentu {start}–{end} "
                        f"zakończona "
                        f"(liczba uderzeń: "
                        f"{previous['beat_count']})."
                    ),
                    "green",
                )

            else:
                self.probability_frame.configure(
                    text=(
                        "Wyniki ostatniej analizy: "
                        f"{previous['start']}–"
                        f"{previous['end']}"
                    )
                )

                self.set_status(
                    (
                        "Wybrano nowy zakres."
                    ),
                    "orange",
                )


    def freeze_window_size(self) -> None:
        self.update_idletasks()

        width = self.winfo_width()
        height = self.winfo_height()

        self.geometry(f"{width}x{height}")
        self.minsize(width, height)
        self.resizable(True, True)

    def analyze_segment(self) -> None:
        if self.model is None or self.signals is None:
            self.set_status(
                "Brak modelu lub danych do analizy.",
                "orange",
            )
            return

        start, end = self.clamp_range(
            self.start_var.get(),
            self.end_var.get(),
        )

        repeated_same_range = (
            self.last_analysis is not None
            and start == self.last_analysis["start"]
            and end == self.last_analysis["end"]
        )

        if self.last_analysis is not None:
            previous_start = self.last_analysis["start"]
            previous_end = self.last_analysis["end"]

            if not repeated_same_range:
                replace_result = messagebox.askyesno(
                    "Nowa analiza",
                    (
                        "Istnieją niezapisane wyniki dla zakresu "
                        f"{previous_start}–{previous_end}.\n\n"
                        f"Czy przeanalizować zakres {start}–{end} "
                        "i zastąpić poprzednie wyniki?"
                    ),
                )

                if not replace_result:
                    return

        if self.rr_feature is None:
            rr_segment = None
        else:
            rr_segment = self.rr_feature[start : end + 1]

        try:
            probabilities, predictions = predict_probabilities(
                self.model,
                self.signals[start : end + 1],
                rr_segment,
                self.device,
            )

        except Exception as error:
            messagebox.showerror(
                "Analiza",
                str(error),
            )
            return

        counts = Counter(self.class_names[int(index)] for index in predictions)

        mean_probabilities = {
            name: float(probabilities[:, index].mean())
            for index, name in enumerate(self.class_names)
        }

        for name in self.class_names:
            self.class_result_vars[name].set(
                (f"{name}: {counts[name]}    " f"(p̄ = {mean_probabilities[name]:.2f})")
            )

        mean_confidence = float(probabilities.max(axis=1).mean())

        self.confidence_var.set(
            ("Średnia pewność decyzji modelu: " f"{mean_confidence:.2f}")
        )

        self.last_analysis = {
            "start": start,
            "end": end,
            "beat_count": end - start + 1,
            "probabilities": probabilities,
            "predictions": predictions,
            "counts": dict(counts),
            "mean_probabilities": mean_probabilities,
            "mean_confidence": mean_confidence,
            "records": ",".join(dict.fromkeys(self.record_ids[start : end + 1])),
            "model_name": self.loaded_model_name,
            "class_names": self.class_names,
            "signals": self.signals[start : end + 1].copy(),
        }

        self.probability_frame.configure(text="Prawdopodobieństwo przynależności do klas")

        if repeated_same_range:
            status_text = (
                f"Ponownie przeanalizowano segment {start}–{end} "
            )
        else:
            status_text = (
                f"Analiza segmentu {start}–{end} zakończona "
            )

        self.set_status(
            status_text,
            "green",
        )

    def refresh_class_labels(self) -> None:
        for child in self.probability_frame.winfo_children():
            if child is not self.probability_info_button:
                child.destroy()

        self.class_result_vars = {}

        for name in self.class_names:
            variable = tk.StringVar(value=f"{name}: –")

            self.class_result_vars[name] = variable

            ttk.Label(
                self.probability_frame,
                textvariable=variable,
            ).pack(
                anchor="w",
                padx=6,
                pady=2,
            )

    def show_range_info(self) -> None:
        messagebox.showinfo(
            "Zakres analizy",
            (
                "Suwaki określają indeksy pierwszego "
                "i ostatniego uderzenia, które zostaną "
                "przekazane do modelu. Każde uderzenie "
                "odpowiada osobnemu oknu sygnału EKG.\n\n"
                "Po zmianie zakresu kliknij "
                "„Analizuj segment”, aby obliczyć "
                "nowe wyniki."
            ),
        )

    def show_result_info(self) -> None:
        messagebox.showinfo(
            "Interpretacja wyników",
            (
                "N — pobudzenia prawidłowe i zaliczane "
                "do grupy normalnej\n"
                "S — pobudzenia nadkomorowe\n"
                "V — przedwczesne pobudzenia komorowe\n\n"
                "Wartość p̄ oznacza średnie "
                "prawdopodobieństwo przypisane danej "
                "klasie w analizowanym segmencie.\n\n"
                "Wyniki są rezultatem modelu uczenia "
                "maszynowego i mają charakter wyłącznie "
                "badawczy. Aplikacja nie jest wyrobem "
                "medycznym i nie może służyć do "
                "samodzielnej diagnozy."
            ),
        )

    def export_to_excel(
        self,
        destination: str | Path,
    ) -> None:
        analysis = self.last_analysis

        if analysis is None:
            raise RuntimeError("Brak wyników do eksportu.")

        summary = pd.DataFrame(
            [
                {
                    "rekordy": analysis["records"],
                    "początek": analysis["start"],
                    "koniec": analysis["end"],
                    "liczba_uderzeń": (analysis["beat_count"]),
                    "dominująca_klasa": max(
                        analysis["counts"],
                        key=analysis["counts"].get,
                    ),
                    "średnia_pewność": (analysis["mean_confidence"]),
                }
            ]
        )

        probabilities = np.asarray(analysis["probabilities"])

        predictions = np.asarray(analysis["predictions"])

        rows = []

        for offset, predicted_index in enumerate(predictions):
            row = {
                "indeks_uderzenia": (int(analysis["start"]) + offset),
                "przewidziana_klasa": (self.class_names[int(predicted_index)]),
            }

            row.update(
                {
                    f"p_{name}": float(
                        probabilities[
                            offset,
                            index,
                        ]
                    )
                    for index, name in enumerate(self.class_names)
                }
            )

            rows.append(row)

        with pd.ExcelWriter(destination) as writer:
            summary.to_excel(
                writer,
                sheet_name="Podsumowanie",
                index=False,
            )

            pd.DataFrame(rows).to_excel(
                writer,
                sheet_name="Uderzenia",
                index=False,
            )

    def save_results(self) -> bool:
        if self.last_analysis is None:
            messagebox.showwarning(
                "Zapis",
                ("Najpierw przeprowadź " "analizę segmentu."),
            )
            return False

        dialog = SaveDialog(self)
        self.wait_window(dialog)

        if dialog.result is None:
            return False

        try:
            model_name = self.last_analysis["model_name"]

            analysis_id = self.last_analysis.get("database_id")

            if analysis_id is None:
                analysis_id = save_analysis(
                    self.last_analysis,
                    self.class_names,
                    model_name,
                    DEFAULT_DATABASE_PATH,
                )

                self.last_analysis["database_id"] = analysis_id

            start = self.last_analysis["start"]
            end = self.last_analysis["end"]

            if dialog.result in (1, 3):
                excel_path = filedialog.asksaveasfilename(
                    title="Zapisz dane jako Excel",
                    defaultextension=".xlsx",
                    filetypes=[
                        (
                            "Pliki Excel",
                            "*.xlsx",
                        )
                    ],
                    initialfile=(f"segment_{start}-{end}.xlsx"),
                )

                if excel_path:
                    self.export_to_excel(excel_path)

            if dialog.result in (2, 3):
                image_path = filedialog.asksaveasfilename(
                    title="Zapisz wykres jako PNG",
                    defaultextension=".png",
                    filetypes=[
                        (
                            "Pliki PNG",
                            "*.png",
                        )
                    ],
                    initialfile=(f"segment_{start}-{end}.png"),
                )

                if image_path:
                    figure, axis = plt.subplots(figsize=(9.5, 3.9))

                    axis.plot(
                        self.last_analysis["signals"].reshape(-1),
                        linewidth=0.9,
                    )

                    axis.set(
                        title=(f"Segment uderzeń " f"{start}–{end}"),
                        xlabel=("próbki sklejonych okien"),
                        ylabel=("znormalizowana amplituda"),
                    )

                    try:
                        figure.savefig(
                            image_path,
                            dpi=160,
                            bbox_inches="tight",
                        )
                    finally:
                        plt.close(figure)

            self.last_analysis = None

            self.probability_frame.configure(
                text=("Prawdopodobieństwo " "przynależności do klas")
            )

            messagebox.showinfo(
                "Zapis",
                ("Wyniki zapisano w bazie " f"(ID analizy: {analysis_id})."),
            )

            return True

        except Exception as error:
            messagebox.showerror(
                "Zapis",
                str(error),
            )
            return False

    def on_close(self) -> None:
        if self.last_analysis is not None:
            answer = messagebox.askyesnocancel(
                "Zamknij",
                (
                    "Masz niezapisane wyniki analizy. "
                    "Czy zapisać je przed zamknięciem?"
                ),
            )

            if answer is None:
                return

            if answer and not self.save_results():
                return

        elif not messagebox.askokcancel(
            "Zamknij",
            "Zamknąć aplikację?",
        ):
            return

        plt.close("all")
        self.destroy()


class SaveDialog(tk.Toplevel):
    """Choose optional export formats; SQLite saving is always performed."""

    def __init__(
        self,
        master: tk.Misc,
    ) -> None:
        super().__init__(master)

        self.title("Zapis wyników")
        self.resizable(False, False)
        self.result: int | None = None

        self.transient(master)
        self.grab_set()

        ttk.Label(
            self,
            text=("Wyniki zostaną zapisane w bazie. " "Wybierz dodatkowy eksport:"),
        ).pack(
            padx=20,
            pady=(20, 10),
        )

        self.choice_var = tk.IntVar(value=1)

        options = ttk.Frame(self)

        options.pack(
            padx=20,
            pady=6,
            anchor="w",
        )

        for text, value in (
            ("Tylko baza danych", 0),
            ("Plik Excel (.xlsx)", 1),
            ("Wykres (.png)", 2),
            ("Oba pliki", 3),
        ):
            ttk.Radiobutton(
                options,
                text=text,
                variable=self.choice_var,
                value=value,
            ).pack(
                anchor="w",
                pady=2,
            )

        buttons = ttk.Frame(self)

        buttons.pack(
            fill=tk.X,
            padx=20,
            pady=(10, 15),
        )

        ttk.Button(
            buttons,
            text="Anuluj",
            command=self.cancel,
        ).pack(side=tk.LEFT)

        ttk.Button(
            buttons,
            text="Zapisz",
            command=self.confirm,
        ).pack(side=tk.RIGHT)

        self.bind(
            "<Escape>",
            lambda _event: self.cancel(),
        )

        self.bind(
            "<Return>",
            lambda _event: self.confirm(),
        )

    def confirm(self) -> None:
        self.result = int(self.choice_var.get())

        self.destroy()

    def cancel(self) -> None:
        self.result = None
        self.destroy()


def main() -> None:
    initialize_database(DEFAULT_DATABASE_PATH)

    CardioAssistantApp().mainloop()


if __name__ == "__main__":
    main()
