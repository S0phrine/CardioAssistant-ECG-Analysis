# Verification — 2026-09-19

Environment: Python 3.12, Linux, CPU. Direct package versions are recorded in
`requirements-tested.txt`; normal installation uses `requirements.txt`.

## Completed

- All Python modules passed compilation.
- All three training command-line interfaces passed `--help`.
- Three integration tests passed (`python -m unittest discover -s tests -v`).
- Original CNN weights loaded strictly with no missing/unexpected layers.
- Predictions were repeatable and agreed across different batch sizes.
- Synthetic WFDB record passed preprocessing, feature extraction, CNN
  prediction, SQLite persistence and Excel export checks.
- Invalid class indices were rejected without inserting another analysis.
- Newly exported training checkpoints and metadata loaded in the app.
- Real MIT-BIH record 100 produced 2271 retained windows, 12 classical features
  per beat and finite three-class CNN probabilities.

- Real-record CNN segments also matched the original notebook loader within
  a maximum absolute difference of 5.4e-7; labels were identical.

## Not completed

- Full retraining/grid search and full-dataset metric reproduction.
- Interactive desktop testing of widgets, dialogs and OS-specific behavior.
- Clinical validation or calibration of model probabilities.

## Before demonstrating on your computer

1. Install dependencies and run `python -m app.app_gui` from the repository root.
2. Select your MIT-BIH folder and initialize the bundled model/metadata.
3. Load record `100`, select a small range and analyze it twice.
4. Save to SQLite, Excel and PNG; verify the same range in all three.
5. Change the selected range and test saving the previous analysis: the PNG
   should retain the analyzed range. Try cancelling save and closing the app.

## Refactoring notes

The original uploaded notebooks, model and historical database were not
edited. The distributable app uses a new database schema/location and fixes
the RR input and evaluation mode. Missing RR values are now imputed per
record. See README for experimental limitations; historical thesis scores are
not presented as measured scores for the refactored release.
