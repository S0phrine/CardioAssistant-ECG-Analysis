# CardioAssistant — klasyfikacja pobudzeń EKG

[Polski](#polski) | [English](#english)

## Polski

Uporządkowana implementacja eksperymentów z modelami **k najbliższych sąsiadów (k-NN)**, **Random Forest** oraz **jednowymiarową konwolucyjną siecią neuronową (1D CNN)**, przeprowadzonych w ramach mojej pracy inżynierskiej dotyczącej klasyfikacji pobudzeń serca w sygnałach EKG.

> Projekt jest edukacyjnym prototypem badawczym. Nie jest wyrobem medycznym i nie może być wykorzystywany do diagnozy klinicznej.

<br>

### Zakres projektu

Potok przetwarzania klasyfikuje pojedyncze pobudzenia serca do trzech grup zgodnych ze standardem AAMI:

- **N** — pobudzenia prawidłowe oraz pobudzenia z blokiem odnogi pęczka Hisa,
- **S** — pobudzenia nadkomorowe,
- **V** — pobudzenia komorowe.

Projekt obsługuje dwa scenariusze oceny:

- **intra-patient** — stratyfikowany podział pobudzeń w proporcji 70/30,
- **inter-patient** — uczenie i testowanie na rozłącznych zbiorach pacjentów zgodnie z podziałem DS1/DS2 stosowanym w literaturze.

Surowe zapisy pochodzą z bazy [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). Nie są one dołączone do repozytorium.

<br>

### Potok przetwarzania

1. Odczyt pierwszego kanału EKG oraz eksperckich adnotacji.
2. Usunięcie dryfu linii izoelektrycznej za pomocą filtru medianowego.
3. Zastosowanie filtru zaporowego 50 Hz oraz filtru pasmowoprzepustowego 0,5–40 Hz.
4. Normalizacja każdego zapisu względem maksymalnej wartości bezwzględnej.
5. Wyodrębnienie 180-próbkowego okna wycentrowanego wokół każdego oznaczonego załamka R.
6. Odrzucenie niemal płaskich segmentów (`odchylenie standardowe < 0.05`).
7. Dla modeli klasycznych obliczenie 12 ręcznie zaprojektowanych cech:
   - maksimum, minimum, średnia i odchylenie standardowe,
   - suma bezwzględnych różnic pierwszego rzędu,
   - sumy modułów FFT w pasmach 0–10, 10–40 oraz 40–100 Hz,
   - energie trzech grup współczynników falki `db4`,
   - poprzedni odstęp RR.
8. Uczenie modelu k-NN albo Random Forest i ocena za pomocą miary macro F1.
9. Alternatywnie przekazanie standaryzowanych, 180-próbkowych segmentów bezpośrednio do sieci 1D CNN. Jej końcowy wariant inter-patient dodaje standaryzowaną wartość `poprzedni RR / następny RR` jako stały drugi kanał wejściowy.

W przypadku k-NN `StandardScaler` i `SMOTE` są dopasowywane **wewnątrz** potoku uczącego, co zapobiega przenikaniu informacji ze zbioru testowego do procesu uczenia. Pierwszy niedostępny odstęp RR jest uzupełniany medianą odstępów RR z tego samego rekordu.

<br>

### Struktura repozytorium

```text
cardio_ml/
  database.py        zapis wyników w SQLite
  cnn.py             zbiór PyTorch i trójwarstwowa sieć 1D CNN
  config.py          mapowanie AAMI, podziały rekordów i nazwy cech
  data.py            filtracja, segmentacja i przygotowanie danych
  features.py        ekstrakcja 12 cech numerycznych
  experiment.py      podział, ocena i zapis artefaktów
  segments.py        segmenty CNN o stałej długości i sąsiednie odstępy RR
experiments/
  train_cnn.py
  train_knn.py
  train_random_forest.py
app/
  app_gui.py         aplikacja desktopowa do analizy segmentów
models/
  best_model.pt      wagi wytrenowanego, końcowego modelu CNN
  best_model.meta.json
```

<br>

### Instalacja

Wymagany jest Python 3.10 lub nowszy. Projekt zweryfikowano w Pythonie 3.12 na systemie Linux z użyciem CPU oraz w Pythonie 3.13.2 na macOS z użyciem Apple MPS. Tkinter musi być zainstalowany razem z Pythonem; na Linuksie może wymagać pakietu systemowego python3-tk. Do uruchomienia aplikacji potrzebne jest środowisko graficzne.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Pobierz MIT-BIH Arrhythmia Database z PhysioNet i umieść pliki w jednym katalogu. Dla każdego rekordu katalog powinien zawierać pliki `.hea`, `.dat` oraz `.atr`, na przykład `100.hea`, `100.dat` i `100.atr`.

<br>

### Uruchamianie eksperymentów

Polecenia należy uruchamiać z katalogu głównego repozytorium. Ścieżkę `data/mitdb` należy zastąpić ścieżką do pobranej bazy.

<br>

#### Końcowe konfiguracje z pracy inżynierskiej

```bash
python -m experiments.train_knn \
  --mode intra --data-dir data/mitdb --quick

python -m experiments.train_knn \
  --mode inter --data-dir data/mitdb --quick

python -m experiments.train_random_forest \
  --mode intra --balance smote --data-dir data/mitdb --quick

python -m experiments.train_random_forest \
  --mode inter --balance none --data-dir data/mitdb --quick
```

<br>

#### 1D CNN

Domyślne polecenie dla CNN wykorzystuje końcową, dwukanałową konfigurację z pracy. Ziarno `864` odpowiada ziarnu zapisanemu dla przebiegu referencyjnego; dokładne wyniki mogą się różnić zależnie od sprzętu i wersji PyTorch.

```bash
python -m experiments.train_cnn \
  --mode intra --data-dir data/mitdb

python -m experiments.train_cnn \
  --mode inter --data-dir data/mitdb
```

Aby uruchomić model bazowy wykorzystujący wyłącznie sygnał EKG, dodaj `--no-rr-channel`. Podczas uczenia automatycznie wybierane jest CUDA, Apple MPS albo CPU; konkretne urządzenie można wskazać przez `--device`.

W przypadku modeli klasycznych usuń `--quick`, aby ponownie przeprowadzić dobór hiperparametrów z pięciokrotną walidacją krzyżową. Aby zastosować bardziej rygorystyczną walidację grupowaną według pacjentów podczas strojenia modelu inter-patient, dodaj `--group-cv`.

Uruchomienia modeli klasycznych zapisują w katalogu `results/`:

- dopasowany potok (`.joblib`),
- metryki i wybrane hiperparametry (`.json`),
- macierz pomyłek (`.png`).

Uruchomienia CNN zapisują pliki `checkpoint.pt`, `checkpoint.meta.json`, `metrics.json`, `predictions.csv` i `confusion_matrix.png` w osobnym katalogu danego uruchomienia. Aby użyć modelu w aplikacji, wybierz checkpoint oraz odpowiadający mu plik metadanych. Opcja `--quick` dla modeli klasycznych nadal wykonuje pięciokrotną walidację krzyżową dla jednej konfiguracji.

<br>

### Aplikacja desktopowa

Aplikacja Tkinter wczytuje wybrane rekordy MIT-BIH, odtwarza przetwarzanie wstępne dla CNN, klasyfikuje wskazany zakres pobudzeń i zapisuje wyniki w bazie SQLite. Umożliwia również eksport szczegółowych prawdopodobieństw do Excela oraz wyświetlanego sygnału do pliku PNG.

Uruchom ją z katalogu głównego repozytorium:

```bash
python -m app.app_gui
```

Aplikacja automatycznie tworzy plik `data/cardioassistant.db`. Baza danych i pobrane rekordy MIT-BIH są ignorowane przez Git. Metadane modelu znajdują się obok pliku wag, dzięki czemu podczas predykcji stosowane są ta sama długość okna, kolejność klas i standaryzowana cecha `poprzedni RR / następny RR`, co podczas uczenia.

Interfejs jest wyłącznie demonstracją badawczą; jego wyniki nie stanowią diagnozy medycznej.

<br>

### Wyniki referencyjne z pracy inżynierskiej

Wartości dokumentują pierwotne eksperymenty i nie są wpisane na stałe w skryptach.

| Model i scenariusz | Accuracy | Macro F1 |
|---|---:|---:|
| k-NN, intra-patient | 0.97 | 0.92 |
| k-NN, inter-patient | 0.85 | 0.53 |
| Random Forest + SMOTE, intra-patient | 0.98 | 0.95 |
| Random Forest, inter-patient | 0.94 | 0.60 |
| 1D CNN, intra-patient | 0.98 | 0.954 |
| 1D CNN, inter-patient, tylko sygnał | 0.91 | 0.741 |
| 1D CNN, inter-patient, sygnał + kanał RR | 0.97 | 0.863 |

Są to historyczne wyniki z notatników i pracy inżynierskiej, **a nie nowy benchmark uporządkowanego kodu**. Pełne uczenie nie zostało ponownie wykonane dla tej wersji. Poza różnicami wersji bibliotek i ziaren losowych refaktoryzacja celowo uzupełnia brakujące wartości RR osobno dla każdego rekordu, podczas gdy notatniki łączyły dostępne odstępy ze wszystkich wczytanych rekordów. Z tego powodu nie jest deklarowane dokładne odtworzenie wyników. Dołączone wagi pochodzą z oryginalnego checkpointu autorki, a nie z ponownego uczenia.

Podzbiór walidacyjny CNN jest stratyfikowanym podzbiorem pobudzeń ze zbioru uczącego. W eksperymencie intra-patient te same rekordy występują również w zbiorze uczącym i testowym, dlatego wyniku nie należy interpretować jako skuteczności dla całkowicie nowych osób.

<br>

### Zakres i ograniczenia

- Segmentacja wykorzystuje dostarczone adnotacje `.atr`, w tym położenia pobudzeń i ich przynależność do klas N/S/V. Ta wersja nie zawiera automatycznego detektora załamków R i nie obsługuje dowolnych, niezaanotowanych plików EKG.
- Odstępy RR są obliczane pomiędzy zachowanymi oknami N/S/V. Odrzucone lub pominięte pobudzenia mogą więc wydłużać odstęp; odpowiada to podejściu zastosowanemu w eksperymentach źródłowych, a nie kompletnemu detektorowi rytmu.
- Następny odstęp RR i filtracja nieprzyczynowa wymagają zapisów offline. Nie jest to implementacja czasu rzeczywistego.
- Wykresy sklejają standaryzowane okna pobudzeń; nie przedstawiają pierwotnego ciągłego zapisu ani odstępów pomiędzy pobudzeniami.
- Prawdopodobieństwa softmax są wynikami modelu, a nie skalibrowaną pewnością kliniczną.
- Aplikacja korzysta obecnie ze współczynnika RR zapisanego w metadanych oraz z trybu `eval()`. Są to poprawki względem pierwotnego GUI, dlatego wyniki starej wersji aplikacji mogą się różnić.
- Historia z oryginalnej bazy `baza.db` nie jest migrowana; aplikacja tworzy nową bazę z angielskimi nazwami tabel w katalogu `data/`.
- Wczytywanie dużych zbiorów i predykcja odbywają się w wątku GUI, dlatego okno może chwilowo nie odpowiadać. Przetwarzanie w tle jest możliwym kierunkiem dalszego rozwoju.

<br>

### Weryfikacja

```bash
python -m unittest discover -s tests -v
```

Testy obejmują poprawne wczytywanie oryginalnych wag modelu, powtarzalność predykcji dla różnych rozmiarów partii oraz pełny przepływ na syntetycznym rekordzie EKG zapisanym w formacie WFDB (WaveForm DataBase): od wczytania i przetworzenia danych, przez wykonanie predykcji, aż po zapis wyników w SQLite i eksport do Excela. Sprawdzają również odrzucanie nieprawidłowych danych podczas zapisu do bazy oraz zgodność plików modelu tworzonych podczas treningu z aplikacją. Testy weryfikują działanie oprogramowania, a nie jego skuteczność diagnostyczną. Interfejs graficzny i elementy zależne od systemu operacyjnego wymagają dodatkowego sprawdzenia lokalnego.

<br>

### Źródło projektu

Repozytorium zawiera uporządkowaną wersję kodu opracowanego w ramach mojej pracy inżynierskiej dotyczącej klasyfikacji pobudzeń serca w sygnałach EKG.

W projekcie wykorzystano dane z bazy [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), którą należy pobrać oddzielnie z PhysioNet zgodnie z warunkami korzystania ze zbioru. Dane źródłowe ani lokalna historia analiz nie są udostępniane w tym repozytorium. Informacje dotyczące cytowania i ponownego wykorzystania danych znajdują się na stronie zbioru.

<br>

---

<br>

## English

A refactored implementation of the **k-nearest neighbours (k-NN)**, **Random Forest**, and **one-dimensional convolutional neural network (1D CNN)** experiments conducted as part of my engineering thesis on heartbeat classification in ECG signals.

> This project is an educational research prototype. It is not a medical device and must not be used for clinical diagnosis.

<br>

### Project scope

The pipeline classifies individual heartbeats into three AAMI groups:

- **N** — normal and bundle branch block beats,
- **S** — supraventricular ectopic beats,
- **V** — ventricular ectopic beats.

It supports two evaluation scenarios:

- **intra-patient** — a stratified 70/30 beat-level split,
- **inter-patient** — training and testing on disjoint patient sets according to the DS1/DS2 split commonly used in the literature.

The raw recordings come from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). They are not included in this repository.

<br>

### Processing pipeline

1. Read the first ECG channel and expert annotations.
2. Remove baseline drift with a median filter.
3. Apply a 50 Hz notch filter and a 0.5–40 Hz band-pass filter.
4. Normalize each recording by its maximum absolute amplitude.
5. Extract a 180-sample window centred on each annotated R peak.
6. Reject nearly flat segments (`standard deviation < 0.05`).
7. For the classical models, calculate 12 handcrafted features:
   - maximum, minimum, mean and standard deviation,
   - sum of absolute first differences,
   - sums of FFT magnitudes in 0–10, 10–40 and 40–100 Hz bands,
   - energies of three `db4` wavelet coefficient groups,
   - previous RR interval.
8. Train k-NN or Random Forest and evaluate it using macro F1.
9. Alternatively, pass the standardized 180-sample segments directly to the 1D CNN. Its final inter-patient variant adds a standardized `previous RR / next RR` value as a constant second input channel.

For k-NN, `StandardScaler` and `SMOTE` are fitted **inside** the training pipeline, preventing information from the test set from leaking into model training. The first unavailable RR interval is imputed with the median RR from the same recording.

<br>

### Repository structure

```text
cardio_ml/
  database.py        SQLite result storage
  cnn.py             PyTorch dataset and three-layer 1D CNN
  config.py          AAMI mappings, record splits and feature names
  data.py            filtering, segmentation and dataset preparation
  features.py        extraction of the 12 numerical features
  experiment.py      splitting, evaluation and artifact saving
  segments.py        fixed-length CNN segments and adjacent RR intervals
experiments/
  train_cnn.py
  train_knn.py
  train_random_forest.py
app/
  app_gui.py         desktop application for segment analysis
models/
  best_model.pt      trained final CNN weights
  best_model.meta.json
```

<br>

### Setup

Python 3.10 or newer is required. The project was verified with Python 3.12 on Linux using CPU and with Python 3.13.2 on macOS using Apple MPS. Tkinter must be installed with Python; on Linux, it may require the python3-tk system package. A graphical desktop environment is required to run the application.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the MIT-BIH Arrhythmia Database from PhysioNet and keep the files in one directory. For every record, the directory should contain `.hea`, `.dat` and `.atr` files, for example `100.hea`, `100.dat` and `100.atr`.

<br>

### Running the experiments

Run the commands from the repository root. Replace `data/mitdb` with the path to your downloaded database.

<br>

#### Final thesis configurations

```bash
python -m experiments.train_knn \
  --mode intra --data-dir data/mitdb --quick

python -m experiments.train_knn \
  --mode inter --data-dir data/mitdb --quick

python -m experiments.train_random_forest \
  --mode intra --balance smote --data-dir data/mitdb --quick

python -m experiments.train_random_forest \
  --mode inter --balance none --data-dir data/mitdb --quick
```

<br>

#### 1D CNN

The default CNN command uses the final two-channel configuration from the thesis. Seed `864` matches the seed recorded for the reference run; exact metrics may still vary between hardware and PyTorch versions.

```bash
python -m experiments.train_cnn \
  --mode intra --data-dir data/mitdb

python -m experiments.train_cnn \
  --mode inter --data-dir data/mitdb
```

To run the baseline model using only the ECG signal, add `--no-rr-channel`. Training automatically selects CUDA, Apple MPS or CPU; use `--device` to choose one explicitly.

For the classical models, remove `--quick` to repeat hyperparameter selection with five-fold cross-validation. For a stricter patient-grouped validation while tuning the inter-patient model, add `--group-cv`.

Classical-model runs save the following files in `results/`:

- fitted pipeline (`.joblib`),
- metrics and selected hyperparameters (`.json`),
- confusion matrix (`.png`).

CNN runs save `checkpoint.pt`, `checkpoint.meta.json`, `metrics.json`, `predictions.csv` and `confusion_matrix.png` in a run-specific subdirectory. Choose the checkpoint and its matching metadata in the application to use it. `--quick` for classical models still performs five-fold CV on one configuration.

<br>

### Desktop application

The Tkinter application loads selected MIT-BIH records, reproduces the CNN preprocessing, classifies a chosen range of heartbeats and stores the result in SQLite. It can additionally export detailed probabilities to Excel and the displayed signal to PNG.

Run it from the repository root:

```bash
python -m app.app_gui
```

The application creates `data/cardioassistant.db` automatically. The database and downloaded MIT-BIH recordings are ignored by Git. Model metadata are kept next to the weights so that inference uses the same window length, class order and standardized `previous RR / next RR` feature as training.

The interface is a research demonstration only; its output is not a medical diagnosis.

<br>

### Reference results from the thesis

These values document the original experiments and are not hard-coded in the scripts.

| Model and scenario | Accuracy | Macro F1 |
|---|---:|---:|
| k-NN, intra-patient | 0.97 | 0.92 |
| k-NN, inter-patient | 0.85 | 0.53 |
| Random Forest + SMOTE, intra-patient | 0.98 | 0.95 |
| Random Forest, inter-patient | 0.94 | 0.60 |
| 1D CNN, intra-patient | 0.98 | 0.954 |
| 1D CNN, inter-patient, signal only | 0.91 | 0.741 |
| 1D CNN, inter-patient, signal + RR channel | 0.97 | 0.863 |

These are historical notebook/thesis results, **not a new benchmark of this refactored code**. Full training has not been rerun for this release. Beyond library versions and random seeds, the refactor deliberately imputes missing RR values per recording; the notebooks pooled available intervals across loaded recordings. Therefore exact reproduction is not claimed. The bundled weights are the author's original checkpoint, not newly trained weights.

The CNN validation subset is a stratified beat-level subset of the training pool. The intra-patient experiment also shares recordings between train/test; it must not be interpreted as performance on entirely unseen people.

<br>

### Scope and limitations

- Segmentation uses supplied `.atr` annotations, including beat locations and N/S/V eligibility. There is no automatic R-peak detector in this version. Arbitrary unannotated ECG files are not supported.
- RR is calculated between retained N/S/V windows. Discarded or excluded beats can therefore lengthen an interval; this follows the retained-beat approach of the source experiments rather than a complete rhythm detector.
- The next RR interval and noncausal filtering require offline recordings. This is not a real-time streaming implementation.
- Plots concatenate standardized beat windows; they do not show the original continuous recording or preserve the gaps between beats.
- Softmax probabilities are model scores, not calibrated clinical certainty.
- The app now uses the metadata's RR ratio and `eval()` mode. Those are fixes to the original GUI, so old application outputs may differ.
- Original `baza.db` history is not migrated: the app creates a new database using English table names under `data/`.
- Large loads and inference run on the GUI thread; the window can pause while processing. Background processing is a possible future improvement.

<br>

### Verification

```bash
python -m unittest discover -s tests -v
```

The tests cover correct loading of the original model weights, repeatable predictions across different batch sizes, and the complete processing flow using a synthetic ECG record stored in the WFDB (WaveForm DataBase) format: from loading and preprocessing the data, through generating predictions, to saving the results in SQLite and exporting them to Excel. They also verify that invalid data are rejected during database writes and that model files created during training are compatible with the application. The tests verify software behaviour, not diagnostic performance. The graphical user interface and operating-system-specific behaviour require additional local verification.

<br>

### Source

The repository contains an organized version of the code developed as part of my engineering thesis on heartbeat classification in ECG signals.

The project uses data from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), which must be downloaded separately from PhysioNet in accordance with the dataset’s terms of use. Neither the source data nor the local analysis history is included in this repository. Information about citation and data reuse is available on the dataset page.
