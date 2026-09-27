# Weryfikacja / Verification — 2026-09-19

[Polski](#polski) | [English](#english)

## Polski

Środowisko: Python 3.12, Linux, CPU. Dokładne wersje bezpośrednich zależności
użytych podczas weryfikacji zapisano w pliku `requirements-tested.txt`;
standardowa instalacja korzysta z `requirements.txt`.

### Wykonane

- Wszystkie moduły Pythona przeszły kompilację.
- Interfejsy wiersza poleceń wszystkich trzech skryptów treningowych poprawnie obsłużyły opcję `--help`.
- Trzy testy integracyjne zakończyły się powodzeniem (`python -m unittest discover -s tests -v`).
- Oryginalne wagi CNN zostały wczytane w trybie ścisłym, bez brakujących ani nieoczekiwanych warstw.
- Predykcje były powtarzalne i zgodne dla różnych rozmiarów partii.
- Syntetyczny rekord WFDB przeszedł cały potok: przetwarzanie wstępne, ekstrakcję cech, predykcję CNN, zapis do SQLite oraz eksport do Excela.
- Nieprawidłowe indeksy klas zostały odrzucone bez zapisania dodatkowej analizy w bazie.
- Nowo wyeksportowane checkpointy treningowe i metadane zostały poprawnie wczytane w aplikacji.
- Dla rzeczywistego rekordu MIT-BIH nr 100 uzyskano 2271 zachowanych okien, 12 cech klasycznych dla każdego pobudzenia oraz skończone prawdopodobieństwa dla trzech klas CNN.
- Segmenty CNN dla rzeczywistego rekordu były zgodne z pierwotnym modułem wczytującym z notebooka; maksymalna różnica bezwzględna wyniosła `5.4e-7`, a etykiety były identyczne.

### Niewykonane

- Pełne ponowne uczenie, przeszukiwanie siatki hiperparametrów ani odtworzenie wyników dla całego zbioru.
- Interaktywne testowanie widżetów, okien dialogowych i zachowania zależnego od systemu operacyjnego.
- Walidacja kliniczna ani kalibracja prawdopodobieństw modelu.

### Przed demonstracją na własnym komputerze

1. Zainstaluj zależności i uruchom `python -m app.app_gui` z głównego katalogu repozytorium.
2. Wskaż katalog z bazą MIT-BIH i zainicjalizuj dołączony model wraz z metadanymi.
3. Wczytaj rekord `100`, wybierz niewielki zakres i przeanalizuj go dwukrotnie.
4. Zapisz wyniki do SQLite, Excela i PNG, a następnie sprawdź zgodność zakresu we wszystkich trzech formatach.
5. Zmień wybrany zakres i przetestuj zapis poprzedniej analizy — plik PNG powinien zachować wcześniej analizowany zakres. Sprawdź również anulowanie zapisu i zamykanie aplikacji.

### Uwagi dotyczące refaktoryzacji

Oryginalne notebooki, model i historyczna baza danych nie zostały zmodyfikowane.
Wersja aplikacji przeznaczona do udostępnienia korzysta z nowego schematu i lokalizacji bazy danych oraz poprawia obsługę wejścia RR i trybu ewaluacji modelu. Brakujące wartości RR są obecnie uzupełniane osobno dla każdego rekordu. Ograniczenia eksperymentalne opisano w README; historyczne wyniki pracy inżynierskiej nie są przedstawiane jako wyniki zmierzone dla wersji po refaktoryzacji.

---

## English

Environment: Python 3.12, Linux, CPU. Direct package versions used for the
verification run are recorded in `requirements-tested.txt`; normal installation
uses `requirements.txt`.

### Completed

- All Python modules passed compilation.
- All three training command-line interfaces passed `--help`.
- Three integration tests passed (`python -m unittest discover -s tests -v`).
- Original CNN weights loaded strictly with no missing or unexpected layers.
- Predictions were repeatable and agreed across different batch sizes.
- A synthetic WFDB record passed preprocessing, feature extraction, CNN prediction, SQLite persistence and Excel export checks.
- Invalid class indices were rejected without inserting another analysis.
- Newly exported training checkpoints and metadata loaded in the application.
- Real MIT-BIH record 100 produced 2271 retained windows, 12 classical features per beat and finite three-class CNN probabilities.
- Real-record CNN segments matched the original notebook loader within a maximum absolute difference of `5.4e-7`; labels were identical.

### Not completed

- Full retraining, grid search or full-dataset metric reproduction.
- Interactive desktop testing of widgets, dialogs and operating-system-specific behavior.
- Clinical validation or calibration of model probabilities.

### Before demonstrating on your computer

1. Install the dependencies and run `python -m app.app_gui` from the repository root.
2. Select the MIT-BIH directory and initialize the bundled model and metadata.
3. Load record `100`, select a small range and analyze it twice.
4. Save the results to SQLite, Excel and PNG; verify that all three use the same range.
5. Change the selected range and test saving the previous analysis — the PNG should retain the previously analyzed range. Also test cancelling the save and closing the application.

### Refactoring notes

The original notebooks, model and historical database were not modified. The distributable application uses a new database schema and location, and fixes the RR input and model evaluation mode. Missing RR values are now imputed separately for each record. See the README for experimental limitations; historical thesis scores are not presented as measured results for the refactored release.
