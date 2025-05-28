# Zero-Day Attack Detection — Dataset Preparation (Phase 2)

Wersja robocza – przygotowanie danych do wykrywania ataków typu zero-day przy użyciu autoenkodera.

## Cele

- Rozdzielenie danych źródłowych na osobne pliki CSV dla każdego ataku i dnia,
- Połączenie danych benign w jeden zbiór i jego podział (train/valid/test),
- Przygotowanie danych treningowych w formacie `.npy` z odpowiednim skalowaniem.

---

## Struktura katalogów

/data/TrafficLabelling/ ← surowe pliki CSV
/separated/ ← pliki podzielone wg typu ruchu i dnia
/benign_splits/ ← benign podzielone: train / valid / test
/prepared_data/ ← gotowe pliki .npy i scaler


---

## Skrypty

### `split_attacks_by_day.py`
Dzieli każdy plik CSV z `TrafficLabelling` na osobne pliki według typu ruchu (`Label`) i dnia, zapisując do `./separated/`.

### `split_benign_dataset.py`
Łączy wszystkie pliki `BENIGN__*.csv` i dzieli dane na:
- 70% → `benign_train.csv` (trening),
- 10% → `benign_valid.csv` (walidacja),
- 20% → `benign_test.csv` (do testowania z atakami).

### `convert_csv_to_npy.py`
Wczytuje `benign_train.csv` i `benign_valid.csv`, usuwa kolumny nienumeryczne i skaluje dane, zapisując je do plików `.npy` oraz `scaler.pkl`.

---

## Następne kroki

- Połączenie `benign_test.csv` z wybranymi atakami do stworzenia `X_test.npy`, `y_test.npy`
- Trenowanie autoenkodera na `X_train.npy`
- Ocena skuteczności na danych atakowych

