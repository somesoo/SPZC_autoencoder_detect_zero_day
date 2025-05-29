# Projekt: Wykrywanie ataków zero-day z wykorzystaniem autoenkodera

## Wymagania

* Python 3.9+
* PyTorch
* scikit-learn
* pandas
* matplotlib
* seaborn

## Struktura katalogów

```
PROJ/
├── data/TrafficLabelling/         # surowe pliki CSV - do ściągnięcia offline
├── separated/                     # dane podzielone na osobne pliki CSV (jeden atak per plik)
├── benign_splits/                # benign_train.csv, benign_valid.csv, benign_test.csv
├── prepared_data/                # dane przerobione do .npy dla trenowania i testów
├── models/                       # zapisane modele .pt i metadane .json
├── plots/                        # wykresy strat, accuracy, wyniki testów
├── tests/                        # foldery z wynikami każdego testu
```

## 1. Przygotowanie danych

Podziel dane na osobne pliki CSV:

```bash
python3 split_attacks_and_benign.py
```

Następnie połącz dane benign i podziel je na:

```bash
python3 split_benign_sets.py
```

(Train/Valid/Test = 70%/10%/20%)

## 2. Konwersja danych do .npy

```bash
python3 convert_csv_to_npy.py
```

Z plików CSV generuje X\_train.npy, X\_valid.npy (bez etykiet).

## 3. Trening autoenkodera

```bash
python3 train_autoencoder.py --epochs 50 --batch-size 512 --lr 0.001 \
  --encoder 64 32 16 --decoder 32 64 --use-batchnorm --dropout 0.0
```

Model zapisuje się automatycznie do `models/`, razem z metadanymi i wykresem strat + accuracy.

## 4. Optymalizacja hiperparametrów

```bash
python3 grid_search_autoencoder.py --max-trials 20
```

Wykonuje 20 losowych treningów z różnymi parametrami i zapisuje najlepszy model + metadane.

## 5. Testowanie modelu

### Tryby testu:

* `--test-type benign` – test tylko na danych benign (walidacja FPR)
* `--test-type mix` – 50/50 benign i ataki (globalne metryki)
* `--test-type single` – test na konkretnym ataku (np. `--attack ddos`)

Przykład:

```bash
python3 run_test.py \
  --model models/autoencoder_20250529_1621.pt \
  --test-type single --attack ddos \
  --thresholds 0.01 0.02 0.03 0.04
```

### Artefakty:

* `metrics_table.csv`, `per_class_recall.csv`
* wykresy: `plot_recall_f1_fpr.png`, `plot_per_class_recall.png`

## 6. Generowanie wykresów z wyników testów

```bash
python3 plots_from_results.py --test-dir tests/<folder> --metric recall
```

Automatycznie wybiera najlepszy threshold i rysuje 3 wykresy.

## 7. Benchmarki

### Benchmark: pojedynczy atak vs model

```bash
python3 benchmark_single.py \
  --models-dir models \
  --data-dir separated \
  --benign-test-csv benign_splits/benign_test.csv \
  --thresholds 0.01 0.02 0.03 0.04
```

Zapisuje `benchmark_single_results.csv`

### Benchmark: pary/kombinacje ataków

```bash
python3 benchmark_pairwise.py \
  --models-dir models \
  --data-dir separated \
  --benign-test-csv benign_splits/benign_test.csv \
  --thresholds 0.01 0.02 0.03 0.04 \
  -n 4 -c 3
```

Zapisuje `benchmark_pairwise_results.csv`

## 8. Wizualizacja benchmarków


### Wykres słupkowy:

```bash
python3 visualize_pairwise.py --metric recall
```

---

## Dodatkowe narzędzia:

* `generate_attack_list.py` – zapisuje wszystkie dostępne ataki do `attack_list.txt`
