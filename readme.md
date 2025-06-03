# Projekt: Wykrywanie ataków zero-day z wykorzystaniem autoenkodera

## Wymagania

* Python 3.9+
* PyTorch
* scikit-learn
* pandas
* matplotlib
* seaborn

## Dataset:

https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune/data?select=ARP%20MitM

## Struktura katalogów

```
PROJ/
├── data/TrafficLabelling/         # surowe pliki CSV - do ściągnięcia offline
├── benign_splits/                # benign_train.csv, benign_valid.csv, benign_test.csv
├── attack_splits/                # atak MitM podzielony na różnej wielkości zbiory
├── separated_kitsune/            # podzielone dane na benign i attack
├── prepared_data/                # dane przerobione benign do .npy dla trenowania i testów
├── models/                       # zapisane modele .pt i metadane .json
├── plots/                        # wykresy strat, accuracy, wyniki testów
├── tests/                        # foldery z wynikami każdego testu + danymi .npy na których był wykonywany test
```

## 1. Przygotowanie danych

Podziel dane na osobne pliki CSV:

```bash
python3 split_attacks_and_benign.py
```

Następnie połącz dane benign i podziel je na:

```bash
python3 split_benign_datasets.py
```

(Train/Valid/Test = 70%/10%/20%)

```bash
python3 split_attack_datasets.py
```

(attack w proporcjach = 50%/35%%/15%)

## 2. Konwersja danych do .npy

```bash
python3 convert_csv_to_npy.py
```

Z plików CSV generuje X\_train.npy, X\_valid.npy (bez etykiet).

## 3. Optymalizacja hiperparametrów

```bash
python3 optimize_autoencoder.py --max-trials 20
```

Wykonuje 20 losowych treningów z różnymi parametrami i zapisuje najlepszy model + metadane.


## 4. Trening autoenkodera

```bash
python3 train_autoencoder.py --epochs 50 --batch-size 512 --lr 0.001 \
  --encoder 64 32 16 --decoder 32 64 --use-batchnorm --dropout 0.0
```

Model zapisuje się automatycznie do `models/`, razem z metadanymi i wykresem strat + accuracy.


## 5. Testowanie modelu

Przykład:

```bash
python3 run_test.py \
  --models-dir models \
  --data-dir separated_kitsune \
  --benign-test-csv benign_splits/benign_test.csv \
  --thresholds 0.01 0.02 0.05 0.1
```

### Artefakty:

* `metrics_table.csv`, `per_class_recall.csv`
* wykresy: `plot_recall_f1_fpr.png`, `plot_per_class_recall.png`

## 6. Generowanie wykresów z wyników testów

```bash
python3 plots_from_results.py --test-dir tests/<folder> 
```

Automatycznie wybiera najlepszy threshold i rysuje 3 wykresy.

## 7. Benchmarki

### Benchmark: pojedynczy atak vs model

```bash
python3 benchmark_single.py \
  --models-dir models \
  --data-dir separated_kitsune \
  --benign-test-csv benign_splits/benign_test.csv \
  --thresholds 0.01 0.02 0.05 0.1
```

Zapisuje `benchmark_single_results.csv`

### Benchmark: pary/kombinacje ataków

```bash
python3 benchmark_multiple.py \
  --models-dir models \
  --data-dir separated_kitsune \
  --benign-test-csv benign_splits/benign_test.csv \
  --thresholds 0.01 0.02 0.03 0.04 \
```

Zapisuje `benchmark_pairwise_results.csv`

## Dodatkowe pliki:

```bash
python3 plot_learning.py
```

Tworzy wykres uczenia się podczas optymalizacji na podstawie logu z terminala