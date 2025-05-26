/home/johnny/Documents/studia/spzc/PROJ/preproces_dataset.py:23: DtypeWarning: Columns (0,1,3,6,84) have mixed types. Specify dtype option on import or set low_memory=False.
  df = pd.read_csv(file, encoding='latin1')  # fallback, np. Windows-1252
Wczytano 3119345 rekordów z 8 plików.
Usunięto 31 silnie skorelowanych cech.
Dane gotowe:
- X_train: (1703490, 49)
- X_valid: (567830, 49)
- X_attack: (556556, 49)


## Aktualny stan

- Przetworzono dane CICIDS2017 (feature selection, skalowanie)
- Podzielono dane: 60% benign (trening), 20% benign (walidacja), 20% benign + ataki (test)
- Wytrenowano autoenkoder na benign (architektura: 80 → 32 → 16 → 32 → 80)
- Model zapisano jako `.pt`, metadane `.json`, wykres strat `.png`
- Testowanie oparte na błędzie rekonstrukcji (MSE), progi dobierane ręcznie
- Skrypty: `prepare_balanced_dataset.py`, `train_autoencoder_prepared.py`, `evaluate_prepared.py`

## Do zrobienia

- Dalsze testy dla różnych progów detekcji
- Analiza wykrywalności typów ataków