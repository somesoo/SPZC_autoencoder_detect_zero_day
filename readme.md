/home/johnny/Documents/studia/spzc/PROJ/preproces_dataset.py:23: DtypeWarning: Columns (0,1,3,6,84) have mixed types. Specify dtype option on import or set low_memory=False.
  df = pd.read_csv(file, encoding='latin1')  # fallback, np. Windows-1252
Wczytano 3119345 rekordów z 8 plików.
Usunięto 31 silnie skorelowanych cech.
Dane gotowe:
- X_train: (1703490, 49)
- X_valid: (567830, 49)
- X_attack: (556556, 49)