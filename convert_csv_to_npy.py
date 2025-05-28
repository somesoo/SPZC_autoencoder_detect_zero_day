import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

INPUT_DIR = "./benign_splits"
OUTPUT_DIR = "./prepared_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_clean_csv(path):
    df = pd.read_csv(path, encoding='latin1', low_memory=False)
    df.columns = df.columns.str.strip()

    # Usuń kolumnę Label jeśli istnieje
    df = df.loc[:, ~df.columns.str.lower().isin(["label"])]

    # Zostaw tylko kolumny numeryczne
    df = df.select_dtypes(include=["number"])

    # Usuń inf, -inf, NaN → zamień na 0.0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)

    return df

# Wczytaj i oczyść dane
df_train = load_and_clean_csv(os.path.join(INPUT_DIR, "benign_train.csv"))
df_valid = load_and_clean_csv(os.path.join(INPUT_DIR, "benign_valid.csv"))

# Zapisz listę użytych kolumn
used_columns = df_train.columns.tolist()
with open(os.path.join(OUTPUT_DIR, "used_columns.txt"), "w") as f:
    for col in used_columns:
        f.write(col + "\n")

# Skalowanie
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train)
X_valid = scaler.transform(df_valid)

# Zapis .npy
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_valid.npy"), X_valid)

# Zapis skalera
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

print("Dane zapisane do ./prepared_data/:")
print(f" - X_train.npy: {X_train.shape}")
print(f" - X_valid.npy: {X_valid.shape}")
print(f" - scaler.pkl")
print(f" - used_columns.txt ({len(used_columns)} cech)")
