import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# === Ścieżki ===
DATA_DIR = "./data/TrafficLabelling"
OUTPUT_DIR = "./prepared_data"
LABEL_COL = " Label"  # uwzględniamy spację

# === Przygotowanie katalogu docelowego ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Wczytanie i połączenie wszystkich CSV ===
all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
df_list = []

for file in all_files:
    try:
        df = pd.read_csv(file, encoding='latin1')
        df_list.append(df)
    except Exception as e:
        print(f"Błąd przy ładowaniu {file}: {e}")

df = pd.concat(df_list, ignore_index=True)
df.columns = df.columns.str.strip()  # usuń spacje z nazw kolumn
LABEL_COL = LABEL_COL.strip()        # też usuń spację ze zmiennej

# === Usunięcie kolumn tekstowych (np. IP, protokoły) ===
non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
non_numeric_cols = [col for col in non_numeric_cols if col != LABEL_COL]
df = df.drop(columns=non_numeric_cols)

# === Usunięcie braków i nieskończoności ===
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# === Podział na benign / attack ===
benign_df = df[df[LABEL_COL] == "BENIGN"].drop(columns=[LABEL_COL])
attack_df = df[df[LABEL_COL] != "BENIGN"]
X_attack = attack_df.drop(columns=[LABEL_COL]).values
y_attack = np.ones(X_attack.shape[0])  # 1 = attack

# === Podział benign: 60% train, 20% valid, 20% test
X_benign = benign_df.values
X_temp, X_test_benign = train_test_split(X_benign, test_size=0.2, random_state=42, shuffle=True)
X_train, X_valid = train_test_split(X_temp, test_size=0.25, random_state=42, shuffle=True)  # 0.25 * 0.8 = 0.2

y_test_benign = np.zeros(X_test_benign.shape[0])  # 0 = benign

# === Skaler tylko na train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_benign_scaled = scaler.transform(X_test_benign)
X_attack_scaled = scaler.transform(X_attack)

# === Tworzenie zbioru testowego
X_test = np.vstack([X_test_benign_scaled, X_attack_scaled])
y_test = np.concatenate([y_test_benign, y_attack])

# === Zapis .npy i skalera ===
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_scaled)
np.save(os.path.join(OUTPUT_DIR, "X_valid.npy"), X_valid_scaled)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

print("Gotowe! Dane zapisane w ./prepared_data/")
print(f"→ X_train: {X_train_scaled.shape}")
print(f"→ X_valid: {X_valid_scaled.shape}")
print(f"→ X_test:  {X_test.shape}")
