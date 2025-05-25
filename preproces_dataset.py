import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Parametry ===
DATA_DIR = "/home/johnny/Documents/studia/spzc/PROJ/data/TrafficLabelling"  
LABEL_COLUMN = " Label"  
CORR_THRESHOLD = 0.9
TEST_SIZE = 0.25
RANDOM_STATE = 42

# === Łączenie wszystkich CSV ===
all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
df_list = []
for file in all_files:
    try:
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='latin1')  # fallback, np. Windows-1252
        df_list.append(df)
    except Exception as e:
        print(f"Błąd przy ładowaniu {file}: {e}")

df = pd.concat(df_list, ignore_index=True)
print(f"Wczytano {len(df)} rekordów z {len(all_files)} plików.")


# === Czyszczenie etykiet ===
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)
df[LABEL_COLUMN] = df[LABEL_COLUMN].str.encode('ascii', 'ignore').str.decode('ascii')

# === Usunięcie kolumn tekstowych i pomocniczych ===
non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
non_numeric_cols = [col for col in non_numeric_cols if col != LABEL_COLUMN]
df = df.drop(columns=non_numeric_cols)

# === Usunięcie wierszy z brakami i infami ===
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# === Usunięcie silnie skorelowanych cech ===
corr_matrix = df.drop(columns=[LABEL_COLUMN]).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
df = df.drop(columns=to_drop)
print(f"Usunięto {len(to_drop)} silnie skorelowanych cech.")

# === Podział na benign / ataki ===
benign_df = df[df[LABEL_COLUMN] == 'BENIGN'].drop(columns=[LABEL_COLUMN])
attack_df = df[df[LABEL_COLUMN] != 'BENIGN']
X_attack = attack_df.drop(columns=[LABEL_COLUMN])
y_attack = attack_df[LABEL_COLUMN].values

# === Skalowanie i podział danych benign ===
X_train, X_valid = train_test_split(benign_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_attack = scaler.transform(X_attack)

# === Zapis lub gotowe dane do użytku ===
print(f"Dane gotowe:")
print(f"- X_train: {X_train.shape}")
print(f"- X_valid: {X_valid.shape}")
print(f"- X_attack: {X_attack.shape}")

# np. zapisz jako pliki .npy (opcjonalnie)
np.save("X_train.npy", X_train)
np.save("X_valid.npy", X_valid)
np.save("X_attack.npy", X_attack)
np.save("y_attack.npy", y_attack)

