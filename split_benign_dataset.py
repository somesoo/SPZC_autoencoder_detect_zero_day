import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_DIR = "./separated_kitsune"
OUTPUT_DIR = "./benign_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Wczytaj wszystkie pliki BENIGN__*.csv
benign_files = glob.glob(os.path.join(SOURCE_DIR, "attack*.csv"))

all_benign = []

for file_path in sorted(benign_files):
    try:
        df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
        all_benign.append(df)
        print(f"Załadowano {os.path.basename(file_path)} ({len(df)} rekordów)")
    except Exception as e:
        print(f"Błąd w {file_path}: {e}")

# Połączenie w jedną całość
df_benign = pd.concat(all_benign, ignore_index=True)
print(f"\nŁączna liczba rekordów benign: {len(df_benign)}")

# Tasowanie i dzielenie
df_temp, df_test = train_test_split(df_benign, test_size=0.20, random_state=42, shuffle=True)
df_train, df_valid = train_test_split(df_temp, test_size=0.125, random_state=42, shuffle=True)  # 0.125 * 0.8 = 0.10

# Zapis do CSV
df_train.to_csv(os.path.join(OUTPUT_DIR, "benign_train.csv"), index=False)
df_valid.to_csv(os.path.join(OUTPUT_DIR, "benign_valid.csv"), index=False)
df_test.to_csv(os.path.join(OUTPUT_DIR, "benign_test.csv"), index=False)

print("\nZbiory zapisane w katalogu ./benign_splits/:")
print(f" - benign_train.csv: {len(df_train)}")
print(f" - benign_valid.csv: {len(df_valid)}")
print(f" - benign_test.csv:  {len(df_test)}")
