import pandas as pd
import numpy as np

FILE = "data/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"

df = pd.read_csv(FILE, encoding='latin1', dtype=object, low_memory=False)
print(f"✅ Załadowano: {FILE}")
print("🔎 Sprawdzanie kolumn pod kątem mieszanych typów...\n")

for col in df.columns:
    # zlicz typy w obrębie jednej kolumny
    types = df[col].map(lambda x: type(x)).value_counts()

    if len(types) > 1:
        print(f"🟡 Kolumna '{col}' ma mieszane typy danych:")
        print(types)
        
        # pokaż przykładowe nietypowe wartości
        dominant_type = types.idxmax()
        problematic = df[~df[col].map(lambda x: isinstance(x, dominant_type))]
        print("  Przykładowe nietypowe wartości:")
        print(problematic[col].dropna().unique()[:5])
        print("-" * 40)
