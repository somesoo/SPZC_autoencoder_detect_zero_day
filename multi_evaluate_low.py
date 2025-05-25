import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

# === PARAMETRY ===
INPUT_DIM = 49
NUM_THRESHOLDS = 20

# === Autoencoder ===
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, INPUT_DIM)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# === Wczytanie danych i modelu ===
X_attack = np.load("X_attack.npy")
y_attack = np.load("y_attack.npy", allow_pickle=True)
X_tensor = torch.tensor(X_attack, dtype=torch.float32)

model = Autoencoder()
model_file = sorted([f for f in os.listdir(".") if f.startswith("autoencoder_") and f.endswith(".pt")])[-1]
model.load_state_dict(torch.load(model_file))
model.eval()

# === Obliczenie błędów rekonstrukcji (MSE)
with torch.no_grad():
    output = model(X_tensor)
    mse = torch.mean((output - X_tensor) ** 2, dim=1).numpy()

# === Testowanie niższych progów
thresholds = np.linspace(0.001, 0.02, NUM_THRESHOLDS)
results = []

for threshold in thresholds:
    y_pred = (mse > threshold).astype(int)
    y_true = np.ones_like(y_pred)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel() if len(np.unique(y_pred)) > 1 else (0,0,0,len(y_pred))
    fpr = fp / (fp + tn + 1e-8)

    results.append({
        "Threshold": round(threshold, 4),
        "Accuracy": round(acc, 4),
        "Recall": round(recall, 4),
        "Precision": round(precision, 4),
        "F1 Score": round(f1, 4),
        "FPR": round(fpr, 4)
    })

# === Tabela wyników
df_results = pd.DataFrame(results)
print("\nWYNIKI DLA NISKICH THRESHOLDÓW:")
print(df_results.to_string(index=False))
