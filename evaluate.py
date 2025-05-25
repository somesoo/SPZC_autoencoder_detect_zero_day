import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime

# === PARAMETRY ===
INPUT_DIM = 49
THRESHOLD = 0.1  # Możesz zmieniać lub testować różne
MODEL_PATH = "autoencoder_*.pt"  # ustaw konkretny plik lub użyj najnowszego
PLOT_DIR = "./plots"

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

# === Wczytaj dane ===
X_attack = np.load("X_attack.npy")
y_attack = np.load("y_attack.npy", allow_pickle=True)  # oryginalne etykiety (str), np. Web Attack, DDoS, itd.

# === Przygotuj dane i model ===
X_tensor = torch.tensor(X_attack, dtype=torch.float32)
model = Autoencoder()
model_file = sorted([f for f in os.listdir(".") if f.startswith("autoencoder_") and f.endswith(".pt")])[-1]
model.load_state_dict(torch.load(model_file))
model.eval()

# === Przewiduj i oblicz MSE dla każdej próbki ===
with torch.no_grad():
    output = model(X_tensor)
    mse = torch.mean((output - X_tensor) ** 2, dim=1).numpy()

# === Klasyfikacja: czy MSE przekracza próg ===
y_pred = (mse > THRESHOLD).astype(int)  # 1 = atak, 0 = benign
y_true = np.ones_like(y_pred)  # wszystkie próbki to ataki

# === Metryki ===
acc = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel() if len(np.unique(y_pred)) > 1 else (0,0,0,len(y_pred))
fpr = fp / (fp + tn + 1e-8)

# === Raport ===
print(f"Wyniki dla threshold = {THRESHOLD}")
print(f"Accuracy:  {acc:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"FPR:       {fpr:.4f}")
print(f"Model:     {model_file}")

# === Histogram błędów rekonstrukcji ===
os.makedirs(PLOT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
plt.figure(figsize=(8, 5))
plt.hist(mse, bins=100, color='steelblue')
plt.axvline(x=THRESHOLD, color='red', linestyle='--', label=f'Threshold = {THRESHOLD}')
plt.title("Rozkład błędów rekonstrukcji (MSE)")
plt.xlabel("MSE")
plt.ylabel("Liczba próbek")
plt.legend()
plot_path = os.path.join(PLOT_DIR, f"mse_histogram_{timestamp}.png")
plt.savefig(plot_path)
print(f"Histogram zapisany do: {plot_path}")
