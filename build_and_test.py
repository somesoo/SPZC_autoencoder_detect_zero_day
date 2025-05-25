import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# === PARAMETRY ===
INPUT_DIM = 49
THRESHOLD = 0.015  # możesz zmienić

# === Model Autoenkodera ===
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
X_valid = np.load("X_valid.npy")          # dane benign
X_attack = np.load("X_attack.npy")        # dane ataków
y_attack = np.load("y_attack.npy", allow_pickle=True)

# === Zbuduj X_test i y_test
X_test = np.vstack([X_valid, X_attack])
y_test = np.concatenate([np.zeros(len(X_valid)), np.ones(len(X_attack))])
X_tensor = torch.tensor(X_test, dtype=torch.float32)

# === Wczytaj model
model = Autoencoder()
model_file = sorted([f for f in os.listdir(".") if f.startswith("autoencoder_") and f.endswith(".pt")])[-1]
model.load_state_dict(torch.load(model_file))
model.eval()

# === Oblicz błędy rekonstrukcji (MSE)
with torch.no_grad():
    output = model(X_tensor)
    mse = torch.mean((output - X_tensor) ** 2, dim=1).numpy()

# === Klasyfikacja na podstawie threshold
y_pred = (mse > THRESHOLD).astype(int)

# === Metryki
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn + 1e-8)

# === Wyniki
print("\nWYNIKI (X_valid + X_attack)")
print(f"Threshold:    {THRESHOLD}")
print(f"Accuracy:     {acc:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"Precision:    {precision:.4f}")
print(f"F1 Score:     {f1:.4f}")
print(f"FPR:          {fpr:.4f}")
print(f"True Positives:  {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives:  {tn}")
