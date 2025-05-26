import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from models import Autoencoder

# === Parametry ===
THRESHOLD = 0.01
DATA_DIR = "./prepared_data"

# === Wczytaj dane i model ===
X_valid = np.load(os.path.join(DATA_DIR, "X_valid.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# === Model
model_file = sorted([f for f in os.listdir(".") if f.startswith("autoencoder_prepared") and f.endswith(".pt")])[-1]
model = Autoencoder(input_dim=X_valid.shape[1])
model.load_state_dict(torch.load(model_file))
model.eval()

# === Oblicz MSE dla walidacji (benign)
with torch.no_grad():
    output_valid = model(X_valid_tensor)
    mse_valid = torch.mean((output_valid - X_valid_tensor) ** 2, dim=1).numpy()

fp_valid = np.sum(mse_valid > THRESHOLD)
total_valid = len(mse_valid)

print(f"\nWalidacja (tylko benign)")
print(f"Threshold:    {THRESHOLD}")
print(f"FP (False Positives): {fp_valid} / {total_valid} → {(fp_valid / total_valid) * 100:.2f}%")

# === Oblicz MSE i klasyfikację dla testu
with torch.no_grad():
    output_test = model(X_test_tensor)
    mse_test = torch.mean((output_test - X_test_tensor) ** 2, dim=1).numpy()

y_pred = (mse_test > THRESHOLD).astype(int)

acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn + 1e-8)

print(f"\nTest (benign + ataki)")
print(f"Accuracy:     {acc:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"Precision:    {precision:.4f}")
print(f"F1 Score:     {f1:.4f}")
print(f"FPR:          {fpr:.4f}")
print(f"True Positives:  {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives:  {tn}")
