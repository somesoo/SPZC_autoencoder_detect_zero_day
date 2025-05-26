import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from models import Autoencoder

# === Parametry ===
INPUT_DIM = 49
BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 1e-3
MODEL_DIR = "."
PLOT_DIR = "./plots"

# === Tworzenie katalogu na wykresy (jeśli nie istnieje) ===
os.makedirs(PLOT_DIR, exist_ok=True)

# === Załaduj dane ===
X_train = np.load("X_train.npy")
X_valid = np.load("X_valid.npy")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(TensorDataset(X_valid_tensor), batch_size=BATCH_SIZE, shuffle=False)

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Trening ===
train_losses = []
valid_losses = []

for epoch in range(EPOCHS):
    start_time = time.time()

    model.train()
    epoch_loss = 0
    for batch in train_loader:
        x = batch[0]
        output = model(x)
        loss = criterion(output, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(x), x).item() for (x,) in valid_loader) / len(valid_loader)
        valid_losses.append(val_loss)

    duration = time.time() - start_time
    print(f"[{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Valid Loss: {val_loss:.6f} | Time: {duration:.2f} sec")


# === Zapis modelu i wykresu ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_path = os.path.join(MODEL_DIR, f"autoencoder_{timestamp}.pt")
torch.save(model.state_dict(), model_path)
print(f"Model zapisany jako: {model_path}")

# === Wykres strat ===
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Autoencoder Training Loss")
plot_path = os.path.join(PLOT_DIR, f"loss_plot_{timestamp}.png")
plt.savefig(plot_path)
print(f"Wykres zapisany do: {plot_path}")


# === Zapis metadanych ===
metadata = {
    "timestamp": timestamp,
    "epochs": EPOCHS,
    "input_dim": INPUT_DIM,
    "train_samples": len(X_train),
    "valid_samples": len(X_valid),
    "final_train_loss": float(train_losses[-1]),
    "final_valid_loss": float(valid_losses[-1]),
    "model_file": os.path.basename(model_path)
}

json_path = os.path.join(MODEL_DIR, f"autoencoder_{timestamp}.json")
with open(json_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Metadane zapisane do: {json_path}")