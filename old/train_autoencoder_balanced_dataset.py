import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime
import json
from models import Autoencoder

# === Parametry
DATA_DIR = "./prepared_data"
PLOT_DIR = "./plots"
EPOCHS = 50
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3

# === Dane
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_valid = np.load(os.path.join(DATA_DIR, "X_valid.npy"))

input_dim = X_train.shape[1]
print(f"input_dim = {input_dim}")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(TensorDataset(X_valid_tensor), batch_size=BATCH_SIZE, shuffle=False)

model = Autoencoder(input_dim=input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Trenowanie
train_losses = []
valid_losses = []

os.makedirs(PLOT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x in train_loader:
        output = model(x[0])
        loss = criterion(output, x[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(x[0]), x[0]).item() for x in valid_loader) / len(valid_loader)
        valid_losses.append(val_loss)

    print(f"[{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Valid Loss: {val_loss:.6f}")

# === Zapis modelu
model_name = f"autoencoder_prepared_{timestamp}.pt"
torch.save(model.state_dict(), model_name)

# === Zapis metadanych
metadata = {
    "timestamp": timestamp,
    "input_dim": input_dim,
    "epochs": EPOCHS,
    "train_samples": len(X_train),
    "valid_samples": len(X_valid),
    "final_train_loss": float(train_losses[-1]),
    "final_valid_loss": float(valid_losses[-1]),
    "model_file": model_name
}
with open(f"autoencoder_prepared_{timestamp}.json", "w") as f:
    json.dump(metadata, f, indent=4)

# === Wykres strat
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Autoencoder Training Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, f"loss_plot_{timestamp}.png"))

print(f"\nModel zapisany jako: {model_name}")
