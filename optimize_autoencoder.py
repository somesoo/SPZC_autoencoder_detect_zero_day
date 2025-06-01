import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from models import Autoencoder
from sklearn.model_selection import ParameterGrid

# === Konfiguracja
DATA_DIR = "./prepared_data"
MODELS_DIR = "./models_final"
PLOTS_DIR = "./plots/plots_final"
RESULTS_DIR = "./optimization_results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_TRIALS = 2
EPOCHS = 10
ENCODER_DIMS = [64, 32, 16]
DECODER_DIMS = [32, 64]

# === Hiperparametry do siatki
param_grid = {
    "dropout": [0.0],
    "batch_size": [512, 1024],
    "lr": [0.001],
    "batchnorm": [True],
}

# === Wczytaj dane
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_valid = np.load(os.path.join(DATA_DIR, "X_valid.npy"))
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
input_dim = X_train.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Tworzenie konfiguracji
grid = list(ParameterGrid(param_grid))[:MAX_TRIALS]
results = []

# === Trening każdej kombinacji
for i, params in enumerate(grid, start=1):
    print(f"\n=== Próba {i}/{len(grid)} ===")
    print(f"Parametry: {params}")

    model = Autoencoder(
        input_dim=input_dim,
        encoder_dims=ENCODER_DIMS,
        decoder_dims=DECODER_DIMS,
        use_batchnorm=params["batchnorm"],
        dropout=params["dropout"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    train_losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        avg_loss = epoch_loss / len(X_train_tensor)
        train_losses.append(avg_loss)
        print(f"[{epoch}/{EPOCHS}] Train Loss: {avg_loss:.6f}")

    # Ewaluacja na valid
    model.eval()
    with torch.no_grad():
        output_valid = model(X_valid_tensor.to(device))
        valid_loss = criterion(output_valid, X_valid_tensor.to(device)).item()

    # Zapisz wynik
    result = {
        "trial": i,
        "dropout": params["dropout"],
        "batch_size": params["batch_size"],
        "lr": params["lr"],
        "batchnorm": params["batchnorm"],
        "valid_loss": valid_loss
    }
    results.append(result)

    # Zapisz wykres
    tag = f"trial_{i:02}_drop{params['dropout']}_bs{params['batch_size']}_lr{params['lr']}_bn{params['batchnorm']}"
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.title(f"Loss Curve (trial {i})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{tag}.png"))
    plt.close()

    # Zapisz model jeśli najlepszy
    if i == 1 or valid_loss < min(r["valid_loss"] for r in results[:-1]):
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"best_model_{tag}.pt"))
        print(f"Nowy najlepszy model zapisany jako: best_model_{tag}.pt")

# Zapisz metadane do CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
print("\nZakończono optymalizację. Wyniki zapisane w summary.csv.")
