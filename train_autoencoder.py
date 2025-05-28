import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import json
from models import Autoencoder

# === Argumenty CLI ===
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="./prepared_data")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--encoder", nargs="+", type=int, default=[64, 32, 16])
parser.add_argument("--decoder", nargs="+", type=int, default=[32, 64])
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--use-batchnorm", action="store_true")
parser.add_argument("--plots-dir", type=str, default="./plots")
parser.add_argument("--models-dir", type=str, default="./models")
args = parser.parse_args()

os.makedirs(args.plots_dir, exist_ok=True)
os.makedirs(args.models_dir, exist_ok=True)

# === Wczytaj dane
X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
X_valid = np.load(os.path.join(args.data_dir, "X_valid.npy"))

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)

input_dim = X_train.shape[1]

# === Model
model = Autoencoder(
    input_dim=input_dim,
    encoder_dims=args.encoder,
    decoder_dims=args.decoder,
    use_batchnorm=args.use_batchnorm,
    dropout=args.dropout
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Trening
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_losses = []
valid_losses = []

train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

print(f"Trenowanie modelu: input_dim={input_dim}, encoder={args.encoder}, decoder={args.decoder}")
start_all = datetime.now()

for epoch in range(1, args.epochs + 1):
    model.train()
    epoch_loss = 0
    start_epoch = datetime.now()

    for (batch,) in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)

    avg_train_loss = epoch_loss / len(X_train_tensor)
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        output_valid = model(X_valid_tensor.to(device))
        valid_loss = criterion(output_valid, X_valid_tensor.to(device)).item()
        valid_losses.append(valid_loss)

    end_epoch = datetime.now()
    duration = (end_epoch - start_epoch).total_seconds()
    print(f"[{epoch}/{args.epochs}] Train Loss: {avg_train_loss:.6f} | Valid Loss: {valid_loss:.6f} | Time: {duration:.2f}s")

# === Zapis modelu i metadanych
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_name = f"autoencoder_{timestamp}.pt"
torch.save(model.state_dict(), os.path.join(args.models_dir, model_name))

metadata = {
    "timestamp": timestamp,
    "input_dim": input_dim,
    "encoder_dims": args.encoder,
    "decoder_dims": args.decoder,
    "batchnorm": args.use_batchnorm,
    "dropout": args.dropout,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "lr": args.lr,
    "final_train_loss": train_losses[-1],
    "final_valid_loss": valid_losses[-1]
}
with open(os.path.join(args.models_dir, f"meta_{timestamp}.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# === Wykres
plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Training")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(args.plots_dir, f"loss_{timestamp}.png"))

print(f"Zapisano model: {model_name}")
print(f"Zapisano metadane i wykres do: {args.models_dir}, {args.plots_dir}")
