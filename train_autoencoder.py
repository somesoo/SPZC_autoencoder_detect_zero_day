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

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_losses, valid_losses = [], []
train_acc,    valid_acc    = [], []

for epoch in range(1, args.epochs + 1):
    model.train()
    epoch_loss, correct = 0.0, 0
    for (batch,) in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)

        # accuracy na FIXED_THRESHOLD = 0.01  (możesz zmienić)
        mse_vals = torch.mean((output - batch) ** 2, dim=1)
        correct += (mse_vals < 0.01).sum().item()

    train_losses.append(epoch_loss / len(X_train_tensor))
    train_acc.append(correct / len(X_train_tensor))


    model.eval()
    with torch.no_grad():
        output_v = model(X_valid_tensor.to(device))
        v_loss = criterion(output_v, X_valid_tensor.to(device)).item()
        v_mse  = torch.mean((output_v - X_valid_tensor.to(device)) ** 2, dim=1)
        v_acc  = (v_mse < 0.01).sum().item() / len(X_valid_tensor)

    valid_losses.append(v_loss)
    valid_acc.append(v_acc)

    print(f"[{epoch}/{args.epochs}] "
          f"TrainLoss={train_losses[-1]:.5f}  ValidLoss={v_loss:.5f}  "
          f"TAcc={train_acc[-1]:.4f}  VAcc={v_acc:.4f}")
    
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
plt.figure(figsize=(6,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Valid Loss")
plt.plot(train_acc,   label="Train Acc")
plt.plot(valid_acc,   label="Valid Acc")
plt.xlabel("Epoch"); plt.grid(True); plt.legend()
plt.title("Autoencoder – Loss & Accuracy")
plt.tight_layout()
plt.savefig(os.path.join("plots", "loss_acc_curve.png"))
plt.close()

print(f"Zapisano model: {model_name}")
print(f"Zapisano metadane i wykres do: {args.models_dir}, {args.plots_dir}")
