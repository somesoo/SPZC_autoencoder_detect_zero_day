import os, json, numpy as np, joblib, torch
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_scaler(path="./prepared_data/scaler.pkl"):
    return joblib.load(path)

def load_model(model_path, input_dim, device):
    from models import Autoencoder
    meta_path = model_path.replace(".pt", ".json").replace("models/", "models/meta_")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    model = Autoencoder(
        input_dim=input_dim,
        encoder_dims=meta.get("encoder_dims", [64, 32, 16]),
        decoder_dims=meta.get("decoder_dims", [32, 64]),
        use_batchnorm=meta.get("batchnorm", False),
        dropout=meta.get("dropout", 0.0)
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def mse_tensor(a, b):
    return torch.mean((a - b) ** 2, dim=1)

def compute_metrics(y_true, y_pred):
    tn_count = (y_true == 0).sum()
    fpr = ((y_pred == 1) & (y_true == 0)).sum() / tn_count if tn_count > 0 else None

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "fpr":       fpr
    }

def save_artifacts(out_dir, X, y, params, metrics):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_test.npy"), X)
    np.save(os.path.join(out_dir, "y_test.npy"), y)
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
