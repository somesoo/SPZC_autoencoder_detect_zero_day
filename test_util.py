import os, json, numpy as np, joblib, torch
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_scaler(path="./prepared_data/scaler_train_valid.pkl"):
    return joblib.load(path)

def load_model(model_path, input_dim, device):
    from models import Autoencoder
    checkpoint = torch.load(model_path, map_location=device)
    # Architektura musi odpowiadać checkpointowi – zmień, jeśli masz inną
    model = Autoencoder(input_dim=input_dim,
                        encoder_dims=[64, 32, 16],
                        decoder_dims=[32, 64],
                        use_batchnorm=True,
                        dropout=0.1).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def mse_tensor(a, b):
    return torch.mean((a - b) ** 2, dim=1)

def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "fpr":       ((y_pred==1) & (y_true==0)).sum() / (y_true==0).sum()
    }

def save_artifacts(out_dir, X, y, params, metrics):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_test.npy"), X)
    np.save(os.path.join(out_dir, "y_test.npy"), y)
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
