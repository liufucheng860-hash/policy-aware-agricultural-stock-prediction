import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Config
# =========================
STOCK_CSV = Path("data/raw/yituo/yituo_daily.csv")   #
DATE_COL = "trade_date"                                  #

FEATURE_COLS = [
    "close", "ma5", "ma20", "ma60",
    "volume", "ret_1d",
    "volatility_5d", "volatility_20d",
    "volume_ma5", "volume_ma20",
    "close_ma5_ratio", "close_ma20_ratio", "close_ma60_ratio",
    "price_pos_20d"
]

HORIZONS = [5, 10, 20]
MODELS = ["LSTM", "GRU", "Transformer"]

SEQ_LEN = 60
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
PATIENCE = 5
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_sequences(df, feature_cols, label_col, seq_len):
    X_list, y_list = [], []

    values = df[feature_cols].values
    labels = df[label_col].values

    for i in range(seq_len - 1, len(df)):
        if pd.isna(labels[i]):
            continue

        x = values[i - seq_len + 1:i + 1]
        y = labels[i]

        if np.isnan(x).any():
            continue

        X_list.append(x)
        y_list.append(int(y))

    return np.array(X_list), np.array(y_list)


def chronological_split(X, y):
    n = len(y)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:]
    )


def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            logits = model(X).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().numpy()

            y_true.extend(y.numpy().astype(int))
            y_pred.extend(preds)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


# =========================
# Models
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        out = self.encoder(x)
        last = out[:, -1, :]
        return self.fc(last)


def build_model(model_name, input_dim):
    if model_name == "LSTM":
        return LSTMModel(input_dim)
    elif model_name == "GRU":
        return GRUModel(input_dim)
    elif model_name == "Transformer":
        return TransformerModel(input_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =========================
# Train
# =========================
def train_one_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    model = build_model(model_name, X_train.shape[-1]).to(DEVICE)

    train_loader = DataLoader(
        SeqDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        SeqDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_loader = DataLoader(
        SeqDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1
    best_state = None
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader)
        val_f1 = val_metrics["f1"]

        print(f"[{model_name}] Epoch {epoch:02d} | Val F1 = {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)

    return test_metrics


def main():
    set_seed(SEED)

    df = pd.read_csv(STOCK_CSV, encoding="utf-8-sig")

    print("[INFO] Columns:")
    print(df.columns.tolist())

    if DATE_COL not in df.columns:
        if "date" in df.columns:
            df["trade_date"] = df["date"]
        else:
            raise ValueError("No valid date column found.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    print("[INFO] Using features:")
    print(available_features)

    results = []

    for horizon in HORIZONS:
        label_col = f"label_{horizon}d"

        if label_col not in df.columns:
            raise ValueError(f"Missing label column: {label_col}")

        temp = df.copy()
        temp = temp.dropna(subset=available_features + [label_col]).reset_index(drop=True)

        X, y = make_sequences(
            temp,
            available_features,
            label_col,
            SEQ_LEN
        )

        print(f"\n========== Horizon {horizon}d ==========")
        print("X shape:", X.shape)
        print("y distribution:", dict(zip(*np.unique(y, return_counts=True))))

        X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

        scaler = StandardScaler()

        n_train, seq_len, n_feat = X_train.shape

        X_train_2d = X_train.reshape(-1, n_feat)
        X_val_2d = X_val.reshape(-1, n_feat)
        X_test_2d = X_test.reshape(-1, n_feat)

        scaler.fit(X_train_2d)

        X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
        X_val = scaler.transform(X_val_2d).reshape(X_val.shape)
        X_test = scaler.transform(X_test_2d).reshape(X_test.shape)

        for model_name in MODELS:
            print(f"\n----- Training {model_name} | {horizon}d -----")

            metrics = train_one_model(
                model_name,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test
            )

            print(f"\n[{model_name} {horizon}d Test]")
            print(metrics)

            results.append({
                "model": model_name,
                "horizon": f"{horizon}d",
                **metrics
            })

    result_df = pd.DataFrame(results)

    output_dir = Path("outputs/sequential_baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_name = STOCK_CSV.parent.name
    output_file = output_dir / f"{stock_name}_sequential_baselines.csv"

    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n========== Final Results ==========")
    print(result_df)
    print(f"\n[OK] saved to {output_file}")


if __name__ == "__main__":
    main()
