from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. configuration
# =========================================================
DATA_FILE = Path("data/processed/xingguang/xingguang_image_dataset.npz")
META_FILE = Path("data/processed/xingguang/xingguang_image_metadata.csv")
OUTPUT_DIR = Path("outputs/xingguang")

TARGET = "10d"

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 2. fix random seed
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# 3. Dataset
# =========================================================
class ImageDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: shape = (n_samples, n_features, window_size)
        y: shape = (n_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # CNN input needs (channels, height, width)
        # current X[idx] = (n_features, window_size)
        # add one channel dimension -> (1, n_features, window_size)
        x = self.X[idx].unsqueeze(0)
        y = self.y[idx]
        return x, y


# =========================================================
# 4. micro CNN
# =========================================================
class SmallCNN(nn.Module):
    def __init__(self, n_features: int, window_size: int, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# 5. time split
# =========================================================
def time_split(X, y, meta_df, train_ratio=0.7, val_ratio=0.15):
    n = len(y)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]
    meta_train = meta_df.iloc[:train_end].reset_index(drop=True)

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    meta_val = meta_df.iloc[train_end:val_end].reset_index(drop=True)

    X_test = X[val_end:]
    y_test = y[val_end:]
    meta_test = meta_df.iloc[val_end:].reset_index(drop=True)

    return X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test


# =========================================================
# 6. training / testing functions
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    all_true = []
    all_pred = []
    all_prob = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * X_batch.size(0)

        all_true.extend(y_batch.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_prob.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_true, all_pred),
        "f1": f1_score(all_true, all_pred, zero_division=0),
        "precision": precision_score(all_true, all_pred, zero_division=0),
        "recall": recall_score(all_true, all_pred, zero_division=0),
        "y_true": np.array(all_true),
        "y_pred": np.array(all_pred),
        "y_prob": np.array(all_prob),
    }

    return metrics


# =========================================================
# 7. main function
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # read data
    # -------------------------
    if not DATA_FILE.exists():
        raise FileNotFoundError(f": {DATA_FILE}")
    if not META_FILE.exists():
        raise FileNotFoundError(f": {META_FILE}")

    data = np.load(DATA_FILE, allow_pickle=True)
    meta_df = pd.read_csv(META_FILE)

    X = data["X"]   # shape: (n_samples, n_features, window_size)
    y_5d = data["y_5d"]
    y_10d = data["y_10d"]
    y_20d = data["y_20d"]
    feature_names = data["feature_names"]

    if TARGET == "5d":
        y = y_5d
    elif TARGET == "10d":
        y = y_10d
    elif TARGET == "20d":
        y = y_20d
    else:
        raise ValueError("TARGET  '5d' '10d' '20d'")

    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] Target: {TARGET}")
    print(f"[INFO] : {feature_names.tolist()}")

    # -------------------------
    # -------------------------
    (
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        X_test, y_test, meta_test
    ) = time_split(X, y, meta_df)

    print(f"[INFO] Train size: {len(y_train)}")
    print(f"[INFO] Val size: {len(y_val)}")
    print(f"[INFO] Test size: {len(y_test)}")

    # -------------------------
    # DataLoader
    # -------------------------
    train_dataset = ImageDataset(X_train, y_train)
    val_dataset = ImageDataset(X_val, y_val)
    test_dataset = ImageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # -------------------------
    n_features = X.shape[1]
    window_size = X.shape[2]

    model = SmallCNN(n_features=n_features, window_size=window_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -------------------------
    # -------------------------
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        })

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

    # -------------------------
    # -------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    # -------------------------
    # -------------------------
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)

    print("\n=== Final Test Metrics ===")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"F1       : {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")

    # -------------------------
    # -------------------------
    history_df = pd.DataFrame(history)
    history_file = OUTPUT_DIR / f"metrics_image_baseline_{TARGET}_history.csv"
    history_df.to_csv(history_file, index=False, encoding="utf-8-sig")

    # -------------------------
    # -------------------------
    test_result_df = meta_test.copy()
    test_result_df["y_true"] = test_metrics["y_true"]
    test_result_df["y_pred"] = test_metrics["y_pred"]
    test_result_df["y_prob"] = test_metrics["y_prob"]

    pred_file = OUTPUT_DIR / f"predictions_image_baseline_{TARGET}.csv"
    test_result_df.to_csv(pred_file, index=False, encoding="utf-8-sig")

    # -------------------------
    # save
    # -------------------------
    summary_df = pd.DataFrame([{
        "model": "image_baseline_cnn",
        "target": TARGET,
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
    }])

    summary_file = OUTPUT_DIR / f"metrics_image_baseline_{TARGET}_summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print(f"- {history_file}")
    print(f"- {pred_file}")
    print(f"- {summary_file}")


if __name__ == "__main__":
    main()
