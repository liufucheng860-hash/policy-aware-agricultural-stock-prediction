from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader


# =========================================================
# =========================================================
IMAGE_DATA_FILE = Path("data/processed/xingguang/xingguang_image_dataset.npz")
IMAGE_META_FILE = Path("data/processed/xingguang/xingguang_image_metadata.csv")
MACRO_FILE = Path("data/raw/macro_policy_yearly.csv")
OUTPUT_DIR = Path("outputs/xingguang")

TARGET = "10d"   #

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MACRO_FEATURE_COLUMNS = [
    "policy_strength",
    "machinery_focus",
    "smart_agri_focus",
    "seed_focus",
    "rural_reform_focus",
    "subsidy_support",
    "technology_focus",
    "macro_total_score",
]


# =========================================================
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# =========================================================
class ImageMacroDataset(Dataset):
    def __init__(self, X_image: np.ndarray, X_macro: np.ndarray, y: np.ndarray):
        """
        X_image: (n_samples, n_features, window_size)
        X_macro: (n_samples, n_macro_features)
        y:       (n_samples,)
        """
        self.X_image = torch.tensor(X_image, dtype=torch.float32)
        self.X_macro = torch.tensor(X_macro, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_img = self.X_image[idx].unsqueeze(0)   # (1, n_features, window_size)
        x_macro = self.X_macro[idx]
        y = self.y[idx]
        return x_img, x_macro, y


# =========================================================
# =========================================================
class ImageMacroModel(nn.Module):
    def __init__(self, n_macro_features: int):
        super().__init__()

        self.image_branch = nn.Sequential(
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

        self.image_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.macro_branch = nn.Sequential(
            nn.Linear(n_macro_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x_img, x_macro):
        img_feat = self.image_branch(x_img)
        img_feat = self.image_head(img_feat)

        macro_feat = self.macro_branch(x_macro)

        fused = torch.cat([img_feat, macro_feat], dim=1)
        logits = self.fusion_head(fused)
        return logits


# =========================================================
# =========================================================
def time_split(X_img, X_macro, y, meta_df, train_ratio=0.7, val_ratio=0.15):
    n = len(y)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_img_train = X_img[:train_end]
    X_macro_train = X_macro[:train_end]
    y_train = y[:train_end]
    meta_train = meta_df.iloc[:train_end].reset_index(drop=True)

    X_img_val = X_img[train_end:val_end]
    X_macro_val = X_macro[train_end:val_end]
    y_val = y[train_end:val_end]
    meta_val = meta_df.iloc[train_end:val_end].reset_index(drop=True)

    X_img_test = X_img[val_end:]
    X_macro_test = X_macro[val_end:]
    y_test = y[val_end:]
    meta_test = meta_df.iloc[val_end:].reset_index(drop=True)

    return (
        X_img_train, X_macro_train, y_train, meta_train,
        X_img_val, X_macro_val, y_val, meta_val,
        X_img_test, X_macro_test, y_test, meta_test
    )


# =========================================================
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for x_img, x_macro, y in loader:
        x_img = x_img.to(device)
        x_macro = x_macro.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x_img, x_macro)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_img.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    all_true = []
    all_pred = []
    all_prob = []

    for x_img, x_macro, y in loader:
        x_img = x_img.to(device)
        x_macro = x_macro.to(device)
        y = y.to(device)

        logits = model(x_img, x_macro)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * x_img.size(0)

        all_true.extend(y.cpu().numpy())
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
# =========================================================
def load_and_merge_data():
    if not IMAGE_DATA_FILE.exists():
        raise FileNotFoundError(f": {IMAGE_DATA_FILE}")
    if not IMAGE_META_FILE.exists():
        raise FileNotFoundError(f": {IMAGE_META_FILE}")
    if not MACRO_FILE.exists():
        raise FileNotFoundError(f": {MACRO_FILE}")

    data = np.load(IMAGE_DATA_FILE, allow_pickle=True)
    meta_df = pd.read_csv(IMAGE_META_FILE)
    macro_df = pd.read_csv(MACRO_FILE)

    X = data["X"]
    y_5d = data["y_5d"]
    y_10d = data["y_10d"]
    y_20d = data["y_20d"]

    meta_df = meta_df.copy()
    meta_df["window_end_date"] = pd.to_datetime(meta_df["window_end_date"])
    meta_df["year"] = meta_df["window_end_date"].dt.year

    macro_df = macro_df.copy()
    missing_cols = [c for c in MACRO_FEATURE_COLUMNS if c not in macro_df.columns]
    if missing_cols:
        raise ValueError(f": {missing_cols}")

    merged = meta_df.merge(
        macro_df[["year"] + MACRO_FEATURE_COLUMNS],
        on="year",
        how="left"
    )

    if merged[MACRO_FEATURE_COLUMNS].isna().any().any():
        bad = merged[merged[MACRO_FEATURE_COLUMNS].isna().any(axis=1)]
        raise ValueError(f"\n{bad[['window_end_date', 'year']].head()}")

    X_macro = merged[MACRO_FEATURE_COLUMNS].values.astype(np.float32)

    if TARGET == "5d":
        y = y_5d
    elif TARGET == "10d":
        y = y_10d
    elif TARGET == "20d":
        y = y_20d
    else:
        raise ValueError("TARGET  '5d'  '20d'")

    return X, X_macro, y, merged


# =========================================================
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_img, X_macro, y, merged_meta = load_and_merge_data()

    print(f"[INFO] X_img shape: {X_img.shape}")
    print(f"[INFO] X_macro shape: {X_macro.shape}")
    print(f"[INFO] Target: {TARGET}")
    print(f"[INFO] Macro features: {MACRO_FEATURE_COLUMNS}")

    (
        X_img_train, X_macro_train, y_train, meta_train,
        X_img_val, X_macro_val, y_val, meta_val,
        X_img_test, X_macro_test, y_test, meta_test
    ) = time_split(X_img, X_macro, y, merged_meta)

    print(f"[INFO] Train size: {len(y_train)}")
    print(f"[INFO] Val size: {len(y_val)}")
    print(f"[INFO] Test size: {len(y_test)}")

    train_dataset = ImageMacroDataset(X_img_train, X_macro_train, y_train)
    val_dataset = ImageMacroDataset(X_img_val, X_macro_val, y_val)
    test_dataset = ImageMacroDataset(X_img_test, X_macro_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ImageMacroModel(n_macro_features=len(MACRO_FEATURE_COLUMNS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

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

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, DEVICE)

    print("\n=== Final Test Metrics ===")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"F1       : {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")

    history_df = pd.DataFrame(history)
    history_file = OUTPUT_DIR / f"metrics_image_macro_{TARGET}_history.csv"
    history_df.to_csv(history_file, index=False, encoding="utf-8-sig")

    test_result_df = meta_test.copy()
    test_result_df["y_true"] = test_metrics["y_true"]
    test_result_df["y_pred"] = test_metrics["y_pred"]
    test_result_df["y_prob"] = test_metrics["y_prob"]

    pred_file = OUTPUT_DIR / f"predictions_image_macro_{TARGET}.csv"
    test_result_df.to_csv(pred_file, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame([{
        "model": "image_macro_fusion",
        "target": TARGET,
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
    }])

    summary_file = OUTPUT_DIR / f"metrics_image_macro_{TARGET}_summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print('\n[OK]')
    print(f"- {history_file}")
    print(f"- {pred_file}")
    print(f"- {summary_file}")


if __name__ == "__main__":
    main()
