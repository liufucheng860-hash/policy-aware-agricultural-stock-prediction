from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# =========================================================
# =========================================================
IMAGE_DATA_FILE = Path("data/processed/longping/longping_image_dataset.npz")
IMAGE_META_FILE = Path("data/processed/longping/longping_image_metadata.csv")
MACRO_FILE = Path("data/raw/macro_policy_yearly.csv")
MICRO_FILE = Path("data/processed/longping/longping_micro_features.csv")
OUTPUT_DIR = Path("outputs/longping")

TARGET = "20d"              # "5d" or "20d"
USE_SUBSAMPLE_2021 = True

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

MACRO_DROPOUT_PROB = 0.25
MICRO_DROPOUT_PROB = 0.25

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

MICRO_FEATURE_COLUMNS = [
    "days_since_last_any",
    "days_since_last_pos",
    "days_since_last_neg",
    "has_event_today",
    "event_count_today",
    "all_count_90d",
    "pos_count_90d",
    "neg_count_90d",
    "net_sent_90d",
    "equity_pledge_90d",
    "regulatory_inquiry_90d",
    "shareholder_reduction_90d",
    "strategic_cooperation_90d",
    "other_90d",
    "all_count_180d",
    "pos_count_180d",
    "neg_count_180d",
    "net_sent_180d",
    "equity_pledge_180d",
    "regulatory_inquiry_180d",
    "shareholder_reduction_180d",
    "strategic_cooperation_180d",
    "other_180d",
]


# =========================================================
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageMacroMicroDataset(Dataset):
    def __init__(self, X_image, X_macro, X_micro, y):
        self.X_image = torch.tensor(X_image, dtype=torch.float32)
        self.X_macro = torch.tensor(X_macro, dtype=torch.float32)
        self.X_micro = torch.tensor(X_micro, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_img = self.X_image[idx].unsqueeze(0)
        x_macro = self.X_macro[idx]
        x_micro = self.X_micro[idx]
        y = self.y[idx]
        return x_img, x_macro, x_micro, y


# =========================================================
# =========================================================
class ImageMacroMicroGatedModel(nn.Module):
    def __init__(self, n_macro_features, n_micro_features):
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

        self.micro_branch = nn.Sequential(
            nn.Linear(n_micro_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.fusion_macro = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fusion_micro = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2),
        )

    def forward(self, x_img, x_macro, x_micro):
        h_img = self.image_head(self.image_branch(x_img))
        h_macro = self.macro_branch(x_macro)
        h_micro = self.micro_branch(x_micro)

        z_macro = self.fusion_macro(torch.cat([h_img, h_macro], dim=1))
        z_micro = self.fusion_micro(torch.cat([h_img, h_micro], dim=1))

        alpha = torch.sigmoid(self.gate_mlp(torch.cat([z_macro, z_micro], dim=1)))
        z_final = alpha * z_macro + (1.0 - alpha) * z_micro

        logits = self.classifier(z_final)
        return logits, alpha


# =========================================================
# =========================================================
def apply_modality_dropout(x_macro, x_micro, macro_p, micro_p):
    batch_size = x_macro.size(0)
    device = x_macro.device

    macro_keep = (torch.rand(batch_size, 1, device=device) > macro_p).float()
    micro_keep = (torch.rand(batch_size, 1, device=device) > micro_p).float()

    x_macro = x_macro * macro_keep
    x_micro = x_micro * micro_keep

    return x_macro, x_micro


def apply_eval_mask(x_macro, x_micro, mode):
    if mode == "full":
        return x_macro, x_micro
    if mode == "drop_macro":
        return torch.zeros_like(x_macro), x_micro
    if mode == "drop_micro":
        return x_macro, torch.zeros_like(x_micro)
    if mode == "drop_both":
        return torch.zeros_like(x_macro), torch.zeros_like(x_micro)
    raise ValueError(f"Unknown eval mode: {mode}")


# =========================================================
# =========================================================
def load_and_align_data():
    image_data = np.load(IMAGE_DATA_FILE, allow_pickle=True)
    image_meta = pd.read_csv(IMAGE_META_FILE, encoding="utf-8-sig")
    micro_df = pd.read_csv(MICRO_FILE, encoding="utf-8-sig")
    macro_df = pd.read_csv(MACRO_FILE, encoding="utf-8-sig")

    X_img = image_data["X"]
    y_5d = image_data["y_5d"]
    y_20d = image_data["y_20d"]

    image_meta["window_end_date"] = pd.to_datetime(image_meta["window_end_date"])
    micro_df["window_end_date"] = pd.to_datetime(micro_df["window_end_date"])

    image_meta = image_meta.reset_index(drop=True)
    micro_df = micro_df.reset_index(drop=True)

    image_meta["row_id"] = image_meta.index
    micro_df["row_id"] = micro_df.index

    merged = image_meta.merge(
        micro_df,
        on=["row_id", "window_end_date"],
        how="inner",
        suffixes=("_img", "_micro"),
    )

    merged["year"] = merged["window_end_date"].dt.year

    missing_macro_cols = [c for c in MACRO_FEATURE_COLUMNS if c not in macro_df.columns]
    if missing_macro_cols:
        raise ValueError(f": {missing_macro_cols}")

    merged = merged.merge(
        macro_df[["year"] + MACRO_FEATURE_COLUMNS],
        on="year",
        how="left",
    )

    if merged[MACRO_FEATURE_COLUMNS].isna().any().any():
        bad = merged[merged[MACRO_FEATURE_COLUMNS].isna().any(axis=1)]
        raise ValueError(f":\n{bad[['window_end_date', 'year']].head()}")

    missing_micro_cols = [c for c in MICRO_FEATURE_COLUMNS if c not in merged.columns]
    if missing_micro_cols:
        raise ValueError(f": {missing_micro_cols}")

    merged = merged.sort_values("row_id").reset_index(drop=True)
    row_indices = merged["row_id"].astype(int).values

    X_img_aligned = X_img[row_indices]

    if TARGET == "5d":
        y_all = y_5d
    elif TARGET == "20d":
        y_all = y_20d
    else:
        raise ValueError("TARGET  '5d'  '20d'")

    y_aligned = y_all[row_indices]

    if USE_SUBSAMPLE_2021:
        mask = merged["window_end_date"] >= pd.Timestamp("2021-01-01")
        merged = merged.loc[mask].reset_index(drop=True)
        X_img_aligned = X_img_aligned[mask.values]
        y_aligned = y_aligned[mask.values]

    X_macro = merged[MACRO_FEATURE_COLUMNS].fillna(0.0).astype(np.float32).values
    X_micro = merged[MICRO_FEATURE_COLUMNS].fillna(0.0).astype(np.float32).values

    return X_img_aligned, X_macro, X_micro, y_aligned, merged


def time_split(X_img, X_macro, X_micro, y, meta_df, train_ratio=0.7, val_ratio=0.15):
    n = len(y)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        X_img[:train_end], X_macro[:train_end], X_micro[:train_end], y[:train_end], meta_df.iloc[:train_end].reset_index(drop=True),
        X_img[train_end:val_end], X_macro[train_end:val_end], X_micro[train_end:val_end], y[train_end:val_end], meta_df.iloc[train_end:val_end].reset_index(drop=True),
        X_img[val_end:], X_macro[val_end:], X_micro[val_end:], y[val_end:], meta_df.iloc[val_end:].reset_index(drop=True),
    )


# =========================================================
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x_img, x_macro, x_micro, y in loader:
        x_img = x_img.to(device)
        x_macro = x_macro.to(device)
        x_micro = x_micro.to(device)
        y = y.to(device)

        x_macro, x_micro = apply_modality_dropout(
            x_macro,
            x_micro,
            MACRO_DROPOUT_PROB,
            MICRO_DROPOUT_PROB,
        )

        optimizer.zero_grad()
        logits, _ = model(x_img, x_macro, x_micro)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_img.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, eval_mode="full"):
    model.eval()

    total_loss = 0.0
    all_true, all_pred, all_prob, all_alpha = [], [], [], []

    for x_img, x_macro, x_micro, y in loader:
        x_img = x_img.to(device)
        x_macro = x_macro.to(device)
        x_micro = x_micro.to(device)
        y = y.to(device)

        x_macro, x_micro = apply_eval_mask(x_macro, x_micro, eval_mode)

        logits, alpha = model(x_img, x_macro, x_micro)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * x_img.size(0)

        all_true.extend(y.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_prob.extend(probs.cpu().numpy())
        all_alpha.extend(alpha.squeeze(1).cpu().numpy())

    return {
        "mode": eval_mode,
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_true, all_pred),
        "f1": f1_score(all_true, all_pred, zero_division=0),
        "precision": precision_score(all_true, all_pred, zero_division=0),
        "recall": recall_score(all_true, all_pred, zero_division=0),
        "alpha_mean": float(np.mean(all_alpha)),
        "alpha_std": float(np.std(all_alpha)),
        "y_true": np.array(all_true),
        "y_pred": np.array(all_pred),
        "y_prob": np.array(all_prob),
        "alpha_all": np.array(all_alpha),
    }


# =========================================================
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_img, X_macro, X_micro, y, merged_meta = load_and_align_data()

    print(f"[INFO] X_img shape: {X_img.shape}")
    print(f"[INFO] X_macro shape: {X_macro.shape}")
    print(f"[INFO] X_micro shape: {X_micro.shape}")
    print(f"[INFO] Target: {TARGET}")
    print(f"[INFO] Macro dropout prob: {MACRO_DROPOUT_PROB}")
    print(f"[INFO] Micro dropout prob: {MICRO_DROPOUT_PROB}")

    (
        X_img_train, X_macro_train, X_micro_train, y_train, meta_train,
        X_img_val, X_macro_val, X_micro_val, y_val, meta_val,
        X_img_test, X_macro_test, X_micro_test, y_test, meta_test,
    ) = time_split(X_img, X_macro, X_micro, y, merged_meta)

    macro_scaler = StandardScaler()
    X_macro_train = macro_scaler.fit_transform(X_macro_train)
    X_macro_val = macro_scaler.transform(X_macro_val)
    X_macro_test = macro_scaler.transform(X_macro_test)

    micro_scaler = StandardScaler()
    X_micro_train = micro_scaler.fit_transform(X_micro_train)
    X_micro_val = micro_scaler.transform(X_micro_val)
    X_micro_test = micro_scaler.transform(X_micro_test)

    train_loader = DataLoader(
        ImageMacroMicroDataset(X_img_train, X_macro_train, X_micro_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        ImageMacroMicroDataset(X_img_val, X_macro_val, X_micro_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        ImageMacroMicroDataset(X_img_test, X_macro_test, X_micro_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ImageMacroMicroGatedModel(
        n_macro_features=len(MACRO_FEATURE_COLUMNS),
        n_micro_features=len(MICRO_FEATURE_COLUMNS),
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE, eval_mode="full")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_alpha_mean": val_metrics["alpha_mean"],
            "val_alpha_std": val_metrics["alpha_std"],
        })

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"alpha={val_metrics['alpha_mean']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        if patience >= EARLY_STOPPING_PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_modes = ["full", "drop_macro", "drop_micro", "drop_both"]
    results = []

    print("\n=== Final Test Metrics by Modality Mask ===")
    for mode in eval_modes:
        m = evaluate(model, test_loader, criterion, DEVICE, eval_mode=mode)
        results.append({
            "model": "gated_modality_dropout",
            "target": TARGET,
            "eval_mode": mode,
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "alpha_mean": m["alpha_mean"],
            "alpha_std": m["alpha_std"],
        })

        print(
            f"[{mode}] "
            f"Acc={m['accuracy']:.4f} | "
            f"F1={m['f1']:.4f} | "
            f"Precision={m['precision']:.4f} | "
            f"Recall={m['recall']:.4f} | "
            f"AlphaMean={m['alpha_mean']:.4f}"
        )

    suffix = f"{TARGET}_{'sub2021' if USE_SUBSAMPLE_2021 else 'full'}"

    history_df = pd.DataFrame(history)
    history_file = OUTPUT_DIR / f"metrics_gated_dropout_{suffix}_history.csv"
    history_df.to_csv(history_file, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(results)
    summary_file = OUTPUT_DIR / f"metrics_gated_dropout_{suffix}_summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print('\n[OK]')
    print(f"- {history_file}")
    print(f"- {summary_file}")


if __name__ == "__main__":
    main()
