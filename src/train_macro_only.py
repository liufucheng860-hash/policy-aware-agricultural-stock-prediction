import os
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =====================
# Config
# =====================
STOCK_CSV_PATH = "data/raw/xingguang/xingguang_daily.csv"      #
MACRO_CSV_PATH = "data/raw/macro_policy_yearly.csv"     #
OUTPUT_DIR = "outputs/xingguang/macro_only"

DATE_COL = "trade_date"
PRICE_COL = "close"

HORIZON = 10
SEED = 42

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================
# Seed
# =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# Dataset
# =====================
class MacroDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================
# Model
# =====================
class MacroOnlyMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# =====================
# Load stock daily data
# =====================
stock_df = pd.read_csv(STOCK_CSV_PATH)
stock_df[DATE_COL] = pd.to_datetime(stock_df[DATE_COL])
stock_df = stock_df.sort_values(DATE_COL).reset_index(drop=True)

stock_df["year"] = stock_df[DATE_COL].dt.year

stock_df[f"future_close_{HORIZON}d"] = stock_df[PRICE_COL].shift(-HORIZON)
stock_df[f"ret_{HORIZON}d"] = (
    stock_df[f"future_close_{HORIZON}d"] - stock_df[PRICE_COL]
) / stock_df[PRICE_COL]

stock_df[f"label_{HORIZON}d"] = (stock_df[f"ret_{HORIZON}d"] > 0).astype(int)


# =====================
# Load macro yearly data
# =====================
macro_df = pd.read_csv(MACRO_CSV_PATH)

drop_cols = [c for c in macro_df.columns if "relevant" in c.lower()]
macro_df = macro_df.drop(columns=drop_cols, errors="ignore")

macro_cols = [
    c for c in macro_df.columns
    if c != "year"
]

print("Macro columns used:")
print(macro_cols)


# =====================
# Merge by year
# =====================
df = stock_df.merge(macro_df, on="year", how="left")

label_col = f"label_{HORIZON}d"

df = df.dropna(subset=macro_cols + [label_col])

X = df[macro_cols].values
y = df[label_col].values.astype(int)

print("\nData range:")
print(df[DATE_COL].min(), "to", df[DATE_COL].max())
print("X shape:", X.shape)
print("y distribution:", np.bincount(y))


# =====================
# Chronological split
# =====================
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

train_loader = DataLoader(
    MacroDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=False
)

val_loader = DataLoader(
    MacroDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    MacroDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =====================
# Train
# =====================
model = MacroOnlyMLP(input_dim=X_train.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

best_val_f1 = -1
best_state = None
bad_epochs = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(yb)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_probs = []
    val_true = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()

            val_probs.extend(probs)
            val_true.extend(yb.numpy())

    val_pred = (np.array(val_probs) >= 0.5).astype(int)
    val_f1 = f1_score(val_true, val_pred, zero_division=0)

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_f1={val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = model.state_dict()
        bad_epochs = 0
    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print("Early stopping.")
        break


# =====================
# Test
# =====================
model.load_state_dict(best_state)
model.eval()

test_probs = []
test_true = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()

        test_probs.extend(probs)
        test_true.extend(yb.numpy())

test_pred = (np.array(test_probs) >= 0.5).astype(int)

acc = accuracy_score(test_true, test_pred)
f1 = f1_score(test_true, test_pred, zero_division=0)
precision = precision_score(test_true, test_pred, zero_division=0)
recall = recall_score(test_true, test_pred, zero_division=0)

print("\n========== Macro-only Test Result ==========")
print(f"Horizon  : {HORIZON}d")
print(f"Accuracy : {acc:.4f}")
print(f"F1       : {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")

result = pd.DataFrame([{
    "model": "Macro-only",
    "horizon": HORIZON,
    "accuracy": acc,
    "f1": f1,
    "precision": precision,
    "recall": recall
}])

save_path = os.path.join(
    OUTPUT_DIR,
    f"macro_only_{HORIZON}d.csv"
)

result.to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"\n[OK] saved: {save_path}")
