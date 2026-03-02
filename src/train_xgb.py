import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import get_train_val_datasets

DATA_PATH   = "data/processed/final_training_data.csv"
DEEPFM_AUC  = 0.7011
LGBM_AUC    = 0.7330


def extract_numpy(dataset):
    """Pull all tensors from a BWFDataset and return (X, y) numpy arrays."""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    cat, cont, labels = next(iter(loader))
    X = np.hstack([cat.numpy(), cont.numpy()])
    y = labels.numpy().ravel()
    return X, y


def train():
    # ------------------------------------------------------------------
    # Data — identical split/scaling/vocab as DeepFM and LightGBM runs
    # ------------------------------------------------------------------
    train_ds, val_ds, vocab_sizes, _ = get_train_val_datasets(DATA_PATH)

    print(f"Train size : {len(train_ds)}  |  Val size : {len(val_ds)}")
    print(f"Vocab sizes: {vocab_sizes}\n")

    X_train, y_train = extract_numpy(train_ds)
    X_val,   y_val   = extract_numpy(val_ds)

    print(f"X_train shape: {X_train.shape}  |  X_val shape: {X_val.shape}\n")

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="auc",
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"Best iteration : {model.best_iteration}")
    print(f"Best val AUC   : {model.best_score:.4f}\n")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    val_acc = (val_preds == y_val).mean()
    val_auc = roc_auc_score(y_val, val_probs)

    def tag(auc):
        if auc > LGBM_AUC:
            return "▲ best so far"
        if auc > DEEPFM_AUC:
            return "▲ beats DeepFM"
        return "▼ below DeepFM"

    print(f"{'='*42}")
    print(f"  XGBoost Results")
    print(f"{'='*42}")
    print(f"  Val Accuracy : {val_acc:.4f}")
    print(f"  Val ROC-AUC  : {val_auc:.4f}")
    print(f"\n  Benchmark Comparison")
    print(f"  DeepFM   AUC : {DEEPFM_AUC:.4f}")
    print(f"  LightGBM AUC : {LGBM_AUC:.4f}")
    print(f"  XGBoost  AUC : {val_auc:.4f}  ({tag(val_auc)})")
    print(f"{'='*42}")


if __name__ == "__main__":
    train()
