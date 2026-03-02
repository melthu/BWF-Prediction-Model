import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import get_train_val_datasets

DATA_PATH   = "data/processed/final_training_data.csv"
MODEL_PATH  = "models/best_lgbm.pkl"
DEEPFM_AUC  = 0.7011   # DeepFM baseline (epoch 7, best val loss)


def extract_numpy(dataset):
    """
    Pull all tensors from a BWFDataset in one pass and return
    (X, y) numpy arrays with cat and cont features concatenated horizontally.
    """
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    cat, cont, labels = next(iter(loader))
    X = np.hstack([cat.numpy(), cont.numpy()])
    y = labels.numpy().ravel()
    return X, y


def train():
    # ------------------------------------------------------------------
    # Data — identical split/scaling/vocab as the DeepFM run
    # ------------------------------------------------------------------
    train_ds, val_ds, vocab_sizes, _ = get_train_val_datasets(DATA_PATH)

    print(f"Train size : {len(train_ds)}  |  Val size : {len(val_ds)}")
    print(f"Vocab sizes: {vocab_sizes}\n")

    X_train, y_train = extract_numpy(train_ds)
    X_val,   y_val   = extract_numpy(val_ds)

    print(f"X_train shape: {X_train.shape}  |  X_val shape: {X_val.shape}\n")

    # ------------------------------------------------------------------
    # LightGBM — first 4 columns are the encoded categoricals
    # ------------------------------------------------------------------
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=[0, 1, 2, 3],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    val_acc = (val_preds == y_val).mean()
    val_auc = roc_auc_score(y_val, val_probs)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {MODEL_PATH}\n")

    print(f"\n{'='*40}")
    print(f"  LightGBM Results")
    print(f"{'='*40}")
    print(f"  Val Accuracy : {val_acc:.4f}")
    print(f"  Val ROC-AUC  : {val_auc:.4f}")
    print(f"\n  Benchmark Comparison")
    print(f"  DeepFM  AUC  : {DEEPFM_AUC:.4f}")
    print(f"  LightGBM AUC : {val_auc:.4f}  ({'▲ better' if val_auc > DEEPFM_AUC else '▼ worse'})")
    print(f"{'='*40}")


if __name__ == "__main__":
    train()
