import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score

from src.dataset import extract_numpy, get_train_val_datasets

DATA_PATH  = "data/processed/final_training_data.csv"
MODEL_PATH = "models/best_tabnet.pkl"


class TabNetWrapper:
    """
    Thin wrapper exposing the same predict_proba(X) interface as the
    tree models and DeepFMWrapper, plus n_features_in_ for X-trimming.
    """

    def __init__(self, model: TabNetClassifier, n_features: int):
        self.model           = model
        self.n_features_in_  = n_features

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def train():
    # ------------------------------------------------------------------
    # Data — same split/encoding/scaling as all other trainers
    # ------------------------------------------------------------------
    train_ds, val_ds, vocab_sizes, _ = get_train_val_datasets(DATA_PATH)

    print(f"Train size : {len(train_ds)}  |  Val size : {len(val_ds)}")
    print(f"Vocab sizes: {vocab_sizes}\n")

    X_train, y_train = extract_numpy(train_ds)
    X_val,   y_val   = extract_numpy(val_ds)

    print(f"X_train: {X_train.shape}  |  X_val: {X_val.shape}\n")

    # ------------------------------------------------------------------
    # Categorical spec
    # Columns 0-3 are integer-encoded categoricals: tier, round, pa, pb
    # TabNet will create its own embeddings for these.
    # ------------------------------------------------------------------
    cat_idxs = [0, 1, 2, 3]
    cat_dims  = [
        vocab_sizes["num_tiers"],
        vocab_sizes["num_rounds"],
        vocab_sizes["num_players"],
        vocab_sizes["num_players"],
    ]

    # ------------------------------------------------------------------
    # TabNet
    # ------------------------------------------------------------------
    clf = TabNetClassifier(
        cat_idxs         = cat_idxs,
        cat_dims         = cat_dims,
        cat_emb_dim      = 8,      # embedding dim for each categorical
        n_d              = 32,     # width of decision-step output
        n_a              = 32,     # width of attention embedding (should equal n_d)
        n_steps          = 5,      # number of sequential attention steps
        gamma            = 1.5,    # feature-reuse coefficient (>1 = more reuse)
        n_independent    = 2,      # independent FC layers per step
        n_shared         = 2,      # shared FC layers per step
        lambda_sparse    = 1e-4,   # sparsity regularisation
        optimizer_fn     = torch.optim.Adam,
        optimizer_params = {"lr": 2e-2, "weight_decay": 1e-5},
        scheduler_fn     = torch.optim.lr_scheduler.StepLR,
        scheduler_params = {"step_size": 15, "gamma": 0.9},
        mask_type        = "sparsemax",
        verbose          = 10,
        seed             = 42,
    )

    clf.fit(
        X_train          = X_train,
        y_train          = y_train,
        eval_set         = [(X_val, y_val)],
        eval_name        = ["val"],
        eval_metric      = ["auc"],
        max_epochs       = 200,
        patience         = 25,
        batch_size       = 1024,
        virtual_batch_size = 256,
        num_workers      = 0,
        drop_last        = False,
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    val_acc   = (val_preds == y_val).mean()
    val_auc   = roc_auc_score(y_val, val_probs)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    wrapper = TabNetWrapper(clf, n_features=X_train.shape[1])
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(wrapper, f)
    print(f"\nModel saved to: {MODEL_PATH}")

    BASELINE = 0.7872  # XGBoost
    print(f"\n{'='*40}")
    print(f"  TabNet Results")
    print(f"{'='*40}")
    print(f"  Val Accuracy : {val_acc:.4f}")
    print(f"  Val ROC-AUC  : {val_auc:.4f}")
    print(f"\n  Benchmark Comparison")
    print(f"  XGBoost AUC  : {BASELINE:.4f}")
    print(f"  TabNet  AUC  : {val_auc:.4f}  "
          f"({'▲ better' if val_auc > BASELINE else '▼ worse'})")
    print(f"{'='*40}")


if __name__ == "__main__":
    train()
