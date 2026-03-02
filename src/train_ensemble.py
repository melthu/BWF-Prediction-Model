import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import get_train_val_datasets
from src.model import BWFDeepFM

DATA_PATH  = "data/processed/final_training_data.csv"
MODEL_PATHS = {
    "lgbm":     "models/best_lgbm.pkl",
    "catboost": "models/best_catboost.pkl",
    "xgb":      "models/best_xgb.pkl",
}
DEEPFM_PATH     = "models/best_deepfm.pt"
BEST_MODEL_PATH = "models/best_model.pkl"
DEEPFM_AUC_THRESHOLD = 0.74   # include DeepFM in ensemble only if AUC >= this


class DeepFMWrapper:
    """
    Wraps a trained BWFDeepFM so it exposes the same predict_proba(X) interface
    as scikit-learn / LightGBM / XGBoost classifiers.

    X is a 2-D numpy array with shape (N, 4 + num_cont_features).
    Columns 0-3 are the categorical IDs (int64); the rest are scaled continuous.
    """

    def __init__(self, model: BWFDeepFM):
        self.model = model
        self.model.eval()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cat  = torch.tensor(X[:, :4],  dtype=torch.long)
        cont = torch.tensor(X[:, 4:],  dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(cat, cont).squeeze(1)
            probs  = torch.sigmoid(logits).numpy()
        # Return (N, 2) to match sklearn convention — col 1 = P(class=1)
        return np.column_stack([1 - probs, probs])


def extract_numpy(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    cat, cont, labels = next(iter(loader))
    X = np.hstack([cat.numpy(), cont.numpy()])
    y = labels.numpy().ravel()
    return X, y


def load_tree_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"  Loaded {name} from {path}")
    return models


def load_deepfm_wrapper():
    """Try to load the DeepFM checkpoint; return (wrapper, auc) or (None, None)."""
    if not os.path.exists(DEEPFM_PATH):
        print(f"  DeepFM checkpoint not found at {DEEPFM_PATH} — skipping.")
        return None, None

    ckpt = torch.load(DEEPFM_PATH, map_location="cpu")
    vocab_sizes = ckpt["vocab_sizes"]
    saved_auc   = ckpt.get("val_auc", 0.0)

    model = BWFDeepFM(
        vocab_sizes=vocab_sizes,
        embed_dim=32,
        num_cont_features=24,
        hidden_dims=[256, 128, 64],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    wrapper = DeepFMWrapper(model)
    print(f"  Loaded deepfm from {DEEPFM_PATH}  (checkpoint val AUC = {saved_auc:.4f})")
    return wrapper, saved_auc


def train():
    _, val_ds, _, _ = get_train_val_datasets(DATA_PATH)
    X_val, y_val = extract_numpy(val_ds)

    print("Loading saved tree models...")
    tree_models = load_tree_models()

    print("\nLoading DeepFM...")
    deepfm_wrapper, deepfm_ckpt_auc = load_deepfm_wrapper()

    # ------------------------------------------------------------------
    # Individual AUCs (tree models)
    # ------------------------------------------------------------------
    print("\nIndividual model AUCs:")
    individual_probs: dict[str, np.ndarray] = {}
    individual_aucs:  dict[str, float]      = {}

    for name, model in tree_models.items():
        probs = model.predict_proba(X_val)[:, 1]
        auc   = roc_auc_score(y_val, probs)
        individual_probs[name] = probs
        individual_aucs[name]  = auc
        print(f"  {name:<12} AUC = {auc:.4f}")

    # Optionally add DeepFM
    if deepfm_wrapper is not None:
        deepfm_probs = deepfm_wrapper.predict_proba(X_val)[:, 1]
        deepfm_auc   = roc_auc_score(y_val, deepfm_probs)
        print(f"  {'deepfm':<12} AUC = {deepfm_auc:.4f}", end="")
        if deepfm_auc >= DEEPFM_AUC_THRESHOLD:
            individual_probs["deepfm"] = deepfm_probs
            individual_aucs["deepfm"]  = deepfm_auc
            print(f"  ✓ included (>= {DEEPFM_AUC_THRESHOLD})")
        else:
            print(f"  ✗ excluded (< {DEEPFM_AUC_THRESHOLD})")
    else:
        deepfm_auc = 0.0

    best_single_name = max(individual_aucs, key=individual_aucs.get)
    best_single_auc  = individual_aucs[best_single_name]

    # ------------------------------------------------------------------
    # Equal-weight ensemble over qualifying models
    # ------------------------------------------------------------------
    weights = [1 / len(individual_probs)] * len(individual_probs)
    ensemble_probs = sum(
        w * individual_probs[name]
        for w, name in zip(weights, individual_probs.keys())
    )
    ensemble_auc = roc_auc_score(y_val, ensemble_probs)

    print(f"\n  {'ensemble':<12} AUC = {ensemble_auc:.4f}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*46}")
    print(f"  Ensemble Results")
    print(f"{'='*46}")
    for name, auc in individual_aucs.items():
        marker = " *" if name == best_single_name else ""
        print(f"  {name:<12}: {auc:.4f}{marker}")
    print(f"  {'ensemble':<12}: {ensemble_auc:.4f}")
    print(f"{'='*46}")

    # ------------------------------------------------------------------
    # Build payload — include DeepFMWrapper only if it was added to ensemble
    # ------------------------------------------------------------------
    all_models: dict = {}
    for name, m in tree_models.items():
        if name in individual_probs:
            all_models[name] = m
    if "deepfm" in individual_probs:
        all_models["deepfm"] = deepfm_wrapper

    if ensemble_auc >= best_single_auc:
        payload = {
            "type":    "ensemble",
            "models":  all_models,
            "weights": [1 / len(all_models)] * len(all_models),
            "names":   list(all_models.keys()),
        }
        saved_auc  = ensemble_auc
        saved_name = "ensemble"
    else:
        best_model = all_models.get(best_single_name,
                                    tree_models[best_single_name])
        payload = {
            "type":  "single",
            "model": best_model,
            "name":  best_single_name,
        }
        saved_auc  = best_single_auc
        saved_name = best_single_name

    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n  Winner : {saved_name}  (AUC = {saved_auc:.4f})")
    print(f"  Saved  : {BEST_MODEL_PATH}")
    print(f"{'='*46}")


if __name__ == "__main__":
    train()
