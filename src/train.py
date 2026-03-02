import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.dataset import get_train_val_datasets
from src.model import BWFDeepFM

DATA_PATH    = "data/processed/final_training_data.csv"
MODEL_PATH   = "models/best_deepfm.pt"

BATCH_SIZE   = 64
LR           = 5e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 50
PATIENCE     = 5


def train():
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds, val_ds, vocab_sizes, _ = get_train_val_datasets(DATA_PATH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train size : {len(train_ds)}  |  Val size : {len(val_ds)}")
    print(f"Vocab sizes: {vocab_sizes}\n")

    # ------------------------------------------------------------------
    # Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = BWFDeepFM(
        vocab_sizes=vocab_sizes,
        embed_dim=32,
        num_cont_features=24,
        hidden_dims=[256, 128, 64],
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=MAX_EPOCHS, eta_min=1e-5
    )

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_val_auc  = 0.0
    epochs_no_improve = 0

    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val Acc':>7} | {'Val AUC':>7}")
    print("-" * 52)

    for epoch in range(1, MAX_EPOCHS + 1):

        # --- Train ---
        model.train()
        train_loss = 0.0
        for cat, cont, labels in train_loader:
            cat, cont, labels = cat.to(device), cont.to(device), labels.to(device)
            optimiser.zero_grad()
            logits = model(cat, cont)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_ds)
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for cat, cont, labels in val_loader:
                cat, cont, labels = cat.to(device), cont.to(device), labels.to(device)
                logits = model(cat, cont)
                val_loss += criterion(logits, labels).item() * len(labels)
                all_logits.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels.cpu())
        val_loss /= len(val_ds)

        all_probs     = torch.cat(all_logits).numpy().ravel()
        all_labels_np = torch.cat(all_labels).numpy().ravel()

        val_acc = ((all_probs >= 0.5) == all_labels_np).mean()
        val_auc = roc_auc_score(all_labels_np, all_probs)

        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>8.4f} | {val_acc:>7.4f} | {val_auc:>7.4f}")

        # --- Early stopping & checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc  = val_auc
            epochs_no_improve = 0
            # Save full checkpoint including vocab_sizes (needed for ensemble loading)
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_sizes":      vocab_sizes,
                "val_auc":          val_auc,
            }, MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("\nEarly stopping triggered.")
                break

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Best val AUC  : {best_val_auc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
