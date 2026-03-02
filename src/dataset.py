import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

DATA_PATH = "data/processed/final_training_data.csv"

CONT_COLS = [
    "same_nationality",
    "h2h_win_rate_a_vs_b",
    "player_a_is_home",
    "player_a_matches_last_14_days",
    "player_a_days_since_last_match",
    "player_a_recent_win_rate",
    "player_b_is_home",
    "player_b_matches_last_14_days",
    "player_b_days_since_last_match",
    "player_b_recent_win_rate",
]

UNK_ID = 0  # reserved for players not seen during training


class BWFDataset(Dataset):
    """
    Thin wrapper that holds pre-encoded numpy arrays and serves tensors.
    Encoding and scaling are handled externally by get_train_val_datasets().
    """

    def __init__(self, cat: np.ndarray, cont: np.ndarray, labels: np.ndarray):
        self.cat    = cat.astype(np.int64)
        self.cont   = cont.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Returns:
            cat_features  : LongTensor  (4,)  — [tier, round, player_a, player_b]
            cont_features : FloatTensor (10,) — scaled continuous features
            label         : FloatTensor (1,)  — player_a_won
        """
        return (
            torch.tensor(self.cat[idx],         dtype=torch.long),
            torch.tensor(self.cont[idx],        dtype=torch.float32),
            torch.tensor([self.labels[idx]],    dtype=torch.float32),
        )


def get_train_val_datasets(csv_path: str = DATA_PATH):
    """
    Load final_training_data.csv, split chronologically, and apply
    strictly leakage-free preprocessing:

      - Vocabularies (players, tiers, rounds) built from train only.
      - StandardScaler fit on train only, then applied to both splits.
      - 2026 players unseen in training receive UNK_ID (0).

    Returns:
        train_dataset : BWFDataset
        val_dataset   : BWFDataset
        vocab_sizes   : dict with num_players, num_tiers, num_rounds
    """
    df = pd.read_csv(csv_path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["round"] = df["round"].str.lower()   # normalise casing (safeguard)

    # ------------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------------
    train_df = df[df["start_date"].dt.year <= 2025].copy()
    val_df   = df[df["start_date"].dt.year >= 2026].copy()

    # ------------------------------------------------------------------
    # Build vocabularies from train only
    # ------------------------------------------------------------------

    # Shared player vocab: UNK=0, then sorted training players from 1..N
    train_players = sorted(
        set(train_df["player_a"].unique()) | set(train_df["player_b"].unique())
    )
    player_to_id = {name: idx + 1 for idx, name in enumerate(train_players)}
    # UNK_ID (0) is implicitly assigned to any name not in the dict

    tiers  = sorted(train_df["tier"].unique())
    tier_to_id = {t: i for i, t in enumerate(tiers)}

    rounds = sorted(train_df["round"].unique())
    round_to_id = {r: i for i, r in enumerate(rounds)}

    vocab_sizes = {
        "num_players": len(player_to_id) + 1,   # +1 for UNK slot
        "num_tiers":   len(tier_to_id),
        "num_rounds":  len(round_to_id),
    }

    # ------------------------------------------------------------------
    # Fit scaler on train only
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(train_df[CONT_COLS].values)

    # ------------------------------------------------------------------
    # Encode and scale both splits
    # ------------------------------------------------------------------
    def encode(split_df: pd.DataFrame):
        cat = np.column_stack([
            split_df["tier"].map(tier_to_id).values,
            split_df["round"].map(round_to_id).fillna(0).values,      # unseen rounds → 0
            split_df["player_a"].map(player_to_id).fillna(UNK_ID).values,
            split_df["player_b"].map(player_to_id).fillna(UNK_ID).values,
        ])
        cont   = scaler.transform(split_df[CONT_COLS].values)
        labels = split_df["player_a_won"].values
        return cat, cont, labels

    train_cat, train_cont, train_labels = encode(train_df)
    val_cat,   val_cont,   val_labels   = encode(val_df)

    train_dataset = BWFDataset(train_cat, train_cont, train_labels)
    val_dataset   = BWFDataset(val_cat,   val_cont,   val_labels)

    return train_dataset, val_dataset, vocab_sizes


if __name__ == "__main__":
    train_ds, val_ds, vocab_sizes = get_train_val_datasets()

    print("=== Split Sizes ===")
    print(f"  Train rows : {len(train_ds)}")
    print(f"  Val rows   : {len(val_ds)}")

    print("\n=== Vocabulary Sizes (fit on train) ===")
    print(f"  num_players : {vocab_sizes['num_players']}  (includes 1 UNK slot)")
    print(f"  num_tiers   : {vocab_sizes['num_tiers']}")
    print(f"  num_rounds  : {vocab_sizes['num_rounds']}")

    print("\n=== train_ds[0] ===")
    cat, cont, label = train_ds[0]
    print(f"  cat_features  : {cat}  shape={tuple(cat.shape)}  dtype={cat.dtype}")
    print(f"  cont_features : {cont}  shape={tuple(cont.shape)}  dtype={cont.dtype}")
    print(f"  label         : {label}  shape={tuple(label.shape)}  dtype={label.dtype}")

    print("\n=== val_ds[0] ===")
    cat, cont, label = val_ds[0]
    print(f"  cat_features  : {cat}  shape={tuple(cat.shape)}  dtype={cat.dtype}")
    print(f"  cont_features : {cont}  shape={tuple(cont.shape)}  dtype={cont.dtype}")
    print(f"  label         : {label}  shape={tuple(label.shape)}  dtype={label.dtype}")
