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


class BWFDataset(Dataset):
    """
    PyTorch Dataset for BWF Men's Singles match prediction.

    Categorical features (4):  tier, round, player_a, player_b
    Continuous features (10):  see CONT_COLS above
    Target (1):                player_a_won  (0 or 1)

    Shared vocabulary rule: player_a and player_b are encoded with the same
    player_to_id dictionary so the model learns one embedding per player
    regardless of which slot they appear in.
    """

    def __init__(self, csv_path: str = DATA_PATH):
        df = pd.read_csv(csv_path)

        # ------------------------------------------------------------------
        # Categorical vocabularies
        # ------------------------------------------------------------------

        # Shared player vocab — union of both columns
        all_players = sorted(
            set(df["player_a"].unique()) | set(df["player_b"].unique())
        )
        self.player_to_id = {name: idx for idx, name in enumerate(all_players)}
        self.num_players = len(self.player_to_id)

        tiers = sorted(df["tier"].unique())
        self.tier_to_id = {t: i for i, t in enumerate(tiers)}
        self.num_tiers = len(self.tier_to_id)

        df["round"] = df["round"].str.lower()
        rounds = sorted(df["round"].unique())
        self.round_to_id = {r: i for i, r in enumerate(rounds)}
        self.num_rounds = len(self.round_to_id)

        # ------------------------------------------------------------------
        # Pre-encode categorical columns → int64 array (N, 4)
        # Column order: [tier, round, player_a, player_b]
        # ------------------------------------------------------------------
        self.cat = np.column_stack([
            df["tier"].map(self.tier_to_id).values,
            df["round"].map(self.round_to_id).values,
            df["player_a"].map(self.player_to_id).values,
            df["player_b"].map(self.player_to_id).values,
        ]).astype(np.int64)

        # ------------------------------------------------------------------
        # Scale continuous columns → float32 array (N, 10)
        # ------------------------------------------------------------------
        self.scaler = StandardScaler()
        self.cont = self.scaler.fit_transform(
            df[CONT_COLS].values
        ).astype(np.float32)

        # ------------------------------------------------------------------
        # Target
        # ------------------------------------------------------------------
        self.labels = df["player_a_won"].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Returns:
            cat_features  : LongTensor  (4,)   — tier, round, player_a, player_b indices
            cont_features : FloatTensor (10,)  — scaled continuous features
            label         : FloatTensor (1,)   — player_a_won
        """
        cat_features  = torch.tensor(self.cat[idx],         dtype=torch.long)
        cont_features = torch.tensor(self.cont[idx],        dtype=torch.float32)
        label         = torch.tensor([self.labels[idx]],    dtype=torch.float32)
        return cat_features, cont_features, label


if __name__ == "__main__":
    dataset = BWFDataset()

    print("=== Vocabulary Sizes ===")
    print(f"  num_players : {dataset.num_players}")
    print(f"  num_tiers   : {dataset.num_tiers}   {sorted(dataset.tier_to_id.keys())}")
    print(f"  num_rounds  : {dataset.num_rounds}  {sorted(dataset.round_to_id.keys())}")
    print(f"  total rows  : {len(dataset)}")

    print("\n=== dataset[0] ===")
    cat, cont, label = dataset[0]
    print(f"  cat_features  : {cat}  shape={tuple(cat.shape)}  dtype={cat.dtype}")
    print(f"  cont_features : {cont}  shape={tuple(cont.shape)}  dtype={cont.dtype}")
    print(f"  label         : {label}  shape={tuple(label.shape)}  dtype={label.dtype}")
