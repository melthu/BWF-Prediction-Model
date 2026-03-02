import torch
import torch.nn as nn


class BWFDeepFM(nn.Module):
    """
    Deep Factorization Machine for BWF Men's Singles match prediction.

    Categorical inputs (4): tier, round, player_a, player_b
    Continuous inputs (10): scaled numerical features

    Architecture:
      FM component  — 2nd-order pairwise interactions between the 4 embeddings
      Deep component — MLP over concatenated embeddings + continuous features
      Output         — single logit (use BCEWithLogitsLoss, no sigmoid here)
    """

    def __init__(
        self,
        vocab_sizes: dict,          # {"num_tiers", "num_rounds", "num_players"}
        embed_dim: int = 16,
        num_cont_features: int = 10,
        hidden_dims: list[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        # ------------------------------------------------------------------
        # Embedding tables (shared embed_dim for FM formula consistency)
        # ------------------------------------------------------------------
        self.embed_tier     = nn.Embedding(vocab_sizes["num_tiers"],   embed_dim)
        self.embed_round    = nn.Embedding(vocab_sizes["num_rounds"],  embed_dim)
        self.embed_player_a = nn.Embedding(vocab_sizes["num_players"], embed_dim, padding_idx=0)
        self.embed_player_b = nn.Embedding(vocab_sizes["num_players"], embed_dim, padding_idx=0)

        self.num_fields = 4   # tier, round, player_a, player_b

        # ------------------------------------------------------------------
        # Deep component MLP
        # deep_input_dim = num_fields * embed_dim + num_cont_features
        # ------------------------------------------------------------------
        deep_input_dim = self.num_fields * embed_dim + num_cont_features
        layers = []
        in_dim = deep_input_dim
        for out_dim in hidden_dims:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(p=0.2)]
            in_dim = out_dim
        self.deep = nn.Sequential(*layers)

        # ------------------------------------------------------------------
        # Output: FM scalar (1) + Deep last hidden (hidden_dims[-1]) → logit
        # ------------------------------------------------------------------
        self.output_layer = nn.Linear(1 + hidden_dims[-1], 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, cat_features: torch.Tensor, cont_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cat_features  : LongTensor  (B, 4)  — [tier, round, player_a, player_b]
            cont_features : FloatTensor (B, 10)
        Returns:
            logits        : FloatTensor (B, 1)
        """
        # --- Embed each categorical field → list of (B, embed_dim) ---
        e_tier     = self.embed_tier(cat_features[:, 0])      # (B, E)
        e_round    = self.embed_round(cat_features[:, 1])     # (B, E)
        e_player_a = self.embed_player_a(cat_features[:, 2])  # (B, E)
        e_player_b = self.embed_player_b(cat_features[:, 3])  # (B, E)

        # Stack → (B, num_fields, embed_dim)
        embeds = torch.stack([e_tier, e_round, e_player_a, e_player_b], dim=1)

        # --- FM 2nd-order: ½ [(Σ eᵢ)² − Σ eᵢ²] summed over embed dim ---
        sum_of_embeds = embeds.sum(dim=1)           # (B, E)
        sum_of_squares = (embeds ** 2).sum(dim=1)   # (B, E)
        fm_out = 0.5 * ((sum_of_embeds ** 2) - sum_of_squares)  # (B, E)
        fm_out = fm_out.sum(dim=1, keepdim=True)    # (B, 1)

        # --- Deep: flatten embeddings + concat continuous ---
        flat_embeds  = embeds.view(embeds.size(0), -1)          # (B, num_fields*E)
        deep_input   = torch.cat([flat_embeds, cont_features], dim=1)  # (B, deep_input_dim)
        deep_out     = self.deep(deep_input)                    # (B, hidden_dims[-1])

        # --- Combine FM and Deep → single logit ---
        combined = torch.cat([fm_out, deep_out], dim=1)         # (B, 1 + hidden_dims[-1])
        logits   = self.output_layer(combined)                  # (B, 1)
        return logits
