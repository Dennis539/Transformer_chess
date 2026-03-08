from huggingface_hub import PyTorchModelHubMixin
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ValueTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Project per-square features (C) -> d_model - Kind of like a feed-forward layer for backward propagation
        self.input_proj = nn.Linear(in_channels, d_model)

        # Learned positional embedding for 64 squares - parameter makes sure values are registered during backpropagation.
        self.pos_emb = nn.Parameter(torch.zeros(64, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Pool -> value head
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),  # Activation function
            nn.Linear(d_model, 1),
        )
        self.loss_fn = nn.SmoothL1Loss(beta=0.1)

    def forward(self, x=None, labels=None):
        """
        x: (B, C, 8, 8)
        returns: (B,) normalized value in roughly [-1,1]
        """
        B, C, H, W = x.shape
        assert (H, W) == (8, 8)

        # Make square tokens: (B, 64, C)
        # Permute switched dimensions
        tokens = x.permute(0, 2, 3, 1).reshape(B, 64, C)

        # Each square becomes a 256-dimensional embedding - Turning chess features into “neural language”
        h = self.input_proj(tokens)

        # add position info - Each square index gets a learnable vector.
        h = h + self.pos_emb.unsqueeze(0)

        h = self.encoder(h)  # (B,64,d_model)
        pooled = h.mean(dim=1)  # (B,d_model) mean pool

        pred = self.value_head(pooled).squeeze(-1)  # (B,)
        loss = None
        if labels is not None:
            loss = self.loss_fn(pred, labels)

        return {"loss": loss, "logits": pred}


class EndgamePolicyTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        num_moves: int,
        in_channels: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Project per-square features (C) -> d_model - Kind of like a feed-forward layer for backward propagation
        self.input_proj = nn.Linear(in_channels, d_model)

        self.pos_emb = nn.Parameter(torch.zeros(64, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_moves),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        """
        x: (B, C, 8, 8)
        returns: dict with
            logits: (B, num_moves)
            loss: scalar or None
        """
        B, C, H, W = x.shape
        assert (H, W) == (8, 8)

        tokens = x.permute(0, 2, 3, 1).reshape(B, 64, C)
        h = self.input_proj(tokens)

        h = h + self.pos_emb.unsqueeze(0)

        h = self.encoder(h)
        pooled = h.mean(dim=1)

        logits = self.policy_head(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
