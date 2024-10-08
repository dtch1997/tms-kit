from __future__ import annotations

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from jaxtyping import Float

from tms_kit.utils.device import get_device


def linear_lr(step, steps):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


class Model(nn.Module):
    W: Float[Tensor, "n_inst d_hidden feats"]
    b_final: Float[Tensor, "n_inst feats"]

    n_inst: int
    n_features: int
    d_hidden: int

    def __init__(
        self, n_features: int, n_inst, d_hidden: int, device: str | None = None
    ):
        if device is None:
            device = get_device()

        super(Model, self).__init__()

        self.W = nn.Parameter(
            nn.init.xavier_normal_(torch.empty((n_inst, d_hidden, n_features)))
        )
        self.b_final = nn.Parameter(torch.zeros((n_inst, n_features)))
        self.to(device)

        self.n_inst = n_inst
        self.n_features = n_features
        self.d_hidden = d_hidden

    def encode(
        self, features: Float[Tensor, "... inst feats"]
    ) -> Float[Tensor, "... inst hidden"]:
        return einops.einsum(
            features, self.W, "... inst feats, inst hidden feats -> ... inst hidden"
        )

    def decode(
        self, hidden: Float[Tensor, "... inst hidden"]
    ) -> Float[Tensor, "... inst feats"]:
        out = einops.einsum(
            hidden, self.W, "... inst hidden, inst hidden feats -> ... inst feats"
        )
        return F.relu(out + self.b_final)

    def forward(
        self,
        features: Float[Tensor, "... inst feats"],
    ) -> Float[Tensor, "... inst feats"]:
        h = self.encode(features)
        return self.decode(h)
