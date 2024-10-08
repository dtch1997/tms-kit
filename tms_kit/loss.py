"""Loss function for toy models of superposition"""

import einops

from abc import ABC, abstractmethod
from jaxtyping import Float
from torch import Tensor


class LossCalculator(ABC):
    n_features: int

    @abstractmethod
    def calculate_loss(
        output: Float[Tensor, "... inst feats"], batch: Float[Tensor, "... inst feats"]
    ) -> Float[Tensor, ""]:
        """Override with custom logic to calculate the loss."""
        pass


class ImportanceWeightedLoss(LossCalculator):
    def __init__(
        self, n_features: int, feature_importance: Float[Tensor, "... inst feats"]
    ):
        self.n_features = n_features
        self.feature_importance = feature_importance

    def calculate_loss(
        self,
        pred: Float[Tensor, "... inst feats"],
        target: Float[Tensor, "... inst feats"],
    ) -> Float[Tensor, ""]:
        error = self.feature_importance * ((target - pred) ** 2)
        loss = einops.reduce(error, "... inst feats -> inst", "mean").sum()
        return loss
