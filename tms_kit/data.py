"""Data generation for toy models of superposition"""

import torch
import einops

from abc import ABC, abstractmethod
from jaxtyping import Float
from torch import Tensor
from tms_kit.model import Model
from tms_kit.utils.device import get_device


class DataGenerator(ABC):
    n_features: int
    n_inst: int
    feature_probability: Float[Tensor, "inst feats"]

    def __init__(
        self,
        n_features: int,
        n_inst: int,
        feature_probability: Float[Tensor, "inst feats"],
    ):
        self.n_features = n_features
        self.n_inst = n_inst
        self.feature_probability = feature_probability

    @abstractmethod
    def generate_batch(batch_size: int) -> Float[Tensor, "batch inst feats"]:
        """Override with custom logic to generate a batch of data."""
        pass


class IIDFeatureGenerator(DataGenerator):
    """Generates features IID.

    - Each feature is present with probability `feature_probability`.
    - For each present feature, the magnitude of each feature is drawn from Uniform(0, 1).
    """

    def generate_batch(self, batch_size: int) -> Float[Tensor, "batch inst feats"]:
        """
        Generates a batch of data.
        """
        batch_shape = (batch_size, self.n_inst, self.n_features)
        feat_mag = torch.rand(batch_shape, device=get_device())
        feat_seeds = torch.rand(batch_shape, device=get_device())
        return torch.where(feat_seeds <= self.feature_probability, feat_mag, 0.0)


class CorrelatedFeatureGenerator(DataGenerator):
    """Generates a batch of correlated features.
    - For each pair `batch[i, j, [2k, 2k+1]]`, one of them is non-zero if and only if the other is non-zero.
    """

    def generate_batch(self, batch_size: int) -> Float[Tensor, "batch inst feats"]:
        """
        Generates a batch of correlated features. For each pair `batch[i, j, [2k, 2k+1]]`, one of
        them is non-zero if and only if the other is non-zero.

        Implemented as follows:
        - Create a boolean mask of shape [batch inst n_correlated_pairs] which represents whether the feature set is present
        - Repeat that mask across feature pairs.
        """
        assert (
            self.n_features % 2 == 0
        ), "Number of features must be even for correlated features."
        n_correlated_pairs = self.n_features // 2

        assert torch.all((self.feature_probability == self.feature_probability[:, [0]]))
        p = self.feature_probability[:, [0]]  # shape (n_inst, 1)

        feat_mag = torch.rand(
            (batch_size, self.n_inst, 2 * n_correlated_pairs), device=get_device()
        )
        feat_set_seeds = torch.rand(
            (batch_size, self.n_inst, n_correlated_pairs), device=get_device()
        )
        feat_set_is_present = feat_set_seeds <= p
        feat_is_present = einops.repeat(
            feat_set_is_present,
            "batch instances features -> batch instances (features pair)",
            pair=2,
        )
        return torch.where(feat_is_present, feat_mag, 0.0)


class AnticorrelatedFeatureGenerator(DataGenerator):
    """Generates a batch of anti-correlated features.

    For each pair `batch[i, j, [2k, 2k+1]]`, each of them can only be non-zero if the other one is zero.
    - batch[i, j, 2k] is present with probability p
    - batch[i, j, 2k+1] is present with probability p / (1 - p), if and only if batch[i, j, 2k] is present.
    """

    def generate_batch(
        self, batch_size: int
    ) -> Float[Tensor, "batch inst 2*n_anticorrelated_pairs"]:
        """
        Generates a batch of anti-correlated features. For each pair `batch[i, j, [2k, 2k+1]]`, each
        of them can only be non-zero if the other one is zero.
        """
        assert (
            self.n_features % 2 == 0
        ), "Number of features must be even for correlated features."
        n_anticorrelated_pairs = self.n_features // 2

        assert torch.all((self.feature_probability == self.feature_probability[:, [0]]))
        p = self.feature_probability[:, [0]]  # shape (n_inst, 1)

        assert p.max().item() <= 0.5, "For anticorrelated features, must have 2p < 1"

        feat_mag = torch.rand(
            (batch_size, self.n_inst, 2 * n_anticorrelated_pairs), device=get_device()
        )
        even_feat_seeds, odd_feat_seeds = torch.rand(
            (2, batch_size, self.n_inst, n_anticorrelated_pairs), device=get_device()
        )
        even_feat_is_present = even_feat_seeds <= p
        odd_feat_is_present = (even_feat_seeds > p) & (odd_feat_seeds <= p / (1 - p))
        feat_is_present = einops.rearrange(
            torch.stack([even_feat_is_present, odd_feat_is_present], dim=0),
            "pair batch instances features -> batch instances (features pair)",
        )
        return torch.where(feat_is_present, feat_mag, 0.0)


class ModelActivationsGenerator:
    """Generates intermediate activations from a model"""

    def __init__(self, model: Model, data_gen: DataGenerator):
        self.model = model
        self.data_gen = data_gen

    @torch.no_grad()
    def generate_batch(self, batch_size: int) -> Float[Tensor, "... inst feats"]:
        orig_batch = self.data_gen.generate_batch(batch_size)
        return self.model.encode(orig_batch)
