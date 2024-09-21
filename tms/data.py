""" Data generation for toy models of superposition """

import torch
import einops

from abc import ABC, abstractmethod
from jaxtyping import Float
from torch import Tensor
from tms.utils.device import get_device

class DataGenerator(ABC):

    n_features: int

    @abstractmethod
    def generate_batch(batch_size: int) -> Float[Tensor, "batch feats"]:
        """ Override with custom logic to generate a batch of data. """
        pass


class IIDGenerator(DataGenerator):
    """ Generates features IID. 

    - Each feature is present with probability `feature_probability`.
    - For each present feature, the magnitude of each feature is drawn from Uniform(0, 1).
    """

    def __init__(self, n_features: int, n_inst: int, feature_probability: Float[Tensor, "inst"]):
        self.n_features = n_features
        self.n_inst = n_inst
        self.feature_probability = feature_probability

    def generate_batch(self, batch_size: int) -> Float[Tensor, "batch inst feats"]:
        """
        Generates a batch of data.
        """
        batch_shape = (batch_size, self.n_inst, self.n_features)
        feat_mag = torch.rand(batch_shape, device=get_device())
        feat_seeds = torch.rand(batch_shape, device=get_device())
        feat_probs = einops.repeat(self.feature_probability, 'inst -> batch inst feats', batch=batch_size, feats = self.n_features)
        return torch.where(feat_seeds <= feat_probs, feat_mag, 0.0)