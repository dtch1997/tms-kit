""" Define the architecture for a TMS """

import torch

from jaxtyping import Float
from abc import ABC
from tms.data import DataGenerator, IIDFeatureGenerator
from tms.loss import LossCalculator, ImportanceWeightedLoss
from tms.model import Model
from dataclasses import dataclass

class TMS(ABC):
    """ A TMS bundles a model architecture, data generator, and loss calculator."""
    model: Model
    data_gen: DataGenerator
    loss_calc: LossCalculator

    def __init__(self, model: Model, data_gen: DataGenerator, loss_calc: LossCalculator):
        self.model = model
        self.data_gen = data_gen
        self.loss_calc = loss_calc

@dataclass
class BottleneckTMSConfig:
    d_hidden: int
    n_inst: int
    n_features: int
    feature_probability: Float[torch.Tensor, "inst feats"]
    feature_importance: Float[torch.Tensor, "inst feats"]

class BottleneckTMS(TMS):

    config: BottleneckTMSConfig

    """ The original TMS setup from https://transformer-circuits.pub/2022/toy_model/index.html """
    def __init__(self, config: BottleneckTMSConfig):
        model = Model(config.n_features, config.n_inst, config.d_hidden)
        data_gen = IIDFeatureGenerator(config.n_features, config.n_inst, config.feature_probability)
        loss_calc = ImportanceWeightedLoss(config.n_features, config.feature_importance)
        super().__init__(model, data_gen, loss_calc)

        self.config = config