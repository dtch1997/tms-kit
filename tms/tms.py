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