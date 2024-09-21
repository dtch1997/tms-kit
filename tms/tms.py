"""Define the architecture for a TMS"""

from abc import ABC
from tms.data import DataGenerator
from tms.loss import LossCalculator
from tms.model import Model


class TMS(ABC):
    """A TMS bundles a model architecture, data generator, and loss calculator."""

    model: Model
    data_gen: DataGenerator
    loss_calc: LossCalculator

    def __init__(
        self, model: Model, data_gen: DataGenerator, loss_calc: LossCalculator
    ):
        self.model = model
        self.data_gen = data_gen
        self.loss_calc = loss_calc
