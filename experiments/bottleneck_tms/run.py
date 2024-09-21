""" Script to train and visualize a Toy Model of Superposition 

Usage as script: 
- Simply do `python experiments/bottleneck_tms/run.py`
- Figures will be saved in the same directory as the script
- Useful if you want to run all experiments at once and save the results

Usage as Jupyter notebook:
- The first cell defines all the necessary classes and functions
- Each subsequent cell runs a different experiment and visualizes the results
- Useful if you want to run each experiment separately and inspect the results
"""
# %%
import torch 
import einops
import matplotlib.pyplot as plt

from dataclasses import dataclass
from jaxtyping import Float
from typing import Type

from tms.data import (
    DataGenerator,
    IIDFeatureGenerator, 
    CorrelatedFeatureGenerator, 
    AnticorrelatedFeatureGenerator
)
from tms.loss import ImportanceWeightedLoss
from tms.model import Model
from tms.optimize import optimize
from tms.tms import TMS
from tms.utils.device import get_device
from tms.utils import utils
from tms.utils.plotly import line, imshow

@dataclass
class BottleneckTMSConfig:
    d_hidden: int
    n_inst: int
    n_features: int
    feature_probability: Float[torch.Tensor, "inst feats"]
    feature_importance: Float[torch.Tensor, "inst feats"]
    data_gen_cls: Type[DataGenerator] = IIDFeatureGenerator

class BottleneckTMS(TMS):

    config: BottleneckTMSConfig

    """ The original TMS setup from https://transformer-circuits.pub/2022/toy_model/index.html """
    def __init__(self, config: BottleneckTMSConfig):
        model = Model(config.n_features, config.n_inst, config.d_hidden)
        data_gen = config.data_gen_cls(config.n_features, config.n_inst, config.feature_probability)
        loss_calc = ImportanceWeightedLoss(config.n_features, config.feature_importance)
        super().__init__(model, data_gen, loss_calc)

        self.config = config

# %% 
def run_5_2_experiment():
    """ Visualize bottleneck superposition of 5 features in 2 dimensions """

    device = get_device()
    n_inst = 10
    n_features = 5
    d_hidden = 2

    feature_probability = (50 ** -torch.linspace(0, 1, n_inst)).to(device)
    feature_probability = einops.repeat(feature_probability, 'inst -> inst feats', feats=n_features)
    feature_importance = (0.9 ** torch.arange(n_features)).to(device)
    feature_importance = einops.repeat(feature_importance, 'feats -> inst feats', inst=n_inst)

    config = BottleneckTMSConfig(
        d_hidden=d_hidden,
        n_inst=n_inst,
        n_features=n_features,
        feature_probability=feature_probability,
        feature_importance=feature_importance
    )
    tms = BottleneckTMS(config)
    
    optimize(tms)
    utils.plot_features_in_2d(
        tms.model.W,
        colors=feature_importance,
        title=f"Superposition: {n_features} features represented in 2D space",
        subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability[:, 0]],
    )

    fig = plt.gcf()
    fig.savefig("5_2_superposition.png")

run_5_2_experiment()

# %%
def run_100_20_experiment():
    """ Visualize bottleneck superposition of 100 features in 20 dimensions """
    device = get_device()
    n_inst = 10
    n_features = 100
    d_hidden = 20

    feature_probability = (20 ** -torch.linspace(0, 1, n_inst)).to(device)
    feature_importance = (100 ** -torch.linspace(0, 1, n_features)).to(device)

    line(feature_importance, width=600, height=400, title="Importance of each feature (same over all instances)", labels={"y": "Feature importance", "x": "Feature"})
    line(feature_probability, width=600, height=400, title="Feature probability (varied over instances)", labels={"y": "Probability", "x": "Instance"})

    feature_importance = einops.repeat(feature_importance, 'feats -> inst feats', inst=n_inst)
    feature_probability = einops.repeat(feature_probability, 'inst -> inst feats', feats=n_features)

    config = BottleneckTMSConfig(
        d_hidden=d_hidden,
        n_inst=n_inst,
        n_features=n_features,
        feature_probability=feature_probability,
        feature_importance=feature_importance
    )
    tms = BottleneckTMS(config)
    optimize(tms)

    utils.plot_features_in_Nd(
        tms.model.W,
        height=800,
        width=1600,
        title="ReLU output model: n_features = 80, d_hidden = 20, I<sub>i</sub> = 0.9<sup>i</sup>",
        subplot_titles=[f"Feature prob = {i:.3f}" for i in feature_probability[:, 0]],
    )

    fig = plt.gcf()
    fig.savefig("100_20_superposition.png")

run_100_20_experiment()

# %%
def run_2x2_correlated_experiment():
    """ Visualize bottleneck superposition of 2 pairs of 2 correlated features in 2 dimensions"""

    device = get_device()
    n_inst = 5
    n_features = 4
    d_hidden = 2

    feature_probability = (400 ** -torch.linspace(0.5, 1, n_inst)).to(device)
    feature_importance = (torch.ones(n_features, dtype=torch.float)).to(device)

    feature_importance = einops.repeat(feature_importance, 'feats -> inst feats', inst=n_inst)
    feature_probability = einops.repeat(feature_probability, 'inst -> inst feats', feats=n_features)

    config = BottleneckTMSConfig(
        d_hidden=d_hidden,
        n_inst=n_inst,
        n_features=n_features,
        feature_probability=feature_probability,
        feature_importance=feature_importance,
        data_gen_cls=CorrelatedFeatureGenerator
    )
    tms = BottleneckTMS(config)

    # Sanity check the data generator
    batch = tms.data_gen.generate_batch(batch_size=1)
    utils.plot_correlated_features(
        batch, title="Correlated feature pairs: should always co-occur"
    )


    optimize(tms)
    utils.plot_features_in_2d(
        tms.model.W,
        colors=["blue"] * 2 + ["limegreen"] * 2,
        title="Correlated feature sets are represented in local orthogonal bases",
        subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability[:, 0]],
    )

    fig = plt.gcf()
    fig.savefig("2x2_correlated.png")

run_2x2_correlated_experiment()

# %%
def run_2x2_anticorrelated_experiment():
    """ Visualize bottleneck superposition of 2 pairs of 2 anticorrelated features in 2 dimensions"""

    device = get_device()
    n_inst = 5
    n_features = 4
    d_hidden = 2

    feature_probability = (10 ** -torch.linspace(0.5, 1, n_inst)).to(device)
    feature_importance = (torch.ones(n_features, dtype=torch.float)).to(device)

    feature_importance = einops.repeat(feature_importance, 'feats -> inst feats', inst=n_inst)
    feature_probability = einops.repeat(feature_probability, 'inst -> inst feats', feats=n_features)

    config = BottleneckTMSConfig(
        d_hidden=d_hidden,
        n_inst=n_inst,
        n_features=n_features,
        feature_probability=feature_probability,
        feature_importance=feature_importance,
        data_gen_cls=AnticorrelatedFeatureGenerator
    )
    tms = BottleneckTMS(config)
    
    # Sanity check the data generator
    batch = tms.data_gen.generate_batch(batch_size=1)
    utils.plot_correlated_features(
        batch, title="Anti-correlated feature pairs: should never co-occur"
    )

    optimize(tms)
    utils.plot_features_in_2d(
        tms.model.W,
        colors=["blue"] * 2 + ["limegreen"] * 2,
        title="Anticorrelated feature sets are represented in antipodal pairs",
        subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability[:, 0]],
    )

    fig = plt.gcf()
    fig.savefig("2x2_anticorrelated.png")

run_2x2_anticorrelated_experiment()

