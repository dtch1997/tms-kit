# %%
"""Script to train and visualize a Toy Model of Superposition

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
import pathlib

from dataclasses import dataclass
from jaxtyping import Float
from typing import Type

from tms.data import (
    DataGenerator,
    IIDFeatureGenerator,
    CorrelatedFeatureGenerator,
    AnticorrelatedFeatureGenerator,
    ModelActivationsGenerator,
)
from tms.loss import ImportanceWeightedLoss
from tms.model import Model
from tms.optimize import optimize, optimize_vanilla_sae
from tms.tms import TMS
from tms.utils.device import get_device
from tms.utils import utils
from tms.utils.plotly import line
from tms.sae import VanillaSAE

MAIN = __name__ == "__main__"
DIR = pathlib.Path(__file__).parent

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
        data_gen = config.data_gen_cls(
            config.n_features, config.n_inst, config.feature_probability
        )
        loss_calc = ImportanceWeightedLoss(config.n_features, config.feature_importance)
        super().__init__(model, data_gen, loss_calc)

        self.config = config


# %%
def run_5_2_experiment():
    """Visualize bottleneck superposition of 5 features in 2 dimensions"""

    device = get_device()
    n_inst = 10
    n_features = 5
    d_hidden = 2

    feature_probability = (50 ** -torch.linspace(0, 1, n_inst)).to(device)
    feature_probability = einops.repeat(
        feature_probability, "inst -> inst feats", feats=n_features
    )
    feature_importance = (0.9 ** torch.arange(n_features)).to(device)
    feature_importance = einops.repeat(
        feature_importance, "feats -> inst feats", inst=n_inst
    )

    config = BottleneckTMSConfig(
        d_hidden=d_hidden,
        n_inst=n_inst,
        n_features=n_features,
        feature_probability=feature_probability,
        feature_importance=feature_importance,
    )
    tms = BottleneckTMS(config)

    optimize(tms)
    fig, ax = utils.plot_features_in_2d(
        tms.model.W,
        colors=feature_importance,
        title=f"Superposition: {n_features} features represented in 2D space",
        subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability[:, 0]],
    )

    utils.save_figure(fig, "5_2_superposition.png")

if MAIN:
    run_5_2_experiment()


# %%
def run_100_20_experiment():
    """Visualize bottleneck superposition of 100 features in 20 dimensions"""
    device = get_device()
    n_inst = 10
    n_features = 100
    d_hidden = 20

    feature_probability = (20 ** -torch.linspace(0, 1, n_inst)).to(device)
    feature_importance = (100 ** -torch.linspace(0, 1, n_features)).to(device)

    line(
        feature_importance,
        width=600,
        height=400,
        title="Importance of each feature (same over all instances)",
        labels={"y": "Feature importance", "x": "Feature"},
    )
    line(
        feature_probability,
        width=600,
        height=400,
        title="Feature probability (varied over instances)",
        labels={"y": "Probability", "x": "Instance"},
    )

    feature_importance = einops.repeat(
        feature_importance, "feats -> inst feats", inst=n_inst
    )
    feature_probability = einops.repeat(
        feature_probability, "inst -> inst feats", feats=n_features
    )

    config = BottleneckTMSConfig(
        d_hidden=d_hidden,
        n_inst=n_inst,
        n_features=n_features,
        feature_probability=feature_probability,
        feature_importance=feature_importance,
    )
    tms = BottleneckTMS(config)
    optimize(tms)

    fig = utils.plot_features_in_Nd(
        tms.model.W,
        height=800,
        width=1600,
        title="ReLU output model: n_features = 80, d_hidden = 20, I<sub>i</sub> = 0.9<sup>i</sup>",
        subplot_titles=[f"Feature prob = {i:.3f}" for i in feature_probability[:, 0]],
    )

    utils.save_figure(fig, "100_20_superposition.png")

if MAIN:
    run_100_20_experiment()


# %%
def run_2x2_correlated_experiment():
    """Visualize bottleneck superposition of 2 pairs of 2 correlated features in 2 dimensions"""

    device = get_device()
    n_inst = 5
    n_features = 4
    d_hidden = 2

    feature_probability = (400 ** -torch.linspace(0.5, 1, n_inst)).to(device)
    feature_importance = (torch.ones(n_features, dtype=torch.float)).to(device)

    feature_importance = einops.repeat(
        feature_importance, "feats -> inst feats", inst=n_inst
    )
    feature_probability = einops.repeat(
        feature_probability, "inst -> inst feats", feats=n_features
    )

    config = BottleneckTMSConfig(
        d_hidden=d_hidden,
        n_inst=n_inst,
        n_features=n_features,
        feature_probability=feature_probability,
        feature_importance=feature_importance,
        data_gen_cls=CorrelatedFeatureGenerator,
    )
    tms = BottleneckTMS(config)

    # Sanity check the data generator
    batch = tms.data_gen.generate_batch(batch_size=1)
    utils.plot_correlated_features(
        batch, title="Correlated feature pairs: should always co-occur"
    )

    optimize(tms)
    fig, _ = utils.plot_features_in_2d(
        tms.model.W,
        colors=["blue"] * 2 + ["limegreen"] * 2,
        title="Correlated feature sets are represented in local orthogonal bases",
        subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability[:, 0]],
    )

    utils.save_figure(fig, "2x2_correlated.png")

if MAIN:
    run_2x2_correlated_experiment()


# %%
def run_2x2_anticorrelated_experiment():
    """Visualize bottleneck superposition of 2 pairs of 2 anticorrelated features in 2 dimensions"""

    device = get_device()
    n_inst = 5
    n_features = 4
    d_hidden = 2

    feature_probability = (10 ** -torch.linspace(0.5, 1, n_inst)).to(device)
    feature_importance = (torch.ones(n_features, dtype=torch.float)).to(device)

    feature_importance = einops.repeat(
        feature_importance, "feats -> inst feats", inst=n_inst
    )
    feature_probability = einops.repeat(
        feature_probability, "inst -> inst feats", feats=n_features
    )

    config = BottleneckTMSConfig(
        d_hidden=d_hidden,
        n_inst=n_inst,
        n_features=n_features,
        feature_probability=feature_probability,
        feature_importance=feature_importance,
        data_gen_cls=AnticorrelatedFeatureGenerator,
    )
    tms = BottleneckTMS(config)

    # Sanity check the data generator
    batch = tms.data_gen.generate_batch(batch_size=1)
    utils.plot_correlated_features(
        batch, title="Anti-correlated feature pairs: should never co-occur"
    )

    optimize(tms)
    fig, _ = utils.plot_features_in_2d(
        tms.model.W,
        colors=["blue"] * 2 + ["limegreen"] * 2,
        title="Anticorrelated feature sets are represented in antipodal pairs",
        subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability[:, 0]],
    )

    utils.save_figure(fig, "2x2_anticorrelated.png")

if MAIN:
    run_2x2_anticorrelated_experiment()
# %%

def run_tms_sae_no_resampling():
    """ Train an SAE on TMS without resampling """
    device = get_device()
    d_hidden = d_in = 2
    n_features = d_sae = 5
    n_inst = 8

    feature_probability = (0.01 * torch.ones(n_inst)).to(device)
    feature_probability = einops.repeat(
        feature_probability, "inst -> inst feats", feats=n_features
    )
    feature_importance = (0.9 ** torch.arange(n_features)).to(device)
    feature_importance = einops.repeat(
        feature_importance, "feats -> inst feats", inst=n_inst
    )

    # Train the TMS
    config = BottleneckTMSConfig(
        d_hidden=d_hidden,
        n_inst=n_inst,
        n_features=n_features,
        feature_probability=feature_probability,
        feature_importance=feature_importance,
    )
    tms = BottleneckTMS(config)
    optimize(tms, steps = 1000)

    # Train the SAE
    sae = VanillaSAE(n_inst, d_in, d_sae, device=device)
    model_act_gen = ModelActivationsGenerator(tms.model, tms.data_gen)

    data_log = optimize_vanilla_sae(sae, model_act_gen, steps=1000)

    utils.frac_active_line_plot(
        frac_active=torch.stack(data_log["frac_active"]),
        title="Probability of sae features being active during training",
        avg_window=10,
    )

    utils.animate_features_in_2d(
        {
            "Encoder weights": torch.stack(data_log["W_enc"]),
            "Decoder weights": torch.stack(data_log["W_dec"]).transpose(-1, -2),
        },
        steps=data_log["steps"],
        filename=str((DIR / "sae_latent_training_history_no_resample.html").absolute()),
        title="SAE on toy model",
    )

if MAIN:
    # NOTE: Not rendering for some reason... 
    run_tms_sae_no_resampling()
# %%
