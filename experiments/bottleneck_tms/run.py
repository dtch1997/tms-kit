""" Script to train and visualize a Toy Model of Superposition 

Usage: Run as Jupyter cells using 'Run Below'
"""
# %%
import torch 
import einops
import matplotlib.pyplot as plt

from tms.tms import BottleneckTMS, BottleneckTMSConfig
from tms.optimize import optimize
from tms.utils.device import get_device
from tms.utils import utils
from tms.utils.plotly import line, imshow

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

# %%
run_5_2_experiment()
# %% 
run_100_20_experiment()