""" Script to train a Toy Model of Superposition """
# %%
import torch 
import einops

from tms.tms import BottleneckTMS, BottleneckTMSConfig
from tms.optimize import optimize
from tms.utils.device import get_device
from tms.utils import utils

def bottleneck_tms_experiment():

    device = get_device()
    n_inst = 10
    n_features = 5
    d_hidden = 2

    feature_probability = (50 ** -torch.linspace(0, 1, n_inst)).to(device)
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
        subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
    )

if __name__ == "__main__":
    bottleneck_tms_experiment()