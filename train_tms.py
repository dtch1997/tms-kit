""" Script to train a Toy Model of Superposition """

import torch 

from tms.tms import BottleneckTMS, BottleneckTMSConfig
from tms.optimize import optimize
from tms.utils.device import get_device

def make_bottleneck_tms():

    device = get_device()
    n_inst = 10
    n_features = 5
    d_hidden = 2

    feature_probability = torch.rand(n_inst).to(device)
    feature_importance = (0.9 ** torch.arange(n_features)).to(device)

    config = BottleneckTMSConfig(
        d_hidden=d_hidden,
        n_inst=n_inst,
        n_features=n_features,
        feature_probability=feature_probability,
        feature_importance=feature_importance
    )

    tms = BottleneckTMS(config)
    return tms

def main():
    tms = make_bottleneck_tms()
    optimize(tms)

if __name__ == "__main__":
    main()