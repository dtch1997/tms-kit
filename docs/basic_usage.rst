Basic usage
===========

This section provides a brief overview of how to reproduce toy models of superposition experiments. 

Quickstart
'''''''''''''''''''''
Here's a minimal example of how to train a toy model of superposition using the :code:`tms_kit` library:

.. code-block:: python

    import torch

    from tms_kit.loss import ImportanceWeightedLoss
    from tms_kit.model import Model
    from tms_kit.data import IIDFeatureGenerator
    from tms_kit.optimize import optimize
    from tms_kit.tms import TMS
    from tms_kit.utils import utils
    from tms_kit.utils.device import set_device

    # Define the configuration 
    set_device('cpu')
    n_inst = 10
    n_features = 5
    d_hidden = 2
    feature_probability = 0.01 * torch.ones(n_inst, n_features)
    feature_importance = 1.0 * torch.ones(n_inst, n_features)

    # Define a TMS subclass with all the necessary components
    class BottleneckTMS(TMS):
        def __init__(self):
            self.model = Model(n_features = n_features, n_inst = n_inst, d_hidden = d_hidden)
            self.loss_calc = ImportanceWeightedLoss(n_features = n_features, feature_importance = feature_importance)
            self.data_gen = IIDFeatureGenerator(n_features = n_features, n_inst = n_inst, feature_probability = feature_probability)

    # Train a TMS
    tms = BottleneckTMS()
    optimize(tms)

    # Inspect a TMS
    fig, ax = utils.plot_features_in_2d(
        tms.model.W,
        colors=feature_importance,
        title=f"Superposition: {n_features} features represented in 2D space",
        subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability[:, 0]],
    )
    utils.save_figure(fig, "5_2_superposition.png")


The TMS Class
'''''''''''''''''''''

The :code:`TMS` class is a lightweight abstraction that bundles three components: 
    * A :code:`Model` that represents the toy model of superposition being trained
    * A :code:`DataGenerator` that generates batches of data for training
    * A :code:`LossCalculator` that computes the loss for the model 

The :code:`TMS` class is designed to be subclassed with the necessary components defined in the subclass's :code:`__init__` method. This design allows for easy experimentation with different model architectures, data generation strategies, and loss functions.


Visualization Utilities
'''''''''''''''''''''

In addition to the :code:`TMS` class, the :code:`tms_kit` library provides a set of utilities for visualizing the results of training a toy model of superposition. The :code:`utils` module contains functions for plotting the features learned by the model in 2D space, as well as saving the resulting plots to disk.