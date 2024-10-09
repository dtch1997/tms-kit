Basic Usage
===========

This section provides a minimal example of training a toy model of superposition using the :code:`tms_kit` library. 

Minimal Example
^^^^^^^^^^^^^^^

Setup
''''''''''''''''

First, we'll import the dependencies. 

.. code-block:: python

    import torch

    from tms_kit.loss import ImportanceWeightedLoss
    from tms_kit.model import Model
    from tms_kit.data import IIDFeatureGenerator
    from tms_kit.optimize import optimize
    from tms_kit.tms import TMS
    from tms_kit.utils import utils
    from tms_kit.utils.device import set_device

Defining hyperparameters
''''''''''''''''''''''''''''

Next, we'll define the important properties.

.. code-block:: python

    # Define the configuration 
    set_device('cpu')
    n_inst = 10
    n_features = 5
    d_hidden = 2
    feature_probability = 0.01 * torch.ones(n_inst, n_features)
    feature_importance = 1.0 * torch.ones(n_inst, n_features)

The hyperparameters are as follows:
    * :code:`n_inst` refers to the number of separate instances of toy models that we train. Our code facilitates training multiple instances in parallel to speed up experimentation.
    * :code:`n_features` refers to the number of features in the input data. This is expected to be constant across all instances.
    * :code:`d_hidden` refers to the dimensionality of the hidden layer in the toy model. Again, this is expected to be constant across all instances.
    * :code:`feature_probability` refers to the probability of each feature being present in the input data. Here, we're using a uniform probability of 0.01 for each feature.
    * :code:`feature_importance` refers to the importance of each feature in the loss function. Here, we're using a uniform importance of 1.0 for each feature. 


Defining the TMS experiment components
''''''''''''''''''''''''''''''''''''''''

Next, we'll define a TMS subclass which establishes the overall settings for the TMS experiment. 

.. code-block:: python

    # Define a TMS subclass with all the necessary components
    class BottleneckTMS(TMS):
        def __init__(self):
            self.model = Model(n_features = n_features, n_inst = n_inst, d_hidden = d_hidden)
            self.loss_calc = ImportanceWeightedLoss(n_features = n_features, feature_importance = feature_importance)
            self.data_gen = IIDFeatureGenerator(n_features = n_features, n_inst = n_inst, feature_probability = feature_probability)

Each :code:`TMS` instance bundles three components: 
    * A :code:`Model` that represents the toy model of superposition being trained
    * A :code:`DataGenerator` that generates batches of data for training
    * A :code:`LossCalculator` that computes the loss for the model 

Here, we're using pre-defined implementations for the model, data generator, and loss calculator. These can be replaced with different implementations to easily change different aspects of the experiment setting. 

Training the TMS
''''''''''''''''''

Finally, training the TMS is as simple as creating an instance of the subclass and calling
the :code:`optimize` function.

.. code-block:: python

    # Train a TMS
    tms = BottleneckTMS()
    optimize(tms)


Inspecting the TMS features
''''''''''''''''''''''''''''

Lastly, we can inspect the results of the training by plotting the features learned by the model in 2D space.

.. code-block:: python

    # Inspect a TMS
    fig, ax = utils.plot_features_in_2d(
        tms.model.W,
        colors=feature_importance,
        title=f"Superposition: {n_features} features represented in 2D space",
        subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability[:, 0]],
    )
    utils.save_figure(fig, "5_2_superposition.png")

The resulting plot is as follows: 

.. image:: _static/5_2_superposition.png
    :alt: Superposition: 5 features represented in 2D space

The full code for this example can be found in the :code:`experiments/demo` directory of the repository, available `here <https://github.com/dtch1997/tms-kit/blob/main/experiments/demo/run.py>`_.