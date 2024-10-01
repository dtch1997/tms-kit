# %%
import torch
import einops

from dataclasses import dataclass
from jaxtyping import Float

from tms.data import DataGenerator, ModelActivationsGenerator
from tms.loss import ImportanceWeightedLoss
from tms.model import Model
from tms.optimize import optimize, optimize_vanilla_sae
from tms.tms import TMS
from tms.utils.device import get_device
from tms.utils import utils
from tms.sae import VanillaSAE

MAIN = __name__ == "__main__"


class HierarchicalFeatureGenerator(DataGenerator):
    """Generates features where odd-indexed features can only occur if odd features occur"""

    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """
        Generates a batch of data.
        """
        batch_shape = (batch_size, self.n_inst, self.n_features)
        feat_mag = torch.rand(batch_shape, device=get_device())
        feat_seeds = torch.rand(batch_shape, device=get_device())
        feat_vals = torch.where(feat_seeds <= self.feature_probability, feat_mag, 0.0)
        # Zero out features that are odd-indexed if the previous feature is zero
        # E.g feature 3 can only occur if feature 2 is non-zero
        feat_vals[..., 1::2] *= (feat_vals[..., 0::2] > 0).float()
        return feat_vals


@dataclass
class FeatureAbsorptionTMSConfig:
    d_hidden: int
    n_inst: int
    n_features: int
    feature_probability: Float[torch.Tensor, "inst feats"]
    feature_importance: Float[torch.Tensor, "inst feats"]


class FeatureAbsorptionTMS(TMS):
    config: FeatureAbsorptionTMSConfig

    def __init__(self, config: FeatureAbsorptionTMSConfig):
        model = Model(config.n_features, config.n_inst, config.d_hidden)
        data_gen = HierarchicalFeatureGenerator(
            config.n_features, config.n_inst, config.feature_probability
        )
        loss_calc = ImportanceWeightedLoss(config.n_features, config.feature_importance)
        super().__init__(model, data_gen, loss_calc)
        self.config = config


# Helper function to plot pairwise cosine similarities

from torch import Tensor
from torchmetrics.functional import pairwise_cosine_similarity
import numpy as np
import matplotlib.pyplot as plt


def plot_W_pairwise_cos_sim_in_2d(
    W: Float[Tensor, "inst d_hidden feats"] | list[Float[Tensor, "d_hidden feats"]],
    colors: Float[Tensor, "inst feats"] | list[str] | list[list[str]] | None = None,
    title: str | None = None,
    subplot_titles: list[str] | None = None,
    n_rows: int | None = None,
):
    """
    Plot the cosine similarities of the decoder weights in 2D space.
    """
    # Convert W into a list of 2D tensors, each of shape [feats, d_hidden=2]
    if isinstance(W, Tensor):
        if W.ndim == 2:
            W = W.unsqueeze(0)
        n_instances, d_hidden, n_feats = W.shape
        n_feats_list = []
        W = W.detach().cpu()
    else:
        # Hacky case which helps us deal with double descent exercises (this is never used outside of those exercises)
        assert all(w.ndim == 2 for w in W)
        n_feats_list = [w.shape[1] for w in W]
        n_feats = max(n_feats_list)
        n_instances = len(W)
        W = [w.detach().cpu() for w in W]

    W_list: list[Tensor] = [W_instance.T for W_instance in W]

    # # Get some plot characteristics
    # limits_per_instance = (
    #     [w.abs().max() * 1.1 for w in W_list]
    #     if allow_different_limits_across_subplots
    #     else [1.5 for _ in range(n_instances)]
    # )
    # linewidth, markersize = (1, 4) if (n_feats >= 25) else (1.5, 6)

    # Maybe break onto multiple rows
    if n_rows is None:
        n_rows, n_cols = 1, n_instances
        row_col_tuples = [(0, i) for i in range(n_instances)]
    else:
        n_cols = n_instances // n_rows
        row_col_tuples = [(i // n_cols, i % n_cols) for i in range(n_instances)]

    # Convert colors into a 2D list of strings, with shape [instances, feats]
    if colors is None:
        colors_list = utils.cast_element_to_nested_list("black", (n_instances, n_feats))
    elif isinstance(colors, str):
        colors_list = utils.cast_element_to_nested_list(colors, (n_instances, n_feats))
    elif isinstance(colors, list):
        # List of strings -> same for each instance and feature
        if isinstance(colors[0], str):
            assert len(colors) == n_feats
            colors_list = [colors for _ in range(n_instances)]
        # List of lists of strings -> different across instances & features (we broadcast)
        else:
            colors_list = []
            for i, colors_for_instance in enumerate(colors):
                assert len(colors_for_instance) in (1, n_feats_list[i])
                colors_list.append(
                    colors_for_instance
                    * (n_feats_list[i] if len(colors_for_instance) == 1 else 1)
                )
    elif isinstance(colors, Tensor):
        assert colors.shape == (n_instances, n_feats)
        colors_list = [
            [utils.get_viridis(v) for v in color] for color in colors.tolist()
        ]

    # Create a figure and axes, and make sure axs is a 2D array
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    axs = np.broadcast_to(axs, (n_rows, n_cols))

    # If there are titles, add more spacing for them
    fig.subplots_adjust(
        bottom=0.2, top=(0.8 if title else 0.9), left=0.1, right=0.9, hspace=0.5
    )

    # Initialize lines and markers
    for instance_idx, ((row, col)) in enumerate(row_col_tuples):
        # Get the right axis, and set the limits
        ax = axs[row, col]
        ax.set_aspect("equal", adjustable="box")

        W_inst = W_list[instance_idx]
        cos_sims = pairwise_cosine_similarity(W_inst)
        ax.imshow(cos_sims, cmap="viridis")

        # Add titles & subtitles
        if title:
            fig.suptitle(title, fontsize=15)
        if subplot_titles:
            axs[row, col].set_title(subplot_titles[instance_idx], fontsize=12)

    # Add a colorbar
    cbar = fig.colorbar(axs[0, 0].images[0], ax=axs, orientation="horizontal")

    plt.show()
    return fig, axs


# %%

"""Demonstrate that the model can represent 2 hierarchical features in 2 dimensions"""

device = get_device()
n_inst = 10
n_pairs = 1
n_features = 2 * n_pairs
d_hidden = 2

# NOTE: The even features are the ones that can be absorbed
# So their probability ends up being lower than specified here
feature_probability = torch.tensor([0.25] * n_features).to(device)
feature_probability = einops.repeat(
    feature_probability, "feats -> inst feats", inst=n_inst
)
feature_importance = (torch.ones(n_features)).to(device)
feature_importance = einops.repeat(
    feature_importance, "feats -> inst feats", inst=n_inst
)

# First train the TMS
config = FeatureAbsorptionTMSConfig(
    d_hidden=d_hidden,
    n_inst=n_inst,
    n_features=n_features,
    feature_probability=feature_probability,
    feature_importance=feature_importance,
)
tms = FeatureAbsorptionTMS(config)

# Print some statistics from the data
data = tms.data_gen.generate_batch(1000)
print("Data marginal statistics: ")
print((data > 0).float().mean(dim=(0, 1)))
print()

optimize(tms)

# %%
# Visualize the weights learned
fig, ax = utils.plot_features_in_2d(
    tms.model.W,
    colors=["blue", "limegreen"],
    title=f"Model embeddings: {n_features} features represented in 2D space",
)
utils.save_figure(fig, "model_embeddings.png")

# %%
# Now train an SAE on the model
d_sae = 2
sae = VanillaSAE(config.n_inst, config.d_hidden, d_sae, device=device)
data_gen = ModelActivationsGenerator(tms.model, tms.data_gen)
optimize_vanilla_sae(sae, data_gen)

# %%
# Visualize the weights learned
fig, ax = utils.plot_features_in_2d(
    einops.rearrange(sae.W_dec, "inst d_sae d_hidden -> inst d_hidden d_sae"),
    colors=["blue", "limegreen"],
    title=f"SAE decoder weights: {n_features} features represented in 2D space",
)
utils.save_figure(fig, "sae_latents.png")

# Plot the cos sims of the SAE decoder weight and the true model decoder weights

# %%
W_dec_normalized_reshaped = einops.rearrange(
    sae.W_dec_normalized, "inst d_sae d_hidden -> inst d_hidden d_sae"
)

# NOTE: Model W: [inst, d_hidden, n_feats] → W[0, :, 0] is the embedding for a feature
# NOTE: SAE W_dec: [inst, d_hidden, d_sae] → W_dec[0, :, 0] is the embedding for an SAE latent


def pairwise_cosine_similarity(
    A: Float[Tensor, "N d"],
    B: Float[Tensor, "M d"],
) -> Float[Tensor, "N M"]:
    """Compute the pairwise cosine similarities between two sets of vectors."""
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)
    return A_normalized @ B_normalized.T


def get_instancewise_pairwise_cos_sim(
    W1: Float[Tensor, "inst d_hidden feats1"],
    W2: Float[Tensor, "inst d_hidden feats2"],
) -> Float[Tensor, "inst feats1 feats2"]:
    output = torch.empty((W1.shape[0], W1.shape[2], W2.shape[2]))
    for i in range(W1.shape[0]):
        output[i] = pairwise_cosine_similarity(W1[i].T, W2[i].T)
    return output


# %%
def plot_pairwise_cos_sim(
    W1: Float[Tensor, "inst d_hidden feats1"],
    W2: Float[Tensor, "inst d_hidden feats2"],
    title: str | None = None,
):
    cos_sim = get_instancewise_pairwise_cos_sim(W1, W2)
    cos_sim_np = cos_sim.detach().cpu().numpy()

    fig, ax = plt.subplots(1, n_inst, figsize=(30, 3))
    for i in range(n_inst):
        ax[i].imshow(cos_sim_np[i], cmap="RdBu", vmin=-1, vmax=1)
        ax[i].set_title(f"Instance {i}")

    if title:
        fig.suptitle(title, fontsize=15)
    # If there are titles, add more spacing for them
    fig.subplots_adjust(
        bottom=0.2, top=(0.8 if title else 0.9), left=0.1, right=0.9, hspace=0.5
    )

    # Colorbar
    fig.colorbar(ax[0].images[0], ax=ax, orientation="horizontal")
    return fig


# %%

fig = plot_pairwise_cos_sim(
    sae.W_enc, tms.model.W, title="SAE Encoder to model cosine similarities"
)
utils.save_figure(fig, "sae_enc_to_model_cos_sim.png")

# %%

fig = plot_pairwise_cos_sim(
    W_dec_normalized_reshaped,
    tms.model.W,
    title="SAE Decoder to model cosine similarities",
)
utils.save_figure(fig, "sae_dec_to_model_cos_sim.png")

# %%
fig = plot_pairwise_cos_sim(W_dec_normalized_reshaped, W_dec_normalized_reshaped)
utils.save_figure(fig, "sae_to_sae_cos_sim.png")

# %%
fig = plot_pairwise_cos_sim(tms.model.W, tms.model.W)
utils.save_figure(fig, "model_to_model_cos_sim.png")


# %%
