from jaxtyping import Float
from torch import Tensor
from abc import ABC, abstractmethod
from tms_kit.utils.device import get_device

import einops
import torch.nn as nn
import torch
import torch.nn.functional as F


class SAE(nn.Module, ABC):
    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, z):
        pass

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


class VanillaSAE(SAE):
    n_inst: int
    d_in: int
    d_sae: int

    W_enc: Float[Tensor, "inst d_in d_sae"]
    _W_dec: Float[Tensor, "inst d_sae d_in"] | None
    b_enc: Float[Tensor, "inst d_sae"]
    b_dec: Float[Tensor, "inst d_in"]

    def __init__(
        self,
        n_inst: int,
        d_in: int,
        d_sae: int,
        weight_normalize_eps: float = 1e-8,
        tied_weights: bool = False,
        device=None,
    ):
        if device is None:
            device = get_device()
        super(SAE, self).__init__()
        self.n_inst = n_inst
        self.d_in = d_in
        self.d_sae = d_sae
        self.weight_normalize_eps = weight_normalize_eps
        self.tied_weights = tied_weights

        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty((n_inst, d_in, d_sae)))
        )
        self._W_dec = (
            None
            if tied_weights
            else nn.Parameter(
                nn.init.kaiming_uniform_(torch.empty((n_inst, d_sae, d_in)))
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(n_inst, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(n_inst, d_in))

        self.to(device)

    @property
    def W_dec(self) -> Float[Tensor, "inst d_sae d_in"]:
        return self._W_dec if self._W_dec is not None else self.W_enc.transpose(-1, -2)

    @property
    def W_dec_normalized(self) -> Float[Tensor, "inst d_sae d_in"]:
        """Returns decoder weights, normalized over the autoencoder input dimension."""
        return self.W_dec / (
            self.W_dec.norm(dim=-1, keepdim=True) + self.weight_normalize_eps
        )

    def encode(
        self, x: Float[Tensor, "... inst d_in"]
    ) -> Float[Tensor, "... inst d_sae"]:
        x_cent = x - self.b_dec
        pre_acts = (
            einops.einsum(
                x_cent, self.W_enc, "... inst d_in, inst d_in d_sae -> ... inst d_sae"
            )
            + self.b_enc
        )
        return F.relu(pre_acts)

    def decode(
        self, z: Float[Tensor, "... inst d_sae"]
    ) -> Float[Tensor, "... inst d_in"]:
        return (
            einops.einsum(
                z,
                self.W_dec,
                "... inst d_sae, inst d_sae d_in -> ... inst d_in",
            )
            + self.b_dec
        )

    def loss(
        self,
        h: Float[Tensor, "... inst d_in"],
        sae_act: Float[Tensor, "... inst d_sae"],
        h_recon: Float[Tensor, "... inst d_in"],
        l1_coeff: float = 1e-3,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Compute loss terms
        L_reconstruction = (h_recon - h).pow(2).mean(-1)
        L_sparsity = sae_act.abs().sum(-1)
        info_dict = {
            "L_reconstruction": L_reconstruction,
            "L_sparsity": L_sparsity,
        }
        loss = (L_reconstruction + l1_coeff * L_sparsity).mean(0).sum()
        return loss, info_dict

    @torch.no_grad()
    def resample_simple(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
        threshold: float = 1e-8,
    ) -> None:
        """
        Resamples dead latents, by modifying the model's weights and biases inplace.

        Resampling method is:
            - For each dead neuron, generate a random vector of size (d_in,), and normalize these vectors
            - Set new values of W_dec and W_enc to be these normalized vectors, at each dead neuron
            - Set b_enc to be zero, at each dead neuron

        This function performs resampling over all instances at once, using batched operations.
        """
        # Get a tensor of dead latents
        dead_latents_mask = (frac_active_in_window < threshold).all(
            dim=0
        )  # [instances d_sae]
        n_dead = int(dead_latents_mask.int().sum().item())

        # Get our random replacement values of shape [n_dead d_in], and scale them
        replacement_values = torch.randn((n_dead, self.d_in), device=self.W_enc.device)
        replacement_values_normed = replacement_values / (
            replacement_values.norm(dim=-1, keepdim=True) + self.weight_normalize_eps
        )

        # Change the corresponding values in W_enc, W_dec, and b_enc
        self.W_enc.data.transpose(-1, -2)[dead_latents_mask] = (
            resample_scale * replacement_values_normed
        )
        self.W_dec.data[dead_latents_mask] = replacement_values_normed
        self.b_enc.data[dead_latents_mask] = 0.0

    @torch.no_grad()
    def resample_advanced(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
        batch_size: int,
    ) -> None:
        """
        Resamples latents that have been dead for 'dead_feature_window' steps, according to `frac_active`.

        Resampling method is:
            - Compute the L2 reconstruction loss produced from the hidden state vectors `h`
            - Randomly choose values of `h` with probability proportional to their reconstruction loss
            - Set new values of W_dec and W_enc to be these (centered and normalized) vectors, at each dead neuron
            - Set b_enc to be zero, at each dead neuron

        Returns colors and titles (useful for creating the animation: resampled neurons appear in red).
        """

        raise NotImplementedError("Advanced resampling not yet implemented")

    #     h = self.generate_batch(batch_size)
    #     l2_loss = self.forward(h)[0]["L_reconstruction"]

    #     for instance in range(self.n_inst):
    #         # Find the dead latents in this instance. If all latents are alive, continue
    #         is_dead = (frac_active_in_window[:, instance] < 1e-8).all(dim=0)
    #         dead_latents = torch.nonzero(is_dead).squeeze(-1)
    #         n_dead = dead_latents.numel()
    #         if n_dead == 0:
    #             continue  # If we have no dead features, then we don't need to resample

    #         # Compute L2 loss for each element in the batch
    #         l2_loss_instance = l2_loss[:, instance]  # [batch_size]
    #         if l2_loss_instance.max() < 1e-6:
    #             continue  # If we have zero reconstruction loss, we don't need to resample

    #         # Draw `d_sae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss
    #         distn = Categorical(
    #             probs=l2_loss_instance.pow(2) / l2_loss_instance.pow(2).sum()
    #         )
    #         replacement_indices = distn.sample((n_dead,))  # type: ignore

    #         # Index into the batch of hidden activations to get our replacement values
    #         replacement_values = (h - self.b_dec)[
    #             replacement_indices, instance
    #         ]  # [n_dead d_in]
    #         replacement_values_normalized = replacement_values / (
    #             replacement_values.norm(dim=-1, keepdim=True)
    #             + self.weight_normalize_eps
    #         )

    #         # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
    #         W_enc_norm_alive_mean = (
    #             self.W_enc[instance, :, ~is_dead].norm(dim=0).mean().item()
    #             if (~is_dead).any()
    #             else 1.0
    #         )

    #         # Lastly, set the new weights & biases (W_dec is normalized, W_enc needs specific scaling, b_enc is zero)
    #         self.W_dec.data[instance, dead_latents, :] = replacement_values_normalized
    #         self.W_enc.data[instance, :, dead_latents] = (
    #             replacement_values_normalized.T * W_enc_norm_alive_mean * resample_scale
    #         )
    #         self.b_enc.data[instance, dead_latents] = 0.0
