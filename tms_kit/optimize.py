"""Training code for a TMS"""

import torch
import numpy as np

from typing import Callable
from typing import Literal
from tqdm import tqdm

from tms_kit.tms import TMS
from tms_kit.sae import VanillaSAE
from tms_kit.data import ModelActivationsGenerator


def linear_lr(step, steps):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


def optimize(
    tms: TMS,
    *,
    disable_tqdm: bool = False,
    batch_size: int = 1024,
    steps: int = 10_000,
    log_freq: int = 50,
    lr: float = 1e-3,
    lr_scale: Callable[[int, int], float] = constant_lr,
):
    """Optimizes the TMS using the given hyperparameters."""
    optimizer = torch.optim.Adam(list(tms.model.parameters()), lr=lr)

    progress_bar = tqdm(range(steps), disable=disable_tqdm)

    for step in progress_bar:
        # Update learning rate
        step_lr = lr * lr_scale(step, steps)
        for group in optimizer.param_groups:
            group["lr"] = step_lr

        # Optimize
        optimizer.zero_grad()
        batch = tms.data_gen.generate_batch(batch_size)
        out = tms.model(batch)
        loss = tms.loss_calc.calculate_loss(out, batch)
        loss.backward()
        optimizer.step()

        # Display progress bar
        if step % log_freq == 0 or (step + 1 == steps):
            progress_bar.set_postfix(loss=loss.item() / tms.model.n_inst, lr=step_lr)

    return None


def optimize_vanilla_sae(
    sae: VanillaSAE,
    data_gen: ModelActivationsGenerator,
    *,
    l1_coeff: float = 1,
    batch_size: int = 1024,
    steps: int = 10_000,
    log_freq: int = 50,
    lr: float = 1e-3,
    lr_scale: Callable[[int, int], float] = constant_lr,
    resample_method: Literal["simple", "advanced", None] = None,
    resample_freq: int = 2500,
    resample_window: int = 500,
    resample_scale: float = 0.5,
    resample_threshold: float = 1e-8,
) -> dict[str, list]:
    """
    Optimizes the autoencoder using the given hyperparameters.

    Args:
        model:              we reconstruct features from model's hidden activations
        batch_size:         size of batches we pass through model & train autoencoder on
        steps:              number of optimization steps
        log_freq:           number of optimization steps between logging
        lr:                 learning rate
        lr_scale:           learning rate scaling function
        resample_method:    method for resampling dead latents
        resample_freq:      number of optimization steps between resampling dead latents
        resample_window:    number of steps needed for us to classify a neuron as dead
        resample_scale:     scale factor for resampled neurons
        resample_threshold: threshold for classifying a neuron as dead. Only used in resample_simple

    Returns:
        data_log:               dictionary containing data we'll use for visualization
    """
    assert resample_window <= resample_freq

    optimizer = torch.optim.Adam(list(sae.parameters()), lr=lr, betas=(0.0, 0.999))
    frac_active_list = []
    progress_bar = tqdm(range(steps))

    # Create lists to store data we'll eventually be plotting
    data_log = {"steps": [], "W_enc": [], "W_dec": [], "frac_active": []}

    for step in progress_bar:
        # Resample dead latents
        if (resample_method is not None) and ((step + 1) % resample_freq == 0):
            frac_active_in_window = torch.stack(
                frac_active_list[-resample_window:], dim=0
            )
            if resample_method == "simple":
                sae.resample_simple(
                    frac_active_in_window, resample_scale, resample_threshold
                )
            elif resample_method == "advanced":
                sae.resample_advanced(frac_active_in_window, resample_scale, batch_size)

        # Update learning rate
        step_lr = lr * lr_scale(step, steps)
        for group in optimizer.param_groups:
            group["lr"] = step_lr

        # Get a batch of hidden activations from the model
        with torch.inference_mode():
            h = data_gen.generate_batch(batch_size)

        # Calculate acts
        acts = sae.encode(h)
        h_reconstructed = sae.decode(acts)
        loss, info_dict = sae.loss(h, acts, h_reconstructed, l1_coeff=l1_coeff)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Normalize decoder weights by modifying them inplace (if not using tied weights)
        if not sae.tied_weights:
            sae.W_dec.data = sae.W_dec_normalized

        # Calculate the mean sparsities over batch dim for each feature
        frac_active = (acts.abs() > 1e-8).float().mean(0)
        frac_active_list.append(frac_active)

        # Display progress bar, and append new values for plotting
        if step % log_freq == 0 or (step + 1 == steps):
            progress_bar.set_postfix(
                lr=step_lr,
                frac_active=frac_active.mean().item(),
                **{k: v.mean(0).sum().item() for k, v in info_dict.items()},  # type: ignore
            )
            data_log["W_enc"].append(sae.W_enc.detach().cpu().clone())
            data_log["W_dec"].append(sae.W_dec.detach().cpu().clone())
            data_log["frac_active"].append(frac_active.detach().cpu().clone())
            data_log["steps"].append(step)

    return data_log
