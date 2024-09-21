"""Training code for a TMS"""

import torch
import numpy as np

from typing import Callable
from tqdm import tqdm

from tms.tms import TMS


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
