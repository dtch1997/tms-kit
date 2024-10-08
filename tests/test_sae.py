import torch as t

from tms_kit.utils.device import get_device
from tms_kit.sae import VanillaSAE


def test_sae_W_dec_normalized():
    device = get_device()
    n_inst = 8
    d_in = 2
    d_sae = 5
    sae = VanillaSAE(n_inst, d_in, d_sae, device=device)

    W_dec = sae.W_dec
    W_dec_normalized = sae.W_dec_normalized
    t.testing.assert_close(W_dec / W_dec.norm(dim=-1, keepdim=True), W_dec_normalized)

    # Test dividing by zero
    sae.W_dec.data[:] = 0.0
    W_dec_normalized = sae.W_dec_normalized
    assert (
        W_dec_normalized.pow(2).sum() < 1e-6
    ), "Failed: did you forget to add epsilon to the denominator?"


@t.no_grad()
def test_resample_simple():
    window = 5

    device = get_device()
    n_inst = 8
    d_in = 2
    d_sae = 5
    sae = VanillaSAE(n_inst, d_in, d_sae, device=device)
    sae.b_enc.data = t.randn_like(sae.b_enc.data)

    # Get the weights (we rearrange W_enc to be the same shape as W_dec, for easier testing)
    old_W_dec = sae.W_dec.detach().clone()
    old_W_enc = sae.W_enc.detach().clone().transpose(-1, -2)
    old_b_enc = sae.b_enc.detach().clone()

    # Crete 'fract_active_in_window' which is zero at all timesteps with prob 0.5
    frac_active_in_window = t.rand((window, sae.n_inst, sae.d_sae))
    features_are_dead = frac_active_in_window[0] < 0.5
    frac_active_in_window[:, features_are_dead] = 0.0

    # Resample latents, and get new weight values (check we have correct return type)
    sae.resample_simple(frac_active_in_window, 0.5)
    new_W_dec = sae.W_dec.detach().clone()
    new_W_enc = sae.W_enc.detach().clone().transpose(-1, -2)
    new_b_enc = sae.b_enc.detach().clone()

    # Check that b_enc match where the latents aren't dead, and b_enc is zero where they are
    assert (new_b_enc[features_are_dead].abs() < 1e-8).all()
    t.testing.assert_close(new_b_enc[~features_are_dead], old_b_enc[~features_are_dead])
    t.testing.assert_close(
        new_b_enc[features_are_dead], t.zeros_like(new_b_enc[features_are_dead])
    )

    # Check that W_dec is correct:
    # (1) They should match where the latents aren't dead
    # (2) They should generally not match where the latents are dead (I've tested with >0.5 not ==1 to be on the safe side)
    # (3) Resampled neuron weights should be scaled
    t.testing.assert_close(
        new_W_dec[~features_are_dead],
        old_W_dec[~features_are_dead],
        msg="W_dec weights incorrectly changed where latents are alive",
    )
    assert (
        (new_W_dec[features_are_dead] - old_W_dec[features_are_dead]).abs() > 1e-6
    ).float().mean() > 0.5, "W_dec weights not changed where latents are dead"
    t.testing.assert_close(
        new_W_dec[features_are_dead].norm(dim=-1),
        t.ones_like(new_W_dec[features_are_dead].norm(dim=-1)),
        msg="W_dec failed normalization test",
    )

    # Same checks for W_enc, but we can replace (2) and (3) with checking new features match new W_dec features
    t.testing.assert_close(
        new_W_enc[~features_are_dead],
        old_W_enc[~features_are_dead],
        msg="W_enc weights incorrectly changed where latents are alive",
    )
    t.testing.assert_close(
        new_W_dec[features_are_dead],
        new_W_enc[features_are_dead]
        / new_W_enc[features_are_dead].norm(dim=-1, keepdim=True),
        msg="Resampled normalized W_enc weights don't match resampled W_dec weights",
    )

    # Finally, do this again when there are no dead latents, and check it doesn't break
    frac_active_in_window = t.ones((window, sae.n_inst, sae.d_sae))
    sae.resample_simple(frac_active_in_window, 1.0)

    print("All tests in `test_resample_simple` passed!")
