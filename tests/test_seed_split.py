"""Train/test seed split: every training seed must stay below the held-out range.

This is the bug described in the README: with a fixed block of 10000 seeds per
worker, 24 workers reached seed 239999 and overlapped the evaluation seeds
starting at 100000.
"""
import pytest

from t1d_env import HELDOUT_SEED_BASE, train_seed_blocks


@pytest.mark.parametrize("n_envs", range(1, 65))
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_training_seeds_stay_below_heldout(n_envs, seed):
    # same offset as train.py
    seed_base = (seed * 137) % max(1, HELDOUT_SEED_BASE // n_envs)
    block = train_seed_blocks(seed_base, n_envs)
    assert block > 0
    highest_seed = seed_base + n_envs * block - 1
    assert highest_seed < HELDOUT_SEED_BASE


def test_old_fixed_block_would_overlap():
    # the original bug, kept as a regression example
    assert 0 + 24 * 10_000 - 1 >= HELDOUT_SEED_BASE
