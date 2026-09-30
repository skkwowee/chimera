"""Fixed validation crops must not perturb stochastic training."""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from train_world_model import RoundWindows


def fixture_dataset(seed=0):
    return RoundWindows([torch.arange(90).float().reshape(30, 3)],
                        [{"winner": "ct"}], 4, 2, 8, fixed_seed=seed)


def test_fixed_crops_repeat_without_global_rng_consumption():
    before = torch.random.get_rng_state().clone()
    dataset = fixture_dataset()
    first = [dataset[i] for i in range(len(dataset))]
    for i in reversed(range(len(dataset))):
        for actual, expected in zip(dataset[i], first[i]):
            assert torch.equal(actual, expected)
    assert torch.equal(before, torch.random.get_rng_state())
    assert dataset.crop_policy() == fixture_dataset().crop_policy()
    assert dataset.crop_policy()["starts_sha256"] != fixture_dataset(3).crop_policy()["starts_sha256"]


def test_validation_loader_does_not_consume_training_or_global_rng():
    training_generator = torch.Generator().manual_seed(123)
    training_state = training_generator.get_state().clone()
    global_state = torch.random.get_rng_state().clone()
    loader = DataLoader(fixture_dataset(), batch_size=3,
                        generator=torch.Generator().manual_seed(0))
    first, second = list(loader), list(loader)
    for left, right in zip(first, second):
        for a, b in zip(left, right):
            assert torch.equal(a, b)
    assert torch.equal(training_state, training_generator.get_state())
    assert torch.equal(global_state, torch.random.get_rng_state())


def test_exact_minimum_round_has_legal_fixed_crop():
    dataset = RoundWindows([torch.arange(7).float().view(7, 1)],
                           [{"winner": "t"}], 4, 2, 1, fixed_seed=0)
    x, y, prev, _ = dataset[0]
    assert x[:, 0].tolist() == [1, 2, 3, 4]
    assert y[:, 0].tolist() == [3, 4, 5, 6]
    assert prev[:, 0].tolist() == [0, 1, 2, 3]
