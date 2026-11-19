"""``calc_cond_uncond_batch`` splits a batch whose rows have different
conditioning shapes into several model calls. Each call tells model wrappers
which image rows it holds in ``transformer_options["batch_indices"]`` (None:
all of them); per-image state (modulation, NAG, the artist mixer, per-row AND
weights) selects its rows by it, so it must match the rows of every chunk in
the call."""

import pytest
import torch


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


class _Model:
    def __init__(self):
        self.calls = []

    def memory_required(self, shape):
        return 0

    def apply_model(self, x, timestep, **c):
        options = c["transformer_options"]
        self.calls.append((x.shape[0], options["batch_indices"], list(options["cond_or_uncond"]), timestep.shape[0]))
        return x


def test_each_call_of_a_split_batch_names_the_image_rows_of_all_its_chunks(webui):
    from backend.sampling.condition import compile_conditions
    from backend.sampling.sampling_function import calc_cond_uncond_batch

    # image 0 has a 77-token prompt, images 1 and 2 a 154-token one; the negative is shared
    cond = compile_conditions([torch.randn(77, 8), torch.randn(154, 8), torch.randn(154, 8)])
    uncond = compile_conditions(torch.randn(3, 77, 8))
    model = _Model()

    calc_cond_uncond_batch(model, cond, uncond, torch.randn(3, 4, 8, 8), torch.ones(3), {})

    assert sorted((indices or [-1]) for _, indices, _, _ in model.calls) == [[-1], [0], [1, 2]]
    for rows, indices, chunks, timesteps in model.calls:
        images = 3 if indices is None else len(indices)
        assert rows == timesteps == images * len(chunks)


def test_a_cond_and_uncond_chunk_share_a_call_only_for_the_same_images(webui):
    from backend.sampling.condition import compile_conditions
    from backend.sampling.sampling_function import calc_cond_uncond_batch

    # negatives split like the positives: 0 alone, 1 and 2 together, but image 1's negative is longer
    cond = compile_conditions([torch.randn(77, 8), torch.randn(154, 8), torch.randn(154, 8)])
    uncond = compile_conditions([torch.randn(77, 8), torch.randn(231, 8), torch.randn(154, 8)])
    model = _Model()

    calc_cond_uncond_batch(model, cond, uncond, torch.randn(3, 4, 8, 8), torch.ones(3), {})

    merged = [(indices, sorted(chunks)) for _, indices, chunks, _ in model.calls if len(chunks) > 1]
    assert merged == [([0], [0, 1])]  # image 0: its cond and uncond in one call
    for rows, indices, chunks, timesteps in model.calls:
        assert rows == timesteps == len(indices) * len(chunks)
