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

    def memory_required(self, shape) -> int:
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


def test_a_cond_and_uncond_chunk_share_a_call_only_for_the_same_images(webui, monkeypatch):
    from backend.sampling import sampling_function
    from backend.sampling.condition import compile_conditions
    from backend.sampling.sampling_function import calc_cond_uncond_batch

    monkeypatch.setattr(sampling_function, "merges_by_free_memory", lambda device: True)  # as on CUDA

    # negatives split like the positives: 0 alone, 1 and 2 together, but image 1's negative is longer
    cond = compile_conditions([torch.randn(77, 8), torch.randn(154, 8), torch.randn(154, 8)])
    uncond = compile_conditions([torch.randn(77, 8), torch.randn(231, 8), torch.randn(154, 8)])
    model = _Model()

    calc_cond_uncond_batch(model, cond, uncond, torch.randn(3, 4, 8, 8), torch.ones(3), {})

    merged = [(indices, sorted(chunks)) for _, indices, chunks, _ in model.calls if len(chunks) > 1]
    assert merged == [([0], [0, 1])]  # image 0: its cond and uncond in one call
    for rows, indices, chunks, timesteps in model.calls:
        assert rows == timesteps == len(indices) * len(chunks)


@pytest.mark.parametrize("device_merges", [False, True], ids=["mps-cpu", "cuda"])
def test_merging_chunks_follows_free_memory_only_where_free_memory_is_the_gpus(webui, monkeypatch, device_merges):
    """On MPS and CPU ``get_free_memory`` is the system's available RAM: merging the cond and
    uncond into one call when it was high made the same request a batch-2 or two batch-1 calls
    from run to run, and the image changed with it. There the chunks always run one by one; on a
    GPU with its own memory upstream's rule stays."""
    from backend import memory_management
    from backend.sampling import sampling_function
    from backend.sampling.condition import compile_conditions
    from backend.sampling.sampling_function import calc_cond_uncond_batch

    if device_merges:
        monkeypatch.setattr(sampling_function, "merges_by_free_memory", lambda device: True)
    layouts = []
    for free in (1 << 40, 1):
        monkeypatch.setattr(memory_management, "get_free_memory", lambda *args, _free=free, **kwargs: _free)

        class Sized(_Model):
            def memory_required(self, shape):
                return 1 << 20

        model = Sized()
        calc_cond_uncond_batch(model, compile_conditions(torch.randn(1, 77, 8)), compile_conditions(torch.randn(1, 77, 8)), torch.randn(1, 4, 8, 8), torch.ones(1), {})
        layouts.append(sorted(sorted(chunks) for _, _, chunks, _ in model.calls))
    if device_merges:
        assert layouts == [[[0, 1]], [[0], [1]]]  # merged only while the memory allows it
    else:
        assert layouts == [[[0], [1]], [[0], [1]]]  # the same calls whatever the free memory
