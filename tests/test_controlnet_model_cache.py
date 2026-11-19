"""ControlNet's built-model cache (modules_forge/supported_controlnet.py): a file replaced
under the same name is built again, a model built for other dtypes is not reused, and the
"Number of Models to Cache in Memory" setting's 0 caches nothing."""

import os

import pytest

from remi.testing import webui_state


@pytest.fixture(scope="module")
def cn():
    with webui_state():
        from modules_forge import supported_controlnet

        yield supported_controlnet


def test_the_key_follows_the_file_contents_and_the_dtypes(cn, tmp_path):
    path = tmp_path / "control_canny.safetensors"
    path.write_bytes(b"a" * 10)
    key = cn.cache_key(str(path), "torch.float16", "torch.float16")
    assert cn.cache_key(str(path), "torch.float16", "torch.float16") == key
    assert cn.cache_key(str(tmp_path / "." / "control_canny.safetensors"), "torch.float16", "torch.float16") == key

    assert cn.cache_key(str(path), "torch.float16", "torch.bfloat16") != key  # another base model's dtype

    path.write_bytes(b"b" * 12)  # re-downloaded under the same name
    os.utime(path, ns=(1, 1))
    assert cn.cache_key(str(path), "torch.float16", "torch.float16") != key


@pytest.mark.parametrize(("size", "kept"), [(0, []), (1, ["c"]), (2, ["b", "c"])])
def test_the_cache_holds_as_many_models_as_the_setting_says(cn, monkeypatch, size, kept):
    from modules import shared

    monkeypatch.setitem(shared.opts.data, "control_net_model_cache_size", size)
    cache = cn.ModelCache()
    for key in ("a", "b", "c"):
        cache.put(key, object())
    assert list(cache.cache) == kept
