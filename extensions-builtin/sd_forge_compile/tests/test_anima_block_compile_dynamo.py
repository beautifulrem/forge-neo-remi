"""Per-block compile traced by real Dynamo (CPU, a counting backend instead of
Inductor) on a tiny 12-block Anima: the unit tests mock ``torch.compile``."""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "sd_forge_anima_guidance"))

from anima_block_compile import RECOMPILE_LIMIT, AnimaBlockCompileManager, block_compile_config  # noqa: E402
from lib_anima_guidance.modulation import _block_deltas  # noqa: E402

from backend import memory_management, operations  # noqa: E402
from backend.nn.anima import Anima  # noqa: E402
from modules.forward_override import forward_overrides  # noqa: E402

_BLOCKS = 12  # more than Dynamo's recompile limit (8): one graph per block would fall back


def _guard_filter(entries):
    # As scripts/compile.py: no guards on transformer_options (they change every step)
    return [("transformer_options" not in entry.name) for entry in entries]


class _CountingBackend(CompileCounter):
    """Counts compiled frames (``frame_count``) and runs of compiled graphs (``runs``)."""

    def __init__(self):
        super().__init__()
        self.runs = 0

    def __call__(self, gm, example_inputs):
        forward = super().__call__(gm, example_inputs)

        def run(*args):
            self.runs += 1
            return forward(*args)

        return run


class _KModel:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model

    def apply_model(self, x, t, context, transformer_options=None):
        return self.diffusion_model(x, t, context, transformer_options={} if transformer_options is None else transformer_options)


def _anima(num_blocks=_BLOCKS):
    generator = torch.Generator().manual_seed(0)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        model = Anima(4, 4, 2, 1, model_channels=128, crossattn_emb_channels=32, adaln_lora_dim=16, num_blocks=num_blocks, num_heads=2)
    model.load_state_dict({key: torch.randn(value.shape, generator=generator) * 0.05 for key, value in model.state_dict().items()})
    return model.eval()


def _inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(2, 4, 1, 8, 8, generator=generator), torch.tensor([0.3, 0.7]), torch.randn(2, 5, 32, generator=generator)


class AnimaBlockCompileDynamoTests(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        # platform.mac_ver() (attention's upcast check on macOS) stats a file,
        # which Dynamo cannot trace: pin it so the graph is the same everywhere.
        mac_version = memory_management.mac_version()
        patcher = patch.object(memory_management, "mac_version", lambda: mac_version)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(torch._dynamo.reset)

        self.model = _anima()
        self.kmodel = _KModel(self.model)
        self.backend = _CountingBackend()
        self.fallbacks = []
        config = {"backend": self.backend, "dynamic": False, "fullgraph": False, "options": {"guard_filter_fn": _guard_filter}}
        AnimaBlockCompileManager.install(self.kmodel, config, on_fallback=self.fallbacks.append)

    def _both(self, seed):
        x, t, context = _inputs(seed)
        eager = self.model(x, t, context, transformer_options={})
        runs = self.backend.runs
        compiled = self.kmodel.apply_model(x, t, context, transformer_options={})
        return eager, compiled, self.backend.runs - runs

    def test_compiled_blocks_match_eager_without_recompiling(self):
        with torch.inference_mode():
            for seed in range(3):
                eager, compiled, runs = self._both(seed)
                torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-5)
                self.assertEqual(runs, _BLOCKS)  # every block ran its compiled graph
                frames = self.backend.frame_count
                if seed == 0:
                    first = frames
                self.assertEqual(frames, first)  # no recompiles on later steps
        self.assertGreater(first, 0)
        self.assertLessEqual(first, 2)  # the blocks share their graph
        self.assertEqual(self.fallbacks, [])
        self.assertTrue(all(type(block).__name__ != "OptimizedModule" for block in self.model.blocks))

    def test_modulation_pre_hooks_take_effect_in_the_compiled_blocks(self):
        """Dynamo compiles Modulation's per-call pre-hook as a frame of its own
        (one graph with the hook, none without): toggling it recompiles nothing
        after the first time."""
        delta = torch.randn(2, 1, 128 * 3, generator=torch.Generator().manual_seed(9)) * 0.5

        def modulated(seed):
            # As Modulation's model wrapper: the hooks live for the call only.
            with _block_deltas((block, delta) for block in self.model.blocks):
                return self._both(seed)

        with torch.inference_mode():
            self._both(0)
            frames = self.backend.frame_count
            eager, compiled, runs = modulated(1)
            unmodulated, compiled_off, runs_off = self._both(1)
            toggled = self.backend.frame_count
            again, compiled_again, _ = modulated(2)
            self._both(2)
        self.assertGreater((eager - unmodulated).abs().max().item(), 1e-3)  # the delta matters
        torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(compiled_off, unmodulated, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(compiled_again, again, rtol=1e-5, atol=1e-5)
        self.assertEqual((runs, runs_off), (2 * _BLOCKS, _BLOCKS))  # hook graph + block graph per block
        self.assertLessEqual(toggled, frames + 1)
        self.assertEqual(self.backend.frame_count, toggled)
        self.assertEqual(self.fallbacks, [])
        self.assertTrue(all(not block._forward_pre_hooks for block in self.model.blocks))

    def test_a_forward_override_inside_the_blocks_runs_eager(self):
        block = self.model.blocks[3]
        original = block.forward

        def shifted(*args, **kwargs):
            out = original(*args, **kwargs)  # a per-channel shift: LayerNorm would erase a constant one
            return out + torch.linspace(-0.5, 0.5, out.shape[-1])

        with torch.inference_mode():
            self._both(0)
            frames = self.backend.frame_count
            with forward_overrides([(block, shifted)]):
                eager, compiled, runs = self._both(1)
            after, compiled_after, runs_after = self._both(1)
        torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-5)
        self.assertGreater((eager - after).abs().max().item(), 1e-3)
        self.assertEqual(runs, 0)  # the whole call ran the eager blocks
        torch.testing.assert_close(compiled_after, after, rtol=1e-5, atol=1e-5)
        self.assertEqual(runs_after, _BLOCKS)
        self.assertEqual(self.backend.frame_count, frames)


# More (batch, frames, H, W) combinations than Dynamo's default recompile
# limit (8), as a session produces them: batch 1 (CFG 1 / Turbo, steps outside
# the CFG range) and two frames (a reference image) are specialized, so they
# add graphs beyond the dynamic one that covers the other sizes.
_SIZES = (
    (2, 1, 8, 8),
    (1, 1, 8, 8),
    (4, 1, 8, 8),
    (2, 1, 8, 16),
    (1, 1, 8, 16),
    (2, 2, 8, 8),
    (1, 2, 8, 8),
    (4, 1, 16, 8),
    (2, 2, 16, 16),
    (1, 1, 16, 16),
    (4, 1, 8, 24),
    (1, 2, 24, 8),
    (2, 1, 24, 24),
    (4, 2, 16, 24),
    (1, 1, 24, 16),
    (2, 1, 24, 16),
)


class AnimaBlockCompileSizesTests(unittest.TestCase):
    """New image sizes keep running compiled blocks, or report the fallback."""

    def setUp(self):
        torch._dynamo.reset()
        mac_version = memory_management.mac_version()
        patcher = patch.object(memory_management, "mac_version", lambda: mac_version)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(torch._dynamo.reset)
        self.model = _anima(num_blocks=2)
        self.kmodel = _KModel(self.model)
        self.backend = _CountingBackend()
        self.fallbacks = []

    def _install(self, **overrides):
        config = {**block_compile_config(_guard_filter), "backend": self.backend, **overrides}
        AnimaBlockCompileManager.install(self.kmodel, config, on_fallback=self.fallbacks.append)

    def _run_sizes(self):
        runs = []
        with torch.inference_mode(), torch._dynamo.config.patch(recompile_limit=8):
            for seed, (batch, frames, height, width) in enumerate(_SIZES):
                generator = torch.Generator().manual_seed(seed)
                x = torch.randn(batch, 4, frames, height, width, generator=generator)
                context = torch.randn(batch, 5, 32, generator=generator)
                before = self.backend.runs
                self.kmodel.apply_model(x, torch.rand(batch, generator=generator), context, transformer_options={})
                runs.append(self.backend.runs - before)
        return runs

    def test_every_size_runs_the_compiled_blocks(self):
        self._install()
        runs = self._run_sizes()
        self.assertTrue(all(count > 0 for count in runs), runs)
        self.assertEqual(self.fallbacks, [])
        # Specialized batch-1 / two-frame graphs go past Dynamo's default 8, within the preset's limit
        self.assertGreater(self.backend.frame_count, 8)
        self.assertLessEqual(self.backend.frame_count, RECOMPILE_LIMIT)
        # The raised limit is scoped to the compiled call
        self.assertEqual(torch._dynamo.config.recompile_limit, 8)

    def test_the_recompile_limit_is_a_reported_fallback(self):
        self._install(dynamic=False)  # a new cache entry per size
        with patch("anima_block_compile.RECOMPILE_LIMIT", 8):
            runs = self._run_sizes()
        self.assertEqual([type(error).__name__ for error in self.fallbacks], ["FailOnRecompileLimitHit"])
        self.assertEqual(AnimaBlockCompileManager.fallback_reason(self.kmodel), "FailOnRecompileLimitHit")
        self.assertEqual(runs[8:], [0] * (len(_SIZES) - 8))  # persistent eager after the fallback


class MOSSImageBlockCompileDynamoTests(unittest.TestCase):
    """Per-block compile traced by real Dynamo on MOSS-Image ``layers``:
    zero FFN graph breaks, SLG ``patches_replace`` compatibility, eager
    one-shot ``preprocess_refs`` + compiled ``ref_kv`` denoising, and dynamic
    resolutions within ``RECOMPILE_LIMIT``."""

    def setUp(self):
        from backend.args import dynamic_args
        from backend.nn.moss_image import MOSSImage as MOSSDiT

        torch._dynamo.reset()
        mac_version = memory_management.mac_version()
        patcher = patch.object(memory_management, "mac_version", lambda: mac_version)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(torch._dynamo.reset)

        for attr, val in (("ref_latents", []), ("moss_reference_method", None), ("moss_ref_kv", None)):
            p_arg = patch.object(dynamic_args, attr, val)
            p_arg.start()
            self.addCleanup(p_arg.stop)

        generator = torch.Generator().manual_seed(0)
        with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
            self.model = MOSSDiT(dim=128, ffn_dim=256, num_heads=2, num_layers=6, text_dim=32, latent_channels=16, fused_qkv=False).eval()
        self.model.load_state_dict({k: torch.randn(v.shape, generator=generator) * 0.05 for k, v in self.model.state_dict().items()})
        self.kmodel = _KModel(self.model)
        self.backend = _CountingBackend()
        self.fallbacks = []
        config = {**block_compile_config(_guard_filter), "backend": self.backend}
        AnimaBlockCompileManager.install(self.kmodel, config, on_fallback=self.fallbacks.append)

    def test_zero_graph_breaks_slg_and_ref_kv_caching(self):
        from backend.args import dynamic_args

        gen = torch.Generator().manual_seed(1)
        x = torch.randn(2, 16, 4, 4, generator=gen)
        t = torch.tensor([0.8, 0.8])
        ctx = torch.randn(2, 5, 32, generator=gen)

        # Verify Block.forward has 0 graph breaks (linear_input_act bypass works!)
        x_tok = torch.randn(2, 21, 128)
        mod = tuple(torch.randn(2, 1, 128) for _ in range(6))
        pe = self.model.rope(4, 4, 5, torch.device("cpu"))
        with torch.inference_mode():
            explained = torch._dynamo.explain(self.model.layers[0])(x_tok, mod, pe, {})
        self.assertEqual(explained.graph_break_count, 0, [r.reason for r in explained.break_reasons])
        torch._dynamo.reset()

        with torch.inference_mode():
            eager = self.model(x, t, ctx, transformer_options={})
            compiled = self.kmodel.apply_model(x, t, ctx, transformer_options={})
            frames_after_first = self.backend.frame_count
            compiled_step2 = self.kmodel.apply_model(x, torch.tensor([0.3, 0.3]), ctx, transformer_options={})
            self.assertEqual(self.backend.frame_count, frames_after_first)
            torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-5)

            # SLG skips layer 3 while the other 5 layers stay compiled
            slg_opts = {"patches_replace": {"dit": {("single_block", 3): lambda a, _e: a}}}
            eager_slg = self.model(x, t, ctx, transformer_options=slg_opts)
            before_slg = self.backend.runs
            comp_slg = self.kmodel.apply_model(x, t, ctx, transformer_options=slg_opts)
            self.assertEqual(self.backend.runs - before_slg, 5)
            torch.testing.assert_close(comp_slg, eager_slg, rtol=1e-5, atol=1e-5)

            # Cached ref_kv: preprocess_refs stays eager, denoising layers compile and match eager
            ref = torch.randn(1, 16, 2, 2, generator=gen)
            dynamic_args.ref_latents = [ref]
            dynamic_args.moss_reference_method = "index_timestep_zero_text_start"
            dynamic_args.moss_ref_kv = None
            eager_ref = self.model(x, t, ctx, transformer_options={})
            dynamic_args.moss_ref_kv = None
            before_ref = self.backend.runs
            comp_ref = self.kmodel.apply_model(x, t, ctx, transformer_options={})
            self.assertEqual(self.backend.runs - before_ref, 6)  # 6 denoising layers only; preprocess_refs bypassed compile
            torch.testing.assert_close(comp_ref, eager_ref, rtol=1e-5, atol=1e-5)

            # Multiple resolutions & batch=1 (CFG=1) stay inside RECOMPILE_LIMIT
            dynamic_args.ref_latents = []
            dynamic_args.moss_reference_method = None
            dynamic_args.moss_ref_kv = None
            for batch, h, w in ((1, 4, 4), (2, 6, 4), (1, 6, 4), (2, 4, 6), (2, 8, 6)):
                self.kmodel.apply_model(torch.randn(batch, 16, h, w), torch.full((batch,), 0.5), torch.randn(batch, 5, 32), transformer_options={})

        self.assertEqual(self.fallbacks, [])
        self.assertLessEqual(self.backend.frame_count, RECOMPILE_LIMIT)
        self.assertTrue(all(type(layer).__name__ != "OptimizedModule" for layer in self.model.layers))
        del compiled_step2


class Krea2BlockCompileDynamoTests(unittest.TestCase):
    """Per-block compile traced by real Dynamo on Krea 2 ``blocks``:
    zero graph breaks inside ``SingleStreamBlock``, SLG ``patches_replace``
    compatibility, Edit-mode reference latents, and dynamic shapes within
    ``RECOMPILE_LIMIT``."""

    def setUp(self):
        from backend import attention
        from backend.args import dynamic_args
        from backend.nn import krea as krea_mod

        torch._dynamo.reset()
        attn_patch = patch.object(krea_mod, "attention_function", attention.attention_pytorch)
        attn_patch.start()
        self.addCleanup(attn_patch.stop)
        ref_patch = patch.object(dynamic_args, "ref_latents", [])
        ref_patch.start()
        self.addCleanup(ref_patch.stop)
        self.addCleanup(torch._dynamo.reset)

        generator = torch.Generator().manual_seed(0)
        with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
            self.model = krea_mod.SingleStreamDiT(
                features=256,
                tdim=64,
                txtdim=128,
                heads=2,
                kvheads=2,
                multiplier=2,
                layers=4,
                patch=2,
                channels=16,
                txtlayers=12,
                txtheads=2,
                txtkvheads=2,
            ).eval()
        self.model.load_state_dict({k: torch.randn(v.shape, generator=generator) * 0.05 for k, v in self.model.state_dict().items()})
        self.kmodel = _KModel(self.model)
        self.backend = _CountingBackend()
        self.fallbacks = []
        config = {**block_compile_config(_guard_filter), "backend": self.backend}
        AnimaBlockCompileManager.install(self.kmodel, config, on_fallback=self.fallbacks.append)

    def test_zero_graph_breaks_slg_edit_and_dynamic_shapes(self):
        from backend.args import dynamic_args

        gen = torch.Generator().manual_seed(2)
        x = torch.randn(2, 16, 1, 4, 4, generator=gen)
        t = torch.tensor([0.8, 0.8])
        ctx = torch.randn(2, 5, 12, 128, generator=gen)

        # Verify SingleStreamBlock.forward traces with 0 graph breaks
        x_tok = torch.randn(2, 9, 256)
        vec = torch.randn(2, 1, 256 * 6)
        pe = self.model.pe_embedder(torch.zeros(2, 9, 3))
        with torch.inference_mode():
            explained = torch._dynamo.explain(self.model.blocks[0])(x_tok, vec, pe, None, transformer_options={})
        self.assertEqual(explained.graph_break_count, 0, [r.reason for r in explained.break_reasons])
        torch._dynamo.reset()

        with torch.inference_mode():
            eager = self.model(x, t, ctx, transformer_options={})
            compiled = self.kmodel.apply_model(x, t, ctx, transformer_options={})
            frames_after_first = self.backend.frame_count
            compiled_step2 = self.kmodel.apply_model(x, torch.tensor([0.3, 0.3]), ctx, transformer_options={})
            self.assertEqual(self.backend.frame_count, frames_after_first)
            torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-5)

            # SLG skips block 2 while the other 3 blocks stay compiled
            slg_opts = {"patches_replace": {"dit": {("single_block", 2): lambda a, _e: a}}}
            eager_slg = self.model(x, t, ctx, transformer_options=slg_opts)
            before_slg = self.backend.runs
            comp_slg = self.kmodel.apply_model(x, t, ctx, transformer_options=slg_opts)
            self.assertEqual(self.backend.runs - before_slg, 3)
            torch.testing.assert_close(comp_slg, eager_slg, rtol=1e-5, atol=1e-5)

            # Edit mode with reference latents stays compiled and matches eager
            dynamic_args.ref_latents = [torch.randn(1, 16, 4, 4, generator=gen)]
            eager_edit = self.model(x, t, ctx, transformer_options={})
            before_edit = self.backend.runs
            comp_edit = self.kmodel.apply_model(x, t, ctx, transformer_options={})
            self.assertEqual(self.backend.runs - before_edit, 4)
            torch.testing.assert_close(comp_edit, eager_edit, rtol=1e-5, atol=1e-5)

            # Multiple resolutions & batch=1 (Turbo CFG=1) stay inside RECOMPILE_LIMIT
            dynamic_args.ref_latents = []
            for batch, h, w in ((1, 4, 4), (2, 6, 4), (1, 6, 4), (2, 4, 6), (2, 8, 6)):
                self.kmodel.apply_model(
                    torch.randn(batch, 16, 1, h, w),
                    torch.full((batch,), 0.5),
                    torch.randn(batch, 5, 12, 128),
                    transformer_options={},
                )

        self.assertEqual(self.fallbacks, [])
        self.assertLessEqual(self.backend.frame_count, RECOMPILE_LIMIT)
        self.assertTrue(all(type(block).__name__ != "OptimizedModule" for block in self.model.blocks))
        del compiled_step2


if __name__ == "__main__":
    unittest.main()
