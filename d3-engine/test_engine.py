import unittest
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from cpu_attention import AttentionPrecision, bind_attention
from engine import D3Engine, QuestionBranches, TiledNPU
from fast_prefix import FastBranches, cpu_segment
from state_kernels import StateKernels


class EngineTests(unittest.TestCase):
    def test_single_pass_is_scoped_to_one_text_question_and_released_on_error(self):
        owner = object.__new__(D3Engine)
        owner.lock, owner.closed = threading.Lock(), False
        owner.fast_branches = SimpleNamespace(single=False)
        def run(*, state, questions):
            if state == "reject":
                raise ValueError("invalid request")
            return owner.fast_branches.single
        owner.model = SimpleNamespace(system_one=run, permutation_average=False)
        one = {"route": {"type": "choice"}}
        self.assertTrue(owner.system_one(state="", questions=one))
        self.assertFalse(owner.system_one(state="", questions={**one, "feedback": {"type": "noul"}}))
        self.assertFalse(owner.system_one(state="", questions={"image": {"type": "box"}}))
        with self.assertRaises(ValueError):
            owner.system_one(state="reject", questions=one)
        self.assertFalse(owner.fast_branches.single)
        owner.model.permutation_average = True
        self.assertFalse(owner.system_one(state="", questions=one))

    @torch.inference_mode()
    def test_shared_prefix_padding_and_last_query(self):
        torch.manual_seed(621)
        owner = object.__new__(FastBranches)
        owner.npu, owner.counts = None, Counter()
        q = torch.randn(3, 4, 6, 16).bfloat16()
        common = tuple(torch.randn(1, 2, 9, 16).bfloat16() for _ in range(2))
        own = tuple(torch.randn(3, 2, 6, 16).bfloat16() for _ in range(2))
        sizes = [1, 6, 4]
        for last in (False, True):
            qq = torch.stack([q[i, :, n-1:n] for i, n in enumerate(sizes)]) if last else q
            actual = owner.read(qq, own, common, sizes, 0.1)
            for b, size in enumerate(sizes):
                keys, values = [torch.cat((c, o[b:b+1, :, :size]), 2).repeat_interleave(2, dim=1).float()
                                for c, o in zip(common, own)]
                count = 1 if last else size
                ref = torch.nn.functional.scaled_dot_product_attention(qq[b:b+1, :, :count].float(),
                                                                        keys, values, scale=0.1)
                torch.testing.assert_close(actual[b:b+1, :, :count], ref, rtol=2e-5, atol=2e-6)
        padding_only = owner.read(q[:1], tuple(x[:1] for x in own), common, [0], 0.1)
        torch.testing.assert_close(padding_only, cpu_segment(q[:1], *common, 0.1)[0])

    @torch.inference_mode()
    def test_decision_tail_matches_full_layer_with_partial_rope(self):
        from torch import nn
        torch.manual_seed(790)
        model_width, heads, dim = 32, 4, 8
        projection = lambda width: nn.Linear(model_width, width, bias=False).to(torch.bfloat16)
        attn = SimpleNamespace(head_dim=dim, scaling=0.35, q_norm=nn.Identity(), k_norm=nn.Identity(),
                               q_proj=projection(heads * dim * 2), k_proj=projection(2*dim),
                               v_proj=projection(2*dim), o_proj=projection(model_width))
        owner = object.__new__(FastBranches)
        owner.source = SimpleNamespace(qwen=qwen, language_model=SimpleNamespace(norm=nn.Identity()))
        owner.tail = SimpleNamespace(self_attn=attn, input_layernorm=nn.Identity(),
                                     post_attention_layernorm=nn.Identity(), mlp=projection(model_width))
        owner.npu, owner.counts = None, Counter()
        hidden = torch.randn(2, 7, model_width).bfloat16()
        angles = torch.randn(2, 7, 4)
        rope = angles.cos().bfloat16(), angles.sin().bfloat16()
        prefix = tuple(torch.randn(1, 2, 5, dim).bfloat16() for _ in range(2))
        present = torch.arange(7)[None] < torch.tensor([3, 7])[:, None]
        mixed, _ = owner.attention(attn, hidden, rope, prefix, present)
        full = hidden + mixed
        full = full + owner.tail.mlp(full)
        torch.testing.assert_close(owner.last(hidden, rope, [3, 7], prefix), full[[0, 1], [2, 6]],
                                   rtol=0.008, atol=0.008)

    @unittest.skipUnless(Path("/dev/dri").exists(), "needs RK3576 NPU")
    @torch.inference_mode()
    def test_native_256_row_batch_and_late_range_fallback(self):
        from torch import nn

        class Mlp(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(128, 256, bias=False)
                self.up_proj = nn.Linear(128, 256, bias=False)
                self.down_proj = nn.Linear(256, 128, bias=False)
                self.act_fn = nn.SiLU()

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        torch.manual_seed(45)
        with torch.inference_mode(False):
            layer = nn.Module()
            layer.mlp = Mlp().bfloat16().eval()
        model = SimpleNamespace(backbone=SimpleNamespace(language_model=SimpleNamespace(layers=[layer])))
        x = torch.randn(2, 139, 128).bfloat16()
        old = TiledNPU(model)
        try:
            reference = layer.mlp(x)
        finally:
            old.close()
        bad = x.clone()
        bad[-1, -1, 0] = 1e8
        fallback = layer.mlp(bad)
        new = TiledNPU(model, native_tiles=True)
        try:
            torch.testing.assert_close(layer.mlp(x), reference, rtol=0.008, atol=0.002)
            torch.testing.assert_close(layer.mlp(bad), fallback, rtol=0, atol=0)
            self.assertEqual(new.counts["native_tile_batches"], 1)
            self.assertEqual(new.counts["linear_range_fallback"], 1)
        finally:
            new.close()

    @unittest.skipUnless(Path("/dev/dri").exists(), "needs RK3576 NPU")
    @torch.inference_mode()
    def test_rknn_rectangular_segment_and_key_block_tail(self):
        from prefix_rknn import RknnSegments
        torch.manual_seed(806)
        q = torch.randn(2, 4, 19, 256).bfloat16()
        k, v = torch.randn(2, 1, 2, 8201, 256).bfloat16().unbind()
        ref, ref_lse = cpu_segment(q, k, v, 0.0625)
        with ThreadPoolExecutor(2) as pool:
            owner = RknnSegments(pool)
            try:
                got, lse = owner(q, k, v, 0.0625)
                torch.testing.assert_close(got, ref, rtol=0.02, atol=0.0004)
                torch.testing.assert_close(lse, ref_lse, rtol=0.00005, atol=0.0004)
                bad = q.clone()
                bad[0, 0, 0, 0] = 1e8
                self.assertIsNone(owner(bad, k, v, 0.0625))
            finally:
                owner.close()
            self.assertEqual(owner.handles, [])

    def test_default_precision_for_prefix_and_checkpoint_override(self):
        class StateOnce:
            pass

        source = StateOnce()
        source.qwen = object()
        model = SimpleNamespace(backbone_type="qwen3_5", attention_mode="prefix_bidir", device="cpu",
                                readout_dtype="float32", backbone=SimpleNamespace(training=False), state_once=source)
        with (patch("engine.TiledNPU"), patch("engine.QuestionBranches") as branches,
              patch("cpu_attention.bind_attention") as bind):
            with D3Engine(model) as owner:
                bind.assert_called_once_with(branches.return_value.source, owner.stages)
            self.assertIs(model.state_once, source)
            bind.reset_mock()
            with D3Engine(model, attention="checkpoint"):
                bind.assert_not_called()
        with patch("engine.TiledNPU"):
            with D3Engine(model, attention="checkpoint", checkpoint_state_kernels=True):
                self.assertIsNot(model.state_once.source, source)
                self.assertIs(model.state_once.source.qwen, source.qwen)
            self.assertIs(model.state_once, source)
        with patch("engine.TiledNPU"), patch("fast_prefix.FastBranches") as fast:
            with D3Engine(model, aggressive=True) as owner:
                self.assertEqual(owner.prefix_backend, "rknn")
                self.assertEqual(fast.call_args.args[3], "rknn")
            with D3Engine(model, aggressive=True, prefix_backend="arm") as owner:
                self.assertEqual(owner.prefix_backend, "arm")
                self.assertEqual(fast.call_args.args[3], "arm")

    def test_second_owner_cannot_replace_live_buffers(self):
        model = SimpleNamespace(backbone_type="qwen3_5", attention_mode="noncausal_full_attention", device="cpu",
                                readout_dtype="float32", backbone=SimpleNamespace(training=False))
        with patch("engine.TiledNPU"):
            with D3Engine(model) as first:
                with self.assertRaises(ValueError):
                    D3Engine(model)
                self.assertIs(model._rk3576_owner, first)
            self.assertNotIn("_rk3576_owner", vars(model))
            with D3Engine(model):
                pass

    @torch.inference_mode()
    def test_precision_probe_preserves_mask(self):
        q, k, v = torch.randn(3, 2, 3, 5, 8).bfloat16().unbind()
        allowed = torch.ones(2, 1, 5, 5, dtype=torch.bool)
        allowed[1, :, :, -2:] = False
        fn = torch.nn.functional.scaled_dot_product_attention
        standard = fn(q, k, v, attn_mask=allowed)
        alternative = fn(q.float(), k.float(), v.float(), attn_mask=allowed).bfloat16()
        for replay, expected in ((True, standard), (False, alternative)):
            stats = Counter()
            with AttentionPrecision(stats, replay):
                got = fn(q, k, v, attn_mask=allowed)
            self.assertTrue(torch.equal(got, expected))
            self.assertEqual(stats["sdpa_calls"], 1)

        class Layer:
            def _attention(self, q, k, v):
                return fn(q, k, v, attn_mask=allowed)
        layer = Layer()
        restore = bind_attention(layer, Counter())
        self.assertTrue(torch.equal(layer._attention(q, k, v), alternative))
        restore()
        self.assertNotIn("_attention", vars(layer))
        self.assertTrue(torch.equal(layer._attention(q, k, v), standard))

    def test_one_owner_serializes_requests_and_close(self):
        owner = object.__new__(D3Engine)
        owner.lock, owner.resources, owner.closed = threading.Lock(), ExitStack(), False
        first_entered, second_submitted, release = (threading.Event() for _ in range(3))
        seen = []
        def infer(*, state, questions):
            seen.append(("enter", state))
            if state == 1:
                first_entered.set()
                self.assertTrue(release.wait(3))
            seen.append(("exit", state))
            return state
        owner.model = SimpleNamespace(system_one=infer)
        def second():
            second_submitted.set()
            return owner.system_one(state=2, questions={})
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(owner.system_one, state=1, questions={})
            self.assertTrue(first_entered.wait(3))
            b = pool.submit(second)
            self.assertTrue(second_submitted.wait(3))
            release.set()
            self.assertEqual([a.result(), b.result()], [1, 2])
        self.assertEqual(seen, [("enter", 1), ("exit", 1), ("enter", 2), ("exit", 2)])
        owner.close()
        with self.assertRaises(RuntimeError):
            owner.system_one(state=3, questions={})

    @torch.inference_mode()
    def test_convolution_continuation_layout(self):
        kernels = StateKernels(qwen, Counter(), Path(__file__).with_name("libd3_state.so"))
        torch.manual_seed(98)
        x = torch.randn(2, 11, 16).bfloat16().transpose(1, 2)
        weight = torch.randn(16, 4).bfloat16()
        bias = torch.randn(16).bfloat16()
        for activations in (x, x.contiguous()):
            expected = qwen.causal_conv1d_fn(activations, weight, bias, activation="silu")
            actual = kernels.causal_conv1d_fn(activations, weight, bias, activation="silu")
            torch.testing.assert_close(actual.float(), expected.float(), atol=0.001, rtol=0.008)
        self.assertEqual(kernels.counts["conv"], 2)

    @torch.inference_mode()
    def test_scan_fork_and_continue(self):
        torch.manual_seed(443)
        kernels = StateKernels(qwen, Counter(), Path(__file__).with_name("libd3_state.so"))
        # Unequal head widths and strided inputs detect layout errors.
        b, t, h, kd, vd = 2, 71, 3, 12, 20
        storage = torch.randn(b, t, h, kd + kd + vd).bfloat16()
        query, key, value = storage.split((kd, kd, vd), -1)
        decay, beta = -torch.rand(b, t, h), torch.rand(b, t, h).bfloat16()
        args = (query, key, value, decay, beta)
        ref, state = qwen.torch_chunk_gated_delta_rule(*args, output_final_state=True, use_qk_l2norm_in_kernel=True)
        _, prefix = kernels.torch_chunk_gated_delta_rule(
            *(x[:, :35] for x in args), output_final_state=True, use_qk_l2norm_in_kernel=True)
        snapshot = prefix.clone()
        tail, final = kernels.torch_chunk_gated_delta_rule(
            *(x[:, 35:] for x in args), initial_state=prefix, output_final_state=True, use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(tail.float(), ref[:, 35:].float(), atol=0.0005, rtol=0.008)
        torch.testing.assert_close(final, state, atol=0.00001, rtol=0.0001)
        torch.testing.assert_close(prefix, snapshot, atol=0, rtol=0)
        # Two different branches continue the same broadcast history, including a one-token tail.
        initial = prefix[:1].expand(b, -1, -1, -1)
        short = [x[:, -1:] for x in args]
        expected, _ = qwen.torch_chunk_gated_delta_rule(*short, initial_state=initial, use_qk_l2norm_in_kernel=True)
        actual, no_state = kernels.torch_chunk_gated_delta_rule(*short, initial_state=initial, use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.0005, rtol=0.008)
        self.assertIsNone(no_state)
        self.assertEqual(kernels.counts["scan_zero"], 1)
        self.assertEqual(kernels.counts["scan_continue"], 2)

    @torch.inference_mode()
    def test_tiles_and_late_fallback(self):
        owner = object.__new__(TiledNPU)
        owner.counts = Counter()
        with torch.inference_mode(False):
            module = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16).eval()
        weights = [(module.weight, module.weight._version)]
        # Non-contiguous batch with an incomplete last tile.
        x = torch.zeros(131, 2, 8, dtype=torch.bfloat16).transpose(0, 1)
        x[:, :, 0] = torch.arange(131).bfloat16()
        sizes = []
        def execute(_module, part):
            sizes.append(len(part))
            return part[:, :4]
        owner.kernels = SimpleNamespace(execute=execute)
        def run():
            return owner.forward("projection", module, [module], weights, lambda t: t[:, :, :4] - 1, x)
        self.assertTrue(torch.equal(run(), x[:, :, :4]))
        self.assertEqual(sizes, [128, 128, 6])
        def reject_last(m, part):
            if len(part) < 128:
                raise ValueError("outside native range")
            return execute(m, part)
        owner.kernels.execute = reject_last
        self.assertTrue(torch.equal(run(), x[:, :, :4] - 1))
        self.assertEqual(owner.counts["linear_range_fallback"], 1)

    def test_branch_padding_and_order(self):
        batches = []
        cache = object()
        def questions(seen, x, ls, rope):
            self.assertIs(seen, cache)
            self.assertEqual(x.shape, rope[0].shape)
            self.assertEqual(x.shape[1], max(ls))
            batches.append(ls)
            return x[torch.arange(len(ls)), torch.tensor(ls)-1]
        owner = object.__new__(QuestionBranches)
        owner.rows, owner.counts, owner.source = 2, Counter(), SimpleNamespace(questions=questions)
        x = torch.arange(3*7).view(3, 7, 1)
        got = owner.questions(cache, x, [3, 4, 7], (x, x))
        self.assertEqual(got.flatten().tolist(), [2, 10, 20])
        self.assertEqual(batches, [[3, 4], [7]])

    def test_cached_conv_tail_owns_only_three_tokens(self):
        parent = torch.arange(24 * 8).view(1, 24, 8)
        tail = parent[:, -3:, :].transpose(1, 2)
        recurrent, softmax_kv = object(), (object(), object())
        def layer(_, softmax, hidden, rope, entry, padding):
            self.assertIsNone(entry)
            return hidden, softmax_kv if softmax else (tail, recurrent)
        owner = object.__new__(QuestionBranches)
        owner.counts = Counter()
        owner.source = SimpleNamespace(stack=[(0, False), (1, True)], _layer=layer)
        saved, kv = owner.state(None, None)
        torch.testing.assert_close(saved[0], tail, rtol=0, atol=0)
        self.assertEqual(saved[0].untyped_storage().nbytes(), tail.numel() * tail.element_size())
        self.assertIs(saved[1], recurrent)
        self.assertIs(kv, softmax_kv)
        self.assertEqual(owner.counts["conv_tail_excess_backing_bytes"], (24 - 3) * 8 * 8)


if __name__ == "__main__":
    torch.set_num_threads(3)
    unittest.main()
