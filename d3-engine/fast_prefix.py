"""D3 prefix attention with compact GQA keys and the checkpoint's last-token readout."""

import time

import torch

from engine import QuestionBranches


def cpu_segment(q, k, v, scale):
    """Q is [B,Hq,L,D]. Every row reads a single shared [1,Hkv,S,D] segment."""
    batch, heads, length, dim = q.shape
    kv_heads = k.shape[1]
    groups = heads // kv_heads
    packed = q.reshape(batch, kv_heads, groups, length, dim).permute(1, 0, 2, 3, 4)
    packed = packed.reshape(1, kv_heads, batch * groups * length, dim).float().contiguous()
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        packed, k.float().contiguous(), v.float().contiguous(), scale=scale)
    out = out.reshape(kv_heads, batch, groups, length, dim).permute(1, 0, 2, 3, 4)
    lse = lse.reshape(kv_heads, batch, groups, length).permute(1, 0, 2, 3)
    return out.reshape(batch, heads, length, dim), lse.reshape(batch, heads, length)


def merge_segments(first, first_lse, second, second_lse):
    total = torch.logaddexp(first_lse, second_lse)
    return ((first_lse - total).exp().unsqueeze(-1) * first
            + (second_lse - total).exp().unsqueeze(-1) * second)


class FastBranches(QuestionBranches):
    single = False

    def __init__(self, source, counts, rows, backend, pool):
        super().__init__(source, counts, rows)
        self.tail, full = self.source.stack[-1]
        if not full:
            raise ValueError("last-token pruning requires a final full-attention layer")
        self.npu = None
        if backend == "rknn":
            from prefix_rknn import RknnSegments
            self.npu = RknnSegments(pool)
        self.source._attention = self.attention

    def close(self):
        if self.npu is not None:
            self.npu.close()
        del self.source._attention

    def rotate(self, x, rope):
        # The checkpoint also supports partial rotary dimensions. Use its exact implementation.
        return self.source.qwen.apply_rotary_pos_emb(x, x[:, :0], *rope)[0]

    def kv(self, attn, x, rope):
        shape = (*x.shape[:-1], -1, attn.head_dim)
        k = self.rotate(attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2), rope)
        return k, attn.v_proj(x).view(shape).transpose(1, 2)

    def query(self, attn, x, rope):
        shape = (*x.shape[:-1], -1, 2 * attn.head_dim)
        q, gate = attn.q_proj(x).view(shape).chunk(2, dim=-1)
        return self.rotate(attn.q_norm(q).transpose(1, 2), rope), gate.flatten(-2)

    def cache_kv(self, kv):
        if self.npu is None or not self.npu.long_keys(kv[0]):
            kv = tuple(t.float().contiguous() for t in kv)
        self.counts["compact_state_kv_bytes"] += sum(t.numel() * t.element_size() for t in kv)
        return kv

    def segment(self, q, kv, scale):
        start = time.perf_counter()
        name = "arm"
        result = None
        if self.npu is not None and self.npu.supports(q, kv[0]):
            result = self.npu(q, *kv, scale)
            if result is not None:
                name = "rknn"
            else:
                self.counts["segment_range_fallback"] += 1
        if result is None:
            result = cpu_segment(q, *kv, scale)
        self.counts[name + "_segments"] += 1
        self.counts[name + "_attention_s"] += time.perf_counter() - start
        return result

    def read(self, q, kv, prefix, lengths, scale):
        if prefix is None:
            return self.segment(q, kv, scale)[0]
        out, lse = self.segment(q, prefix, scale)
        k, v = kv
        # Padding rows are unobservable. Each real query sees every valid key in its own question.
        for b, length in enumerate(lengths):
            active = min(q.shape[2], length)
            if not active:
                continue
            index = (slice(b, b + 1), slice(None), slice(0, active))
            own = (k[b:b+1, :, :length], v[b:b+1, :, :length])
            suffix, suffix_lse = self.segment(q[index], own, scale)
            out[index] = merge_segments(out[index], lse[index], suffix, suffix_lse)
        return out

    @staticmethod
    def project(attn, out, gate):
        hidden = out.transpose(1, 2).reshape(gate.shape).to(gate.dtype)
        return attn.o_proj(hidden * torch.sigmoid(gate))

    def attention(self, attn, hidden, rope, entry, present):
        kv = self.kv(attn, hidden, rope)
        if entry is None:
            kv = self.cache_kv(kv)
        q, gate = self.query(attn, hidden, rope)
        lengths = present.sum(-1).tolist() if present is not None else None
        out = self.read(q, kv, entry, lengths, attn.scaling)
        return self.project(attn, out, gate), kv

    def state(self, embeds, rope):
        if self.single:
            return embeds, rope
        return super().state(embeds, rope)

    def state_layer(self, layer, softmax, hidden, rope):
        if layer is self.tail:
            self.counts["tail_state_queries_elided"] += hidden.shape[1]
            return None, self.cache_kv(self.kv(layer.self_attn, layer.input_layernorm(hidden), rope))
        return super().state_layer(layer, softmax, hidden, rope)

    def last(self, hidden, rope, lengths, prefix):
        layer = self.tail
        b = torch.arange(len(lengths), device=hidden.device)
        last = torch.tensor(lengths, device=hidden.device) - 1
        x = layer.input_layernorm(hidden)
        attn = layer.self_attn
        kv = self.kv(attn, x, rope)
        q, gate = self.query(attn, x[b, last][:, None], tuple(t[b, last][:, None] for t in rope))
        out = self.read(q, kv, prefix, lengths, attn.scaling)
        h = hidden[b, last][:, None] + self.project(attn, out, gate)
        self.counts["tail_readout_queries"] += len(lengths)
        return self.source.language_model.norm((h + layer.mlp(layer.post_attention_layernorm(h)))[:, 0])

    def question_group(self, cache, hidden, lengths, rope):
        if self.single:
            state, state_rope = cache
            if len(lengths) != 1:
                raise RuntimeError("single-pass request unexpectedly contains multiple rows")
            split = state.shape[1]
            hidden = torch.cat((state, hidden[:, :lengths[0]]), dim=1)
            rope = tuple(torch.cat((a, b[:, :lengths[0]]), dim=1) for a, b in zip(state_rope, rope))
            for layer, full in self.source.stack[:-1]:
                if full:
                    attn = layer.self_attn
                    x = layer.input_layernorm(hidden)
                    k, v = self.kv(attn, x, rope)
                    q, gate = self.query(attn, x, rope)
                    prefix = self.segment(q[:, :, :split], (k[:, :, :split], v[:, :, :split]), attn.scaling)[0]
                    suffix = self.segment(q[:, :, split:], (k, v), attn.scaling)[0]
                    hidden = hidden + self.project(attn, torch.cat((prefix, suffix), dim=2), gate)
                    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
                else:
                    hidden, _ = self.source._layer(layer, False, hidden, rope, None, None)
            self.counts["single_passes"] += 1
            return self.last(hidden, rope, [hidden.shape[1]], None)
        present = torch.arange(hidden.shape[1], device=hidden.device)[None] < torch.tensor(lengths, device=hidden.device)[:, None]
        hidden = hidden * present[..., None].to(hidden.dtype)
        for (layer, full), entry in zip(self.source.stack[:-1], cache):
            hidden, _ = self.source._layer(layer, full, hidden, rope, entry, present)
        return self.last(hidden, rope, lengths, cache[-1])
