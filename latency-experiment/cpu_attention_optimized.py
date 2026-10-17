"""CPU implementation experiment; original weights/masks/model layout retained.

Validated against Torch 2.12.0+cpu / Transformers 5.17.0. The chunk equations
follow the Apache-2.0 Transformers Qwen3.5 reference implementation.
"""
import torch
import torch.nn.functional as F


def causal_conv1d_vectorized(hidden_states, weight, bias=None, activation=None, **kwargs):
    """Same causal depthwise cross-correlation, with a BF16 convolution boundary.

    Avoid dispatching one tiny Conv1d per channel on CPU. Pad implicitly by
    slicing valid input ranges; never read a future token. Accumulate in FP32,
    round the convolution result to the original weight dtype before SiLU.
    """
    if hidden_states.device.type != 'cpu' or weight.ndim != 2:
        raise ValueError('This implementation requires CPU depthwise weights [channels, kernel]')
    inputs = hidden_states.to(weight.dtype).float()
    w = weight.float()
    result = inputs * w[:, -1][None, :, None]
    width = inputs.shape[-1]
    for offset in range(1, min(weight.shape[-1], width)):
        result[:, :, offset:] += inputs[:, :, :-offset] * w[:, -1-offset][None, :, None]
    if bias is not None:
        result += bias.float()[None, :, None]
    result = result.to(weight.dtype)
    if activation is not None:
        if activation not in ('silu', 'swish'):
            raise ValueError('Only the validated SiLU activation is supported')
        result = F.silu(result)
    return result.to(hidden_states.dtype)


def install_conv():
    """Install in an isolated process; return a restore callback."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    original = qwen.causal_conv1d_fn

    def forward(hidden_states, weight, bias=None, activation=None, **kwargs):
        if (hidden_states.device.type != 'cpu' or weight.ndim != 2
                or weight.shape[-1] != 4 or activation not in (None, 'silu', 'swish')):
            return original(hidden_states, weight, bias, activation=activation, **kwargs)
        return causal_conv1d_vectorized(hidden_states, weight, bias, activation, **kwargs)

    qwen.causal_conv1d_fn = forward
    def restore():
        qwen.causal_conv1d_fn = original
    return restore


def chunk_delta_prefill(query, key, value, g, beta, chunk_size=64,
                        initial_state=None, output_final_state=False,
                        use_qk_l2norm_in_kernel=False, **kwargs):
    """Stock FP32 chunk equations, omitting zero/unused state matmuls.

    Restricted to the uncached prefill used by D3. All attention masks and
    convolution inputs remain the caller's original tensors.
    """
    if initial_state is not None or output_final_state:
        raise ValueError('Only zero-initial-state prefill without a returned state is supported')
    dtype = query.dtype
    length = query.shape[1]
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                                     for x in (query, key, value, beta, g)]
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    query = query * query.shape[-1] ** -0.5
    pad = (chunk_size - length % chunk_size) % chunk_size
    query, key, value = [F.pad(x, (0, 0, 0, pad)) for x in (query, key, value)]
    beta, decay = [F.pad(x, (0, pad)) for x in (beta, decay)]
    k_beta = key * beta.unsqueeze(-1)
    v_beta = value * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                 for x in (query, key, k_beta, v_beta)]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cumulative = decay.cumsum(3)
    pairwise = (cumulative.unsqueeze(4) - cumulative.unsqueeze(3)).masked_fill(upper, float('-inf')).exp()
    system = (k_beta @ key.transpose(-1, -2)) * pairwise
    attention = (query @ key.transpose(-1, -2)) * pairwise
    decayed = k_beta * cumulative.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(system, v_beta, upper=False, unitriangular=True)
    chunks = query.shape[2]
    # The first chunk has zero incoming state, so its read transform is unused.
    k_read = (torch.linalg.solve_triangular(system[:, :, 1:], decayed[:, :, 1:],
                                          upper=False, unitriangular=True)
              if chunks > 1 else None)
    query = query * cumulative.exp().unsqueeze(-1)
    key = key * (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    chunk_decay = cumulative[..., -1].exp()[..., None, None]
    output = torch.empty_like(new_values)
    state = None
    for i in range(chunks):
        if state is None:
            corrected = new_values[:, :, i]
            output[:, :, i] = attention[:, :, i] @ corrected
        else:
            corrected = new_values[:, :, i] - k_read[:, :, i-1] @ state
            output[:, :, i] = query[:, :, i] @ state + attention[:, :, i] @ corrected
        if i + 1 < chunks:
            update = key[:, :, i].transpose(-1, -2) @ corrected
            state = update if state is None else state * chunk_decay[:, :, i] + update
    output = output.reshape(output.shape[0], output.shape[1], -1, output.shape[-1])[:, :, :length]
    return output.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format), None


def install_scan(chunk_size_override=None):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    original = qwen.torch_chunk_gated_delta_rule

    def forward(query, key, value, g, beta, chunk_size=64, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        if (query.device.type != 'cpu' or initial_state is not None or output_final_state
                or kwargs.get('cu_seqlens') is not None):
            return original(query, key, value, g, beta, chunk_size=chunk_size,
                            initial_state=initial_state, output_final_state=output_final_state,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
        return chunk_delta_prefill(query, key, value, g, beta,
                                   chunk_size=chunk_size_override or chunk_size,
                                   use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)

    qwen.torch_chunk_gated_delta_rule = forward
    def restore():
        qwen.torch_chunk_gated_delta_rule = original
    return restore


def install_sdpa():
    """Use FP32 only for Qwen text attention; preserve every vision call."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    interface = qwen.ALL_ATTENTION_FUNCTIONS
    original = interface["sdpa"]

    def forward(module, query, key, value, attention_mask, *args, **kwargs):
        if (isinstance(module, qwen.Qwen3_5Attention) and query.device.type == "cpu"
                and query.dtype == torch.bfloat16):
            if attention_mask is not None and attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.float()
            output, weights = original(module, query.float(), key.float(), value.float(),
                                       attention_mask, *args, **kwargs)
            return output.to(query.dtype), weights
        return original(module, query, key, value, attention_mask, *args, **kwargs)

    interface.register("sdpa", forward)
    def restore():
        interface.register("sdpa", original)
    return restore
