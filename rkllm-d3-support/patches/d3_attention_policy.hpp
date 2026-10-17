// Portable policy component for a future RKLLM native-source integration.
// This header is NOT injected into or linked by the precompiled librkllmrt.so.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace d3_rkllm {
enum class LayerKind { LinearAttention, FullAttention };

inline bool is_causal(LayerKind kind) {
  // GatedDeltaNet keeps its original causal scan. Only full attention is noncausal.
  return kind == LayerKind::LinearAttention;
}

inline LayerKind select_layer(const std::vector<LayerKind>& types, size_t index) {
  if (index >= types.size()) throw std::out_of_range("Layer index exceeds explicit layout");
  return types[index];  // Never reconstruct D3 layout from full_attention_interval.
}

// Produces boolean [batch, query_rows, prompt_tokens] masks. True means allowed;
// a GGML backend must translate false to -infinity and true to zero if it uses
// additive masks. The caller must project ALL current-layer K/V before this step.
inline std::vector<uint8_t> full_attention_mask(
    LayerKind kind, const std::vector<uint8_t>& present,
    size_t batch, size_t tokens, size_t query_start, size_t query_rows,
    size_t available_current_layer_kv, size_t history_tokens = 0) {
  if (kind != LayerKind::FullAttention)
    throw std::invalid_argument("Do not replace the GatedDeltaNet scan with a full-attention mask");
  if (!batch || !tokens || !query_rows || query_rows > tokens ||
      query_start > tokens - query_rows || batch > SIZE_MAX / tokens ||
      present.size() != batch * tokens ||
      query_rows > SIZE_MAX / (batch * tokens))
    throw std::invalid_argument("Invalid batch, query or key-mask shape");
  if (history_tokens)
    throw std::invalid_argument("Noncausal D3 requires recomputation, not decoder history reuse");
  if (available_current_layer_kv != tokens)
    throw std::invalid_argument("Full-sequence current-layer K/V must exist before noncausal attention");
  std::vector<uint8_t> mask(batch * query_rows * tokens);
  for (size_t b = 0; b < batch; ++b)
    for (size_t q = 0; q < query_rows; ++q)
      for (size_t k = 0; k < tokens; ++k)
        mask[(b * query_rows + q) * tokens + k] = present[b * tokens + k] != 0;
  return mask;
}
}  // namespace d3_rkllm
