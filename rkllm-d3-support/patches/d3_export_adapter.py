"""D3 export contract adapter for the pinned RKLLM 1.3.1 public writer.

This patches export metadata only. The current precompiled native Runtime does
not implement this contract. Native use is deliberately gated and must remain
disabled until the execution layer is integrated and model parity is validated.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

REQUIRED_NATIVE_FEATURES = (
    'exact_per_layer_layout',
    'noncausal_full_attention',
    'layer_major_full_sequence_prefill',
    'last_hidden_layer',
)


class UnsupportedNativeRuntime(RuntimeError):
    pass


@dataclass(frozen=True)
class D3Contract:
    layer_types: tuple
    hidden_size: int
    attention_mode: str
    config_sha256: str

    @classmethod
    def from_model(cls, model_path):
        root = Path(model_path)
        config_bytes = (root / 'config.json').read_bytes()
        config = json.loads(config_bytes)
        decision = json.loads((root / 'decision_config.json').read_text())
        manifest = json.loads((root / 'MODEL_MANIFEST.json').read_text())
        actual_hash = hashlib.sha256(config_bytes).hexdigest()
        if manifest['files_sha256']['config.json'] != actual_hash:
            raise ValueError('Original checkpoint config integrity check failed')
        if config['model_type'] != 'qwen3_5':
            raise ValueError('Adapter is scoped to the original Qwen3.5 D3 package')
        text = config['text_config']
        kinds = tuple(text['layer_types'])
        if len(kinds) != text['num_hidden_layers']:
            raise ValueError('Layer count and layer_types disagree')
        if set(kinds) - {'full_attention', 'linear_attention'}:
            raise ValueError('Unknown layer type')
        mode = decision['attention_mode']
        if mode != 'noncausal_full_attention':
            raise ValueError('D3 noncausal contract is required')
        return cls(kinds, text['hidden_size'], mode, actual_hash)

    def as_dict(self):
        return {
            'schema': 'd3-rkllm-attention-contract-v1',
            'layer_types': list(self.layer_types),
            'hidden_size': self.hidden_size,
            'full_attention_causal': False,
            'linear_attention_causal': True,
            'prefill_order': 'layer_major_full_sequence',
            'keep_history': False,
            'readout': 'original_cpu_fp32_last_token',
            'original_config_sha256': self.config_sha256,
            'requires_native_runtime_implementation': True,
        }


def require_native_runtime(capabilities):
    missing = [key for key in REQUIRED_NATIVE_FEATURES if capabilities.get(key) is not True]
    if missing:
        raise UnsupportedNativeRuntime('Native RKLLM D3 execution is unavailable: ' + ', '.join(missing))


def write_contract(writer, contract):
    # Standard GGUF field already supported by the published writer. Readback
    # proves serialization only; it does not prove that native attention honors it.
    writer.add_causal_attention(False)
    # Namespaced extension fields are an explicit integration contract for a
    # future native patch, not undocumented switches expected to work today.
    writer.add_string('d3.contract.schema', 'd3-rkllm-attention-contract-v1')
    # Use integer enums for portable metadata readback: 0=linear, 1=full.
    writer.add_array('d3.contract.layer_types', [int(kind == 'full_attention') for kind in contract.layer_types])
    writer.add_bool('d3.contract.linear_attention_causal', True)
    writer.add_uint32('d3.contract.prefill_order', 1)  # 1=layer-major full-sequence prefill.
    writer.add_bool('d3.contract.keep_history', False)
    writer.add_string('d3.contract.config_sha256', contract.config_sha256)


def install_writer_hook(contract):
    """Return the original header method so a caller can restore it in finally.

    Hook installation by itself only prepares metadata. Production execution
    must call require_native_runtime with verified capabilities first.
    """
    import rkllm.base.converter as converter
    original = converter.LLMWriter.write_header_to_file
    def write_header(writer, *args, **kwargs):
        write_contract(writer, contract)
        return original(writer, *args, **kwargs)
    converter.LLMWriter.write_header_to_file = write_header
    return original
