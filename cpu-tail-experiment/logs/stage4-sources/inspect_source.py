import inspect,json
from pathlib import Path
from baseline import load_baseline
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from transformers.integrations.sdpa_attention import sdpa_attention_forward
Path('/work/logs/decoder-source.txt').write_text(inspect.getsource(qwen.Qwen3_5DecoderLayer)+'\n'+inspect.getsource(qwen.Qwen3_5GatedDeltaNet.forward)+'\n'+inspect.getsource(sdpa_attention_forward))
