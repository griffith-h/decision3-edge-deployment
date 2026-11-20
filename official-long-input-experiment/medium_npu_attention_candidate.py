"""Development-only extension of the same full-key kernel to B1/T>615."""
from long_npu_attention_candidate import LongNpuAttention

def install(owner):
    owner.long_npu_attention=LongNpuAttention(owner,min_length=615)
    return owner.long_npu_attention.close
