"""Preserve the original batch/padding, bound aggregate rows and each native tile."""
from long_projection_fp32_candidate import PreciseProjection

def install(owner):
    owner.long_projection=PreciseProjection(owner,max_rows=32768,max_output_bytes=256*1024*1024)
    return owner.long_projection.close
