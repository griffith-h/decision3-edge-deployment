"""Public SDK type4 output ablation, original FP16 weights, no quantization."""
from pathlib import Path
root=Path(__file__).resolve().parent
s=(root.parent/'bridge-parallel-experiment/fused_bridge.cpp').read_text()
s=s.replace('info.type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;','info.type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;')
old='''  const float* src=static_cast<float*>(p->c->virt_addr);
  if (!bias) mm_convert_output(src,output,size_t(rows)*p->n);
  else {
    for (int r=0;r<rows;++r) for (int n=0;n<p->n;++n) output[size_t(r)*p->n+n]=bf_round(src[size_t(r)*p->n+n]+bias[n]);
  }'''
new='''  const __fp16* src=static_cast<__fp16*>(p->c->virt_addr);
  // Deliberate extra FP16 output boundary: precision must pass model checks.
  // Bias-free model linears use vector FP16->FP32->BF16 conversion.
  if (!bias) {
    size_t count=size_t(rows)*p->n,i=0;
    for (;i+4<=count;i+=4) {
      uint32x4_t u=vreinterpretq_u32_f32(vcvt_f32_f16(vld1_f16(src+i)));
      uint32x4_t rounded=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16);
      uint32x4_t nan=vcgtq_u32(vandq_u32(u,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000));
      rounded=vbslq_u32(nan,vdupq_n_u32(0x7fc0),rounded);
      vst1_u16(output+i,vmovn_u32(rounded));
    }
    for (;i<count;++i) output[i]=bf_round(static_cast<float>(src[i]));
  } else {
    for (int r=0;r<rows;++r) for (int n=0;n<p->n;++n) output[size_t(r)*p->n+n]=bf_round(static_cast<float>(src[size_t(r)*p->n+n])+bias[n]);
  }'''
assert old in s;s=s.replace(old,new)
# The legacy API is deliberately disabled: it exposes FP32 C to callers and
# must never memcpy FP32 bytes from a new FP16 C allocation.
start=s.index('extern "C" int mm_run(void*')
end=s.index('extern "C" void mm_destroy',start)
s=s[:start]+'''extern "C" int mm_run(void*,int,const void*,float*) { last_error="Legacy FP32 output API disabled in FP16 ablation"; return -1; }
'''+s[end:]
(root/'half_bridge.cpp').write_text(s)
