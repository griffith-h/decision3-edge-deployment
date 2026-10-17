#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "rknn_matmul_api.h"

// Experimental, synchronous, single-owner handle. Callers must serialize calls.
static thread_local std::string last_error;
struct Matrix {
  rknn_matmul_ctx ctx = 0;
  rknn_tensor_mem *a = nullptr, *b = nullptr, *c = nullptr;
  int k, n, selected = -1, prepared_rows = 0;
  std::vector<rknn_matmul_shape> shapes;
  std::vector<rknn_matmul_io_attr> attrs;
  ~Matrix() {
    if (a) rknn_destroy_mem(ctx, a);
    if (b) rknn_destroy_mem(ctx, b);
    if (c) rknn_destroy_mem(ctx, c);
    if (ctx) rknn_matmul_destroy(ctx);
  }
};
static bool check(int code, const char* operation) {
  if (code == 0) return true;
  last_error = std::string(operation) + " returned " + std::to_string(code);
  return false;
}
extern "C" const char* mm_error() { return last_error.c_str(); }
extern "C" void* mm_create_core(int k, int n, const void* half_kn, int core) {
  last_error.clear();
  if (k <= 0 || k > 10240 || k % 16 || n <= 0 || n % 8 || !half_kn) {
    last_error = "Unsupported K/N alignment or missing weights"; return nullptr;
  }
  auto p = std::make_unique<Matrix>();
  p->k = k; p->n = n;
  for (int m : {128, 256, 512}) p->shapes.push_back({m,k,n});
  p->attrs.resize(p->shapes.size());
  rknn_matmul_info info{};
  info.M=512; info.K=k; info.N=n;
  info.type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
  info.B_layout=RKNN_MM_LAYOUT_NATIVE;
  info.AC_layout=RKNN_MM_LAYOUT_NORM;
  if (!check(rknn_matmul_create_dynamic_shape(&p->ctx, &info,
      p->shapes.size(), p->shapes.data(), p->attrs.data()), "create_dynamic")) return nullptr;
  // RK3576 matmul rejects combined core mask 3; use the supported auto mode.
  if (!check(rknn_matmul_set_core_mask(p->ctx, static_cast<rknn_core_mask>(core)), "set_cores")) return nullptr;
  uint32_t as=0, bs=0, cs=0;
  for (auto& attr : p->attrs) {
    as=std::max(as,attr.A.size); bs=std::max(bs,attr.B.size); cs=std::max(cs,attr.C.size);
  }
  p->a=rknn_create_mem(p->ctx,as); p->b=rknn_create_mem(p->ctx,bs); p->c=rknn_create_mem(p->ctx,cs);
  if (!p->a || !p->b || !p->c) { last_error="rknn_create_mem failed"; return nullptr; }
  if (!check(rknn_B_normal_layout_to_native_layout(const_cast<void*>(half_kn),p->b->virt_addr,k,n,&info), "pack_B")) return nullptr;
  if (!check(rknn_matmul_set_dynamic_shape(p->ctx,&p->shapes[0]),"initial_shape")) return nullptr;
  if (!check(rknn_matmul_set_io_mem(p->ctx,p->b,&p->attrs[0].B),"bind_B")) return nullptr;
  std::memset(p->a->virt_addr,0,as);
  return p.release();
}
extern "C" int mm_run(void*,int,const void*,float*) { last_error="Legacy FP32 output API disabled in FP16 ablation"; return -1; }
extern "C" void mm_destroy(void* handle) { delete static_cast<Matrix*>(handle); }

extern "C" void* mm_create(int k, int n, const void* half_kn) {
  return mm_create_core(k,n,half_kn,0);
}
static uint16_t bf_round(float value) {
  uint32_t u; std::memcpy(&u,&value,4);
  if ((u & 0x7fffffffU)>0x7f800000U) return 0x7fc0;
  return uint16_t((u+0x7fffU+((u>>16)&1U))>>16);
}
static uint16x8_t float_half_bf16(uint16x8_t raw) {
  uint16x4_t halves[2]={vget_low_u16(raw),vget_high_u16(raw)};uint16x4_t result[2];
  for(int j=0;j<2;++j){uint32x4_t u=vreinterpretq_u32_f32(vcvt_f32_f16(vreinterpret_f16_u16(halves[j])));
    uint32x4_t r=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16);
    r=vbslq_u32(vcgtq_u32(vandq_u32(u,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000)),vdupq_n_u32(0x7fc0),r);result[j]=vmovn_u32(r);
  }return vcombine_u16(result[0],result[1]);
}
static uint16_t input_lut[65536],output_lut[65536];
struct D3ConversionTables {
 D3ConversionTables(){for(unsigned i=0;i<65536;++i){
  uint32_t u=i<<16;float f;std::memcpy(&f,&u,4);__fp16 h=static_cast<__fp16>(f);std::memcpy(input_lut+i,&h,2);
  uint16_t raw=i;std::memcpy(&h,&raw,2);output_lut[i]=bf_round(static_cast<float>(h));
 }}
};
static D3ConversionTables d3_tables;
extern "C" int mm_fast_convert_output(const uint16_t*src,uint16_t*dst,size_t count){
 unsigned bad=0;for(size_t i=0;i<count;++i){unsigned x=src[i];bad|=(x&0x7fff)>=0x7bff;dst[i]=output_lut[x];}return bad?-3:0;
}
extern "C" int mm_convert_input(const uint16_t*src,uint16_t*dst,size_t count){
 unsigned bad=0;for(size_t i=0;i<count;++i){uint16_t v=input_lut[src[i]];bad|=(v&0x7c00)==0x7c00;dst[i]=v;}return bad?-2:0;
}
extern "C" void mm_convert_output(const float* src, uint16_t* dst, size_t count) {
  size_t i=0;
  for (;i+4<=count;i+=4) {
    uint32x4_t u=vreinterpretq_u32_f32(vld1q_f32(src+i));
    uint32x4_t rounded=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16);
    uint32x4_t nan=vcgtq_u32(vandq_u32(u,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000));
    rounded=vbslq_u32(nan,vdupq_n_u32(0x7fc0),rounded);
    vst1_u16(dst+i,vmovn_u32(rounded));
  }
  for (;i<count;++i) dst[i]=bf_round(src[i]);
}
static bool bind_shape(Matrix* p, int rows) {
  if (!p || rows<=0 || rows>512) { last_error="Invalid run argument"; return false; }
  int index=rows<=128?0:rows<=256?1:2;
  if (p->selected!=index) {
    if (!check(rknn_matmul_set_dynamic_shape(p->ctx,&p->shapes[index]),"set_shape")) return false;
    if (!check(rknn_matmul_set_io_mem(p->ctx,p->a,&p->attrs[index].A),"bind_A")) return false;
    if (!check(rknn_matmul_set_io_mem(p->ctx,p->c,&p->attrs[index].C),"bind_C")) return false;
    p->selected=index;
  }
  return true;
}
extern "C" int mm_prepare_bf16(void* handle, int rows, const uint16_t* input) {
  auto* p=static_cast<Matrix*>(handle);
  if (!input || !bind_shape(p,rows)) return -1;
  p->prepared_rows=0;
  int status=mm_convert_input(input,static_cast<uint16_t*>(p->a->virt_addr),size_t(rows)*p->k);
  if (status) { last_error="FP16 activation overflow"; return status; }
  std::memset(static_cast<uint16_t*>(p->a->virt_addr)+size_t(rows)*p->k,0,size_t(p->shapes[p->selected].M-rows)*p->k*2);
  p->prepared_rows=rows;
  return 0;
}
extern "C" int mm_copy_prepared(void* destination, void* source, int rows) {
  auto* dst=static_cast<Matrix*>(destination);auto* src=static_cast<Matrix*>(source);
  if (!src || !dst || src->k!=dst->k || src->prepared_rows!=rows || !bind_shape(dst,rows)) { last_error="Invalid shared input"; return -1; }
  std::memcpy(dst->a->virt_addr,src->a->virt_addr,size_t(rows)*src->k*2);
  std::memset(static_cast<uint16_t*>(dst->a->virt_addr)+size_t(rows)*dst->k,0,size_t(dst->shapes[dst->selected].M-rows)*dst->k*2);
  dst->prepared_rows=rows;
  return 0;
}
extern "C" int mm_run_prepared_bf16(void* handle, int rows, const float* bias, uint16_t* output) {
  auto* p=static_cast<Matrix*>(handle);
  if (!p || !output || rows<=0 || rows>512 || p->prepared_rows!=rows) { last_error="Input not prepared"; return -1; }
  if (!check(rknn_matmul_run(p->ctx),"run")) return -1;
  const __fp16* src=static_cast<__fp16*>(p->c->virt_addr);
  uint16x4_t exceptional=vdup_n_u16(0);bool scalar_bad=false;
  // Deliberate extra FP16 output boundary: precision must pass model checks.
  // Bias-free model linears use vector FP16->FP32->BF16 conversion.
  if (!bias) {
    scalar_bad=mm_fast_convert_output(reinterpret_cast<const uint16_t*>(src),output,size_t(rows)*p->n)!=0;
  } else {
    for (int r=0;r<rows;++r) for (int n=0;n<p->n;++n) {
      size_t i=size_t(r)*p->n+n;uint16_t raw;std::memcpy(&raw,src+i,2);
      scalar_bad|=(raw&0x7fff)>=0x7bff;output[i]=bf_round(static_cast<float>(src[i])+bias[n]);
    }
  }
  if(scalar_bad || (vget_lane_u16(exceptional,0)|vget_lane_u16(exceptional,1)|vget_lane_u16(exceptional,2)|vget_lane_u16(exceptional,3))) {
    last_error="FP16 output saturated/nonfinite; original CPU fallback required";return -3;
  }
  return 0;
}
extern "C" int mm_run_bf16(void* handle, int rows, const uint16_t* input, const float* bias, uint16_t* output) {
  int code=mm_prepare_bf16(handle,rows,input);
  return code ? code : mm_run_prepared_bf16(handle,rows,bias,output);
}
// Scatter disjoint output columns directly to original projection buffers.
// Each segment covers [start,start+width) in this packed context; stride is
// the complete destination projection width, target points at its first column.
extern "C" int mm_run_prepared_scatter_bf16(void* handle,int rows,int segments,
    const int* starts,const int* widths,const int* strides,uint16_t* const* targets,const float* bias) {
  auto* p=static_cast<Matrix*>(handle);
  if(!p || rows<=0 || rows>512 || p->prepared_rows!=rows || segments<=0 ||
     !starts || !widths || !strides || !targets) {last_error="Invalid scatter arguments";return -1;}
  int covered=0;
  for(int s=0;s<segments;++s) {
    if(starts[s]!=covered || widths[s]<=0 || strides[s]<widths[s] || !targets[s] ||
       starts[s]+widths[s]>p->n) {last_error="Invalid scatter layout";return -1;}
    covered+=widths[s];
  }
  if(covered!=p->n) {last_error="Incomplete scatter layout";return -1;}
  if(!check(rknn_matmul_run(p->ctx),"run_scatter")) return -1;
  const __fp16* source=static_cast<__fp16*>(p->c->virt_addr);
  uint16x4_t exceptional=vdup_n_u16(0);bool scalar_bad=false;
  for(int r=0;r<rows;++r) for(int s=0;s<segments;++s) {
    const __fp16* src=source+size_t(r)*p->n+starts[s];
    uint16_t* dst=targets[s]+size_t(r)*strides[s];int col=0;
    if(!bias)scalar_bad|=mm_fast_convert_output(reinterpret_cast<const uint16_t*>(src),dst,widths[s])!=0;
    else {
    for(;col+4<=widths[s];col+=4) {
      float16x4_t half=vld1_f16(src+col);
      exceptional=vorr_u16(exceptional,vcge_u16(vand_u16(vreinterpret_u16_f16(half),vdup_n_u16(0x7fff)),vdup_n_u16(0x7bff)));
      float32x4_t f=vcvt_f32_f16(half);
      if(bias) f=vaddq_f32(f,vld1q_f32(bias+starts[s]+col));
      uint32x4_t u=vreinterpretq_u32_f32(f);
      uint32x4_t rounded=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16);
      uint32x4_t nan=vcgtq_u32(vandq_u32(u,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000));
      rounded=vbslq_u32(nan,vdupq_n_u32(0x7fc0),rounded);vst1_u16(dst+col,vmovn_u32(rounded));
    }
    for(;col<widths[s];++col) {
      uint16_t raw;std::memcpy(&raw,src+col,2);scalar_bad|=(raw&0x7fff)>=0x7bff;
      dst[col]=bf_round(static_cast<float>(src[col])+(bias?bias[starts[s]+col]:0.f));
    }
    }
  }
  if(scalar_bad || (vget_lane_u16(exceptional,0)|vget_lane_u16(exceptional,1)|vget_lane_u16(exceptional,2)|vget_lane_u16(exceptional,3))) {
    last_error="FP16 output saturated/nonfinite; original CPU fallback required";return -3;
  }
  return 0;
}
