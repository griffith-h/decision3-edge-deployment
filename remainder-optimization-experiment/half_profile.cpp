#include <algorithm>
#include <chrono>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
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
  double profile[8] = {};
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
extern "C" int mm_convert_input(const uint16_t* src, uint16_t* dst, size_t count) {
  bool bad=false; size_t i=0;
  for (; i+4<=count; i+=4) {
    float32x4_t value=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(src+i),16));
    uint16x4_t half=vreinterpret_u16_f16(vcvt_f16_f32(value));
    vst1_u16(dst+i,half);
    uint16x4_t exceptional=vceq_u16(vand_u16(half,vdup_n_u16(0x7c00)),vdup_n_u16(0x7c00));
    bad |= (vget_lane_u16(exceptional,0)|vget_lane_u16(exceptional,1)|vget_lane_u16(exceptional,2)|vget_lane_u16(exceptional,3))!=0;
  }
  for (; i<count; ++i) {
    uint32_t u=uint32_t(src[i])<<16;float f;std::memcpy(&f,&u,4);
    __fp16 half=static_cast<__fp16>(f);std::memcpy(dst+i,&half,2);
    bad |= (dst[i]&0x7c00)==0x7c00;
  }
  return bad ? -2 : 0;
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
extern "C" void mm_profile_read(void* handle,double* out,int reset) {
  auto* p=static_cast<Matrix*>(handle);
  std::memcpy(out,p->profile,sizeof(p->profile));
  if (reset) std::memset(p->profile,0,sizeof(p->profile));
}
extern "C" int mm_prepare_bf16(void* handle, int rows, const uint16_t* input) {
  auto* p=static_cast<Matrix*>(handle);
  double stamp=now();
  if (!input || !bind_shape(p,rows)) return -1;
  p->profile[1]+=now()-stamp;stamp=now();
  p->prepared_rows=0;
  int status=mm_convert_input(input,static_cast<uint16_t*>(p->a->virt_addr),size_t(rows)*p->k);
  p->profile[2]+=now()-stamp;stamp=now();
  if (status) { last_error="FP16 activation overflow"; return status; }
  std::memset(static_cast<uint16_t*>(p->a->virt_addr)+size_t(rows)*p->k,0,size_t(p->shapes[p->selected].M-rows)*p->k*2);
  p->profile[3]+=now()-stamp;
  p->prepared_rows=rows;
  return 0;
}
extern "C" int mm_copy_prepared(void* destination, void* source, int rows) {
  auto* dst=static_cast<Matrix*>(destination);auto* src=static_cast<Matrix*>(source);
  double stamp=now();
  if (!src || !dst || src->k!=dst->k || src->prepared_rows!=rows || !bind_shape(dst,rows)) { last_error="Invalid shared input"; return -1; }
  std::memcpy(dst->a->virt_addr,src->a->virt_addr,size_t(rows)*src->k*2);
  std::memset(static_cast<uint16_t*>(dst->a->virt_addr)+size_t(rows)*dst->k,0,size_t(dst->shapes[dst->selected].M-rows)*dst->k*2);
  dst->profile[3]+=now()-stamp;
  dst->prepared_rows=rows;
  return 0;
}
extern "C" int mm_run_prepared_bf16(void* handle, int rows, const float* bias, uint16_t* output) {
  auto* p=static_cast<Matrix*>(handle);
  if (!p || !output || rows<=0 || rows>512 || p->prepared_rows!=rows) { last_error="Input not prepared"; return -1; }
  double stamp=now();p->profile[6]=stamp;
  if (!check(rknn_matmul_run(p->ctx),"run")) return -1;
  p->profile[7]=now();p->profile[4]+=p->profile[7]-stamp;p->profile[0]+=1;
  stamp=now();
  const __fp16* src=static_cast<__fp16*>(p->c->virt_addr);
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
  }
  p->profile[5]+=now()-stamp;
  return 0;
}
extern "C" int mm_run_bf16(void* handle, int rows, const uint16_t* input, const float* bias, uint16_t* output) {
  int code=mm_prepare_bf16(handle,rows,input);
  return code ? code : mm_run_prepared_bf16(handle,rows,bias,output);
}
