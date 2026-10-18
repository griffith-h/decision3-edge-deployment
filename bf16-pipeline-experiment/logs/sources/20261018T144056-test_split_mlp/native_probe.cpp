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
static thread_local bool create_fp32=false;
extern "C" int mm_set_output_precision(int value){int old=create_fp32;create_fp32=value!=0;return old;}
struct Matrix {
  rknn_matmul_ctx ctx = 0;
  rknn_tensor_mem *a = nullptr, *b = nullptr, *c = nullptr;
  int k, n, selected = -1, prepared_rows = 0; bool fp32=false;
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
  p->k = k; p->n = n; p->fp32=create_fp32;
  for (int m : {128, 256, 512, 640}) p->shapes.push_back({m,k,n});
  p->attrs.resize(p->shapes.size());
  rknn_matmul_info info{};
  info.M=640; info.K=k; info.N=n;
  info.type=p->fp32?RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
  info.B_layout=RKNN_MM_LAYOUT_NATIVE;
  info.AC_layout=RKNN_MM_LAYOUT_NATIVE;
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
extern "C" int mm_fast_convert_output(const uint16_t* src,uint16_t* dst,size_t count) {
  uint16x8_t bad=vdupq_n_u16(0);bool scalar_bad=false;size_t i=0;
  for(;i+8<=count;i+=8){uint16x8_t raw=vld1q_u16(src+i),mag=vandq_u16(raw,vdupq_n_u16(0x7fff));
    uint16x8_t exceptional=vcgeq_u16(mag,vdupq_n_u16(0x7bff));bad=vorrq_u16(bad,exceptional);
    uint16x8_t sub=vandq_u16(vcltq_u16(mag,vdupq_n_u16(0x400)),vcgtq_u16(mag,vdupq_n_u16(0)));
    uint16x8_t out;
    if(vmaxvq_u16(vorrq_u16(sub,exceptional)))out=float_half_bf16(raw);
    else{out=vaddq_u16(vshrq_n_u16(vaddq_u16(vaddq_u16(mag,vdupq_n_u16(3)),vandq_u16(vshrq_n_u16(mag,3),vdupq_n_u16(1))),3),vdupq_n_u16(0x3800));
      out=vbslq_u16(vceqq_u16(mag,vdupq_n_u16(0)),vdupq_n_u16(0),out);out=vorrq_u16(out,vandq_u16(raw,vdupq_n_u16(0x8000)));}
    vst1q_u16(dst+i,out);
  }
  for(;i<count;++i){__fp16 h;std::memcpy(&h,src+i,2);scalar_bad|=(src[i]&0x7fff)>=0x7bff;dst[i]=bf_round(static_cast<float>(h));}
  return scalar_bad||vmaxvq_u16(bad)?-3:0;
}
extern "C" int mm_convert_input(const uint16_t* src,uint16_t* dst,size_t count) {
  bool bad=false;size_t i=0;
  for(;i+8<=count;i+=8){uint16x8_t raw=vld1q_u16(src+i),mag=vandq_u16(raw,vdupq_n_u16(0x7fff));
    uint16x8_t normal=vandq_u16(vcgeq_u16(mag,vdupq_n_u16(0x3880)),vcltq_u16(mag,vdupq_n_u16(0x4780)));
    uint16x8_t valid=vorrq_u16(normal,vceqq_u16(mag,vdupq_n_u16(0)));uint16x8_t out;
    if(vminvq_u16(valid)){out=vshlq_n_u16(vsubq_u16(mag,vdupq_n_u16(0x3800)),3);out=vbslq_u16(vceqq_u16(mag,vdupq_n_u16(0)),vdupq_n_u16(0),out);out=vorrq_u16(out,vandq_u16(raw,vdupq_n_u16(0x8000)));}
    else{float32x4_t lo=vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(raw),16)),hi=vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(raw),16));out=vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(lo),vcvt_f16_f32(hi)));}
    bad|=vmaxvq_u16(vceqq_u16(vandq_u16(out,vdupq_n_u16(0x7c00)),vdupq_n_u16(0x7c00)))!=0;vst1q_u16(dst+i,out);
  }
  for(;i<count;++i){uint32_t u=uint32_t(src[i])<<16;float f;std::memcpy(&f,&u,4);__fp16 h=static_cast<__fp16>(f);std::memcpy(dst+i,&h,2);bad|=(dst[i]&0x7c00)==0x7c00;}
  return bad?-2:0;
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
  if (!p || rows<=0 || rows>640) { last_error="Invalid run argument"; return false; }
  int index=rows<=128?0:rows<=256?1:rows<=512?2:3;
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
// Conservatively reject the midpoint that rounds to the largest FP16 value.
// Preserve the old finite/saturation contract, while eliminating FP16 C rounding.
static int float_output(const float*src,uint16_t*dst,size_t count,const float*bias=nullptr,int width=0){
 bool bad=false;size_t i=0;
 if(!bias){
  for(;i+4<=count;i+=4){float32x4_t x=vld1q_f32(src+i);
   bad|=vmaxvq_u32(vorrq_u32(vmvnq_u32(vceqq_f32(x,x)),vcgeq_f32(vabsq_f32(x),vdupq_n_f32(65488.f))))!=0;
   uint32x4_t u=vreinterpretq_u32_f32(x);uint32x4_t r=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16);vst1_u16(dst+i,vmovn_u32(r));
  }
 }
 for(;i<count;i++){float raw=src[i];bad|=!std::isfinite(raw)||std::abs(raw)>=65488.f;float value=raw+(bias?bias[i%width]:0.f);bad|=!std::isfinite(value);dst[i]=bf_round(value);}
 if(bad){last_error="FP32 output outside original safe finite range";return -3;}return 0;
}
extern "C" int mm_run_prepared_bf16(void* handle, int rows, const float* bias, uint16_t* output) {
  auto* p=static_cast<Matrix*>(handle);
  if (!p || !output || rows<=0 || rows>640 || p->prepared_rows!=rows) { last_error="Input not prepared"; return -1; }
  if (!check(rknn_matmul_run(p->ctx),"run")) return -1;
  if(p->fp32)return float_output(static_cast<const float*>(p->c->virt_addr),output,size_t(rows)*p->n,bias,p->n);
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
  if(!p || rows<=0 || rows>640 || p->prepared_rows!=rows || segments<=0 ||
     !starts || !widths || !strides || !targets) {last_error="Invalid scatter arguments";return -1;}
  int covered=0;
  for(int s=0;s<segments;++s) {
    if(starts[s]!=covered || widths[s]<=0 || strides[s]<widths[s] || !targets[s] ||
       starts[s]+widths[s]>p->n) {last_error="Invalid scatter layout";return -1;}
    covered+=widths[s];
  }
  if(covered!=p->n) {last_error="Incomplete scatter layout";return -1;}
  if(!check(rknn_matmul_run(p->ctx),"run_scatter")) return -1;
  if(p->fp32){
    const float*source=static_cast<const float*>(p->c->virt_addr);int bad=0;
    for(int r=0;r<rows;r++)for(int s=0;s<segments;s++)bad|=float_output(source+size_t(r)*p->n+starts[s],targets[s]+size_t(r)*strides[s],widths[s],bias?bias+starts[s]:nullptr,widths[s]);
    return bad?-3:0;
  }
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

static int fuse_mlp_range(const uint16_t*gate,const uint16_t*up,uint16_t*out,const uint16_t*table,size_t count){
 int raw_bad=0,input_bad=0;size_t i=0;
 for(;i+8<=count;i+=8){
  uint16_t g[8],u[8],product[8],activation[8];
  raw_bad|=mm_fast_convert_output(gate+i,g,8)!=0;raw_bad|=mm_fast_convert_output(up+i,u,8)!=0;
  for(int j=0;j<8;j++)activation[j]=table[g[j]];
  for(int j=0;j<8;j+=4){float32x4_t a=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(activation+j),16)),b=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(u+j),16));
   uint32x4_t bits=vreinterpretq_u32_f32(vmulq_f32(a,b));uint32x4_t r=vshrq_n_u32(vaddq_u32(vaddq_u32(bits,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(1))),16);vst1_u16(product+j,vmovn_u32(r));
  }
  input_bad|=mm_convert_input(product,out+i,8)!=0;
 }
 for(;i<count;i++){
  __fp16 gh,uh;std::memcpy(&gh,gate+i,2);std::memcpy(&uh,up+i,2);raw_bad|=(gate[i]&0x7fff)>=0x7bff || (up[i]&0x7fff)>=0x7bff;
  uint16_t gb=bf_round(static_cast<float>(gh)),ub=bf_round(static_cast<float>(uh));uint32_t a=uint32_t(table[gb])<<16,b=uint32_t(ub)<<16;float af,bf;std::memcpy(&af,&a,4);std::memcpy(&bf,&b,4);uint16_t prod=bf_round(af*bf);input_bad|=mm_convert_input(&prod,out+i,1)!=0;
 }
 return raw_bad?-3:input_bad?-2:0;
}
extern "C" int mm_fuse_mlp_half(const uint16_t*gate,const uint16_t*up,uint16_t*out,const uint16_t*table,size_t count){
 int raw_bad=0,input_bad=0;
 #pragma omp parallel for num_threads(3) schedule(static) if(count>4096) reduction(|:raw_bad,input_bad)
 for(size_t block=0;block<(count+2047)/2048;block++){
  size_t start=block*2048,n=std::min(size_t(2048),count-start);int status=fuse_mlp_range(gate+start,up+start,out+start,table,n);raw_bad|=status==-3;input_bad|=status==-2;
 }
 return raw_bad?-3:input_bad?-2:0;
}
extern "C" int mm_run_raw(void*handle,int rows){
 auto*p=static_cast<Matrix*>(handle);if(!p||p->prepared_rows!=rows){last_error="Raw input not prepared";return -1;}
 return check(rknn_matmul_run(p->ctx),"run_raw")?0:-1;
}
static int fuse_mlp_float(const float*gate,const float*up,uint16_t*out,const uint16_t*table,size_t count){
 int raw_bad=0,input_bad=0;
 #pragma omp parallel for num_threads(3) schedule(static) if(count>4096) reduction(|:raw_bad,input_bad)
 for(size_t block=0;block<(count+2047)/2048;block++){
  size_t start=block*2048,n=std::min(size_t(2048),count-start);uint16_t g[2048],u[2048],product[2048];
  raw_bad|=float_output(gate+start,g,n)!=0;raw_bad|=float_output(up+start,u,n)!=0;
  size_t j=0;
  for(;j+4<=n;j+=4){uint16_t activation[4]={table[g[j]],table[g[j+1]],table[g[j+2]],table[g[j+3]]};
   float32x4_t a=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(activation),16)),b=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(u+j),16));uint32x4_t bits=vreinterpretq_u32_f32(vmulq_f32(a,b));uint32x4_t r=vshrq_n_u32(vaddq_u32(vaddq_u32(bits,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(bits,16),vdupq_n_u32(1))),16);vst1_u16(product+j,vmovn_u32(r));
  }
  for(;j<n;j++){uint32_t a=uint32_t(table[g[j]])<<16,b=uint32_t(u[j])<<16;float af,bf;std::memcpy(&af,&a,4);std::memcpy(&bf,&b,4);product[j]=bf_round(af*bf);}
  int status=mm_convert_input(product,out+start,n);input_bad|=status!=0;
 }
 return raw_bad?-3:input_bad?-2:0;
}
extern "C" int mm_prepare_mlp(void*down,void*gate,void*up,int rows,const uint16_t*table){
 auto*d=static_cast<Matrix*>(down);auto*g=static_cast<Matrix*>(gate);auto*u=static_cast<Matrix*>(up);
 if(!d||!g||!u||!table||g->n!=d->k||u->n!=d->k||g->prepared_rows!=rows||u->prepared_rows!=rows||!bind_shape(d,rows)){last_error="Invalid MLP pipeline shape";return -1;}
 d->prepared_rows=0;int status=(g->fp32||u->fp32)?(g->fp32&&u->fp32?fuse_mlp_float(static_cast<const float*>(g->c->virt_addr),static_cast<const float*>(u->c->virt_addr),static_cast<uint16_t*>(d->a->virt_addr),table,size_t(rows)*d->k):-1):mm_fuse_mlp_half(static_cast<uint16_t*>(g->c->virt_addr),static_cast<uint16_t*>(u->c->virt_addr),static_cast<uint16_t*>(d->a->virt_addr),table,size_t(rows)*d->k);
 if(status){last_error="MLP intermediate outside safe FP16 range";return status;}
 std::memset(static_cast<uint16_t*>(d->a->virt_addr)+size_t(rows)*d->k,0,size_t(d->shapes[d->selected].M-rows)*d->k*2);d->prepared_rows=rows;return 0;
}

extern "C" int mm_native_attrs(void*handle,int*out){auto*p=static_cast<Matrix*>(handle);if(!p)return -1;for(size_t shape=0;shape<p->attrs.size();shape++)for(int i=0;i<3;i++){auto&a=i==0?p->attrs[shape].A:i==1?p->attrs[shape].B:p->attrs[shape].C;int offset=(shape*3+i)*12;out[offset]=a.n_dims;out[offset+1]=a.size;out[offset+2]=a.type;for(unsigned j=0;j<a.n_dims&&j<9;j++)out[offset+3+j]=a.dims[j];}return p->attrs.size();}
