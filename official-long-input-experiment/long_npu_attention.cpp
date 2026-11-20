// Derived from own dynamic_matmul.cpp 37a7a09e25381bc19eb37214ac2d4ef6eaf2f0eaa0c8a623f459d087092efe6d
// and native_attention.inc cfa0e10e62797b371842dfa0421e718048173d11288c594ba03ec36a34163021.
// Bounded query tiles (up to2048 rows); every query reads all keys; scaled residual P correction.
// Own activation-to-activation MatMul wrapper, using only public RKNN API.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include "rknn_matmul_api.h"
static thread_local std::string error;
struct LongMatrix {
 rknn_matmul_ctx ctx=0; rknn_tensor_mem *a=nullptr,*b=nullptr,*c=nullptr;
 rknn_matmul_info info{}; rknn_matmul_io_attr attr{};
 int m,k,n;bool ready=false;
 ~LongMatrix(){if(a)rknn_destroy_mem(ctx,a);if(b)rknn_destroy_mem(ctx,b);if(c)rknn_destroy_mem(ctx,c);if(ctx)rknn_matmul_destroy(ctx);}
};
static bool check(int code,const char*op){if(!code)return true;error=std::string(op)+" returned "+std::to_string(code);return false;}
extern "C" const char* la_error(){return error.c_str();}
extern "C" void* la_create(int m,int k,int n,int core){
 error.clear();if((m!=128&&m!=512&&m!=1024&&m!=2048)||(core!=1&&core!=2)||!((k==256&&n>615&&n<=8192&&n%128==0)||(n==256&&k>615&&k<=8192&&k%128==0))){error="Invalid aligned matrix shape";return nullptr;}
 auto p=std::make_unique<LongMatrix>();p->m=m;p->k=k;p->n=n;
 p->info.M=m;p->info.K=k;p->info.N=n;p->info.type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;p->info.B_layout=RKNN_MM_LAYOUT_NATIVE;p->info.AC_layout=RKNN_MM_LAYOUT_NORM;
 if(!check(rknn_matmul_create(&p->ctx,&p->info,&p->attr),"create"))return nullptr;
 if(!check(rknn_matmul_set_core_mask(p->ctx,static_cast<rknn_core_mask>(core)),"set_core"))return nullptr;
 p->a=rknn_create_mem(p->ctx,p->attr.A.size);p->b=rknn_create_mem(p->ctx,p->attr.B.size);p->c=rknn_create_mem(p->ctx,p->attr.C.size);
 if(!p->a||!p->b||!p->c){error="allocation failed";return nullptr;}
 if(!check(rknn_matmul_set_io_mem(p->ctx,p->a,&p->attr.A),"bind A")||!check(rknn_matmul_set_io_mem(p->ctx,p->b,&p->attr.B),"bind B")||!check(rknn_matmul_set_io_mem(p->ctx,p->c,&p->attr.C),"bind C"))return nullptr;
 return p.release();
}
extern "C" int la_set_b(void*handle,const void*data){
 auto*p=static_cast<LongMatrix*>(handle);if(!p||!data)return -1;p->ready=false;
 if(!check(rknn_B_normal_layout_to_native_layout(const_cast<void*>(data),p->b->virt_addr,p->k,p->n,&p->info),"pack B"))return -1;
 if(!check(rknn_mem_sync(p->ctx,p->b,RKNN_MEMORY_SYNC_TO_DEVICE),"sync B")||!check(rknn_matmul_set_io_mem(p->ctx,p->b,&p->attr.B),"rebind B"))return -1;
 p->ready=true;return 0;
}
extern "C" int la_run(void*handle,const void*data,float*out,int rows){
 auto*p=static_cast<LongMatrix*>(handle);if(!p||!data||!out||!p->ready||rows<=0||rows>p->m)return -1;
 std::memcpy(p->a->virt_addr,data,size_t(rows)*p->k*2);
 std::memset(static_cast<char*>(p->a->virt_addr)+size_t(rows)*p->k*2,0,size_t(p->m-rows)*p->k*2);
 if(!check(rknn_matmul_run(p->ctx),"run"))return -1;
 std::memcpy(out,p->c->virt_addr,size_t(rows)*p->n*4);return 0;
}
extern "C" void la_destroy(void*handle){delete static_cast<LongMatrix*>(handle);}


// Fused CPU preparation/softmax around FP32-output dynamic NPU MatMul.
#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <vector>
static float from_bf(uint16_t x){uint32_t bits=uint32_t(x)<<16;float value;std::memcpy(&value,&bits,4);return value;}
static uint16_t to_bf(float x){uint32_t u;std::memcpy(&u,&x,4);return (u+0x7fff+((u>>16)&1))>>16;}
static float32x4_t exp4(float32x4_t x){
 if(vmaxvq_f32(vabsq_f32(x))>80.0f){float a[4];vst1q_f32(a,x);for(int i=0;i<4;i++)a[i]=expf(a[i]);return vld1q_f32(a);}
 int32x4_t n=vcvtnq_s32_f32(vmulq_n_f32(x,1.4426950408889634f));float32x4_t nf=vcvtq_f32_s32(n);
 float32x4_t z=vfmsq_n_f32(x,nf,0.693145751953125f);z=vfmsq_n_f32(z,nf,1.428606765330187e-6f);
 float32x4_t p=vdupq_n_f32(1.0f/5040);p=vfmaq_f32(vdupq_n_f32(1.0f/720),p,z);p=vfmaq_f32(vdupq_n_f32(1.0f/120),p,z);p=vfmaq_f32(vdupq_n_f32(1.0f/24),p,z);p=vfmaq_f32(vdupq_n_f32(1.0f/6),p,z);p=vfmaq_f32(vdupq_n_f32(.5),p,z);p=vfmaq_f32(vdupq_n_f32(1),p,z);p=vfmaq_f32(vdupq_n_f32(1),p,z);
 return vmulq_f32(p,vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n,vdupq_n_s32(127)),23)));
}
static bool pack_row(const uint16_t*src,__fp16*dst,int n){
 int i=0;uint16x4_t bad=vdup_n_u16(0);
 for(;i+4<=n;i+=4){float32x4_t x=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(src+i),16));float16x4_t h=vcvt_f16_f32(x);bad=vorr_u16(bad,vcge_u16(vand_u16(vreinterpret_u16_f16(h),vdup_n_u16(0x7fff)),vdup_n_u16(0x7c00)));vst1_f16(dst+i,h);}
 if(vmaxv_u16(bad))return false;
 for(;i<n;i++){dst[i]=static_cast<__fp16>(from_bf(src[i]));if(!std::isfinite(static_cast<float>(dst[i])))return false;}return true;
}
extern "C" int la_long_worker(void*qkh,void*pvh,const uint16_t*q,const uint16_t*k,const uint16_t*v,const uint8_t*mask,uint16_t*out,const int64_t*s,float scale,int batch,int heads,int kvheads,int length,int first,int step,int correction){
 auto*qp=static_cast<LongMatrix*>(qkh);auto*vp=static_cast<LongMatrix*>(pvh);
 if(!qp||!vp||!q||!k||!v||!mask||!out||!s||batch<1||batch>8||heads<1||heads>8||batch*heads>64||kvheads<1||heads%kvheads||first<0||first>=step||step!=2||!std::isfinite(scale)||(correction!=0&&correction!=1))return -1;
 int bucket=qp->n,tile=qp->m;
 if((tile!=128&&tile!=512&&tile!=1024&&tile!=2048)||qp->k!=256||bucket<=615||bucket>8192||bucket%128||vp->m!=tile||vp->k!=bucket||vp->n!=256||length<=615||length>bucket)return -1;
 if(size_t(batch)*heads*length*256*2>(128ULL<<20))return -1;
 for(int i=0;i<9;i++)if(s[i]<0)return -1;
 for(int b=0;b<batch;b++){bool any=false;for(int j=0;j<length;j++)any|=mask[b*length+j];if(!any)return -2;}
 try{
  // At the maximum bucket, live scratch is <128MiB per worker. All vectors
  // are local and released on success and failure; no T-by-T workspace.
  std::vector<__fp16> qa(size_t(tile)*256),kb(size_t(256)*bucket),vb(size_t(bucket)*256),prob(size_t(tile)*bucket);
  std::vector<float> scores(size_t(tile)*bucket),high(size_t(tile)*256),low(size_t(tile)*256);
  int previous_b=-1,previous_kh=-1;
  for(int bh=first;bh<batch*heads;bh+=step){
   int b=bh/heads,h=bh%heads,kh=h/(heads/kvheads);
   if(previous_b!=b||previous_kh!=kh){
    for(int t=0;t<length;t++){
     __fp16 row[256];
     if(!pack_row(k+b*s[3]+kh*s[4]+t*s[5],row,256)||!pack_row(v+b*s[6]+kh*s[7]+t*s[8],vb.data()+size_t(t)*256,256))return -2;
     for(int d=0;d<256;d++)kb[size_t(d)*bucket+t]=row[d];
    }
    if(la_set_b(qkh,kb.data())||la_set_b(pvh,vb.data()))return -1;
    previous_b=b;previous_kh=kh;
   }
   for(int start=0;start<length;start+=tile){
    int rows=std::min(tile,length-start);
    for(int i=0;i<rows;i++)if(!pack_row(q+b*s[0]+h*s[1]+(start+i)*s[2],qa.data()+i*256,256))return -2;
    if(la_run(qkh,qa.data(),scores.data(),rows))return -1;
    int bad_probability=0;
    #pragma omp parallel for if(length>4096) num_threads(2) schedule(static) reduction(|:bad_probability)
    for(int i=0;i<rows;i++){
     float*row=scores.data()+size_t(i)*bucket;float maximum=-INFINITY;
     for(int j=0;j<length;j++){row[j]=mask[b*length+j]?row[j]*scale:-INFINITY;if(row[j]>maximum)maximum=row[j];}
     float sum=0;int j=0;
     for(;j+4<=length;j+=4){float32x4_t e=exp4(vsubq_f32(vld1q_f32(row+j),vdupq_n_f32(maximum)));vst1q_f32(row+j,e);sum+=vaddvq_f32(e);}
     for(;j<length;j++){row[j]=expf(row[j]-maximum);sum+=row[j];}
     if(!(sum>0)||!std::isfinite(sum)){bad_probability|=1;continue;}
     float inverse=1/sum,prob_scale=correction?1.f:1024.f;j=0;
     for(;j+4<=length;j+=4){float32x4_t p=vmulq_n_f32(vld1q_f32(row+j),inverse);vst1q_f32(row+j,p);vst1_f16(prob.data()+size_t(i)*bucket+j,vcvt_f16_f32(vmulq_n_f32(p,prob_scale)));}
     for(;j<length;j++){row[j]*=inverse;prob[size_t(i)*bucket+j]=static_cast<__fp16>(row[j]*prob_scale);}
    }
    if(bad_probability)return -2;
    if(la_run(pvh,prob.data(),high.data(),rows))return -1;
    if(correction){
    // Scale the residual before FP16 conversion so small probabilities do
    // not lose their entire correction in the FP16 subnormal range.
    for(int i=0;i<rows;i++){
     float*row=scores.data()+size_t(i)*bucket;__fp16*p=prob.data()+size_t(i)*bucket;int j=0;
     for(;j+4<=length;j+=4){float32x4_t delta=vmulq_n_f32(vsubq_f32(vld1q_f32(row+j),vcvt_f32_f16(vld1_f16(p+j))),1024.f);vst1_f16(p+j,vcvt_f16_f32(delta));}
     for(;j<length;j++)p[j]=static_cast<__fp16>((row[j]-static_cast<float>(p[j]))*1024.f);
    }
    if(la_run(pvh,prob.data(),low.data(),rows))return -1;
    }
    for(int i=0;i<rows;i++)for(int d=0;d<256;d++){
     size_t at=size_t(i)*256+d;float result=correction?high[at]+low[at]*(1.f/1024.f):high[at]*(1.f/1024.f);
     if(!std::isfinite(result))return -2;
     out[((size_t(b)*length+start+i)*heads+h)*256+d]=to_bf(result);
    }
   }
  }
 }catch(const std::exception& exception){error=std::string("long attention scratch: ")+exception.what();return -1;}
 return 0;
}

// Medium-domain experiment mirroring the pinned PyTorch CPU-Flash reduced-
// precision boundary: unnormalised exp is BF16 before PV, FP32 online state,
// 512-key blocks, final FP32 denominator. Reference (no vendor internals):
// https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp
extern "C" void* la_create_flash(int k,int n,int core){
 if((core!=1&&core!=2)||!((k==256&&n==512)||(k==512&&n==256)))return nullptr;
 auto p=std::make_unique<LongMatrix>();p->m=128;p->k=k;p->n=n;
 p->info.M=128;p->info.K=k;p->info.N=n;p->info.type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;p->info.B_layout=RKNN_MM_LAYOUT_NATIVE;p->info.AC_layout=RKNN_MM_LAYOUT_NORM;
 if(!check(rknn_matmul_create(&p->ctx,&p->info,&p->attr),"flash create"))return nullptr;
 if(!check(rknn_matmul_set_core_mask(p->ctx,static_cast<rknn_core_mask>(core)),"flash core"))return nullptr;
 p->a=rknn_create_mem(p->ctx,p->attr.A.size);p->b=rknn_create_mem(p->ctx,p->attr.B.size);p->c=rknn_create_mem(p->ctx,p->attr.C.size);
 if(!p->a||!p->b||!p->c)return nullptr;
 if(!check(rknn_matmul_set_io_mem(p->ctx,p->a,&p->attr.A),"flash A")||!check(rknn_matmul_set_io_mem(p->ctx,p->b,&p->attr.B),"flash B")||!check(rknn_matmul_set_io_mem(p->ctx,p->c,&p->attr.C),"flash C"))return nullptr;
 return p.release();
}
extern "C" __attribute__((optimize("fp-contract=off"))) int la_flash_worker(void*const*qkh,void*const*pvh,int blocks,
 const uint16_t*q,const uint16_t*k,const uint16_t*v,const uint8_t*mask,uint16_t*out,const int64_t*s,
 float scale,int batch,int heads,int kvheads,int length,int first,int step){
 if(!qkh||!pvh||!q||!k||!v||!mask||!out||!s||batch<1||batch>8||heads<1||heads>8||batch*heads>64||kvheads<1||heads%kvheads||length<=615||length>2048||blocks!=(length+511)/512||first<0||first>=step||step!=2||!std::isfinite(scale))return -1;
 for(int j=0;j<blocks;j++){
  auto*a=static_cast<LongMatrix*>(qkh[j]);auto*b=static_cast<LongMatrix*>(pvh[j]);
  if(!a||!b||a->m!=128||a->k!=256||a->n!=512||b->m!=128||b->k!=512||b->n!=256)return -1;
 }
 for(int j=0;j<9;j++)if(s[j]<0)return -1;
 for(int b=0;b<batch;b++){bool any=false;for(int j=0;j<length;j++)any|=mask[b*length+j];if(!any)return -2;}
 try{
  std::vector<__fp16> qa(128*256),kb(256*512),vb(512*256),prob(128*512);
  std::vector<float> scores(128*512),pv(128*256),acc(128*256),sums(128),maximums(128);
  int previous_b=-1,previous_kh=-1;
  for(int bh=first;bh<batch*heads;bh+=step){
   int b=bh/heads,h=bh%heads,kh=h/(heads/kvheads);
   if(previous_b!=b||previous_kh!=kh){
    for(int block=0;block<blocks;block++){
     std::fill(kb.begin(),kb.end(),__fp16(0));std::fill(vb.begin(),vb.end(),__fp16(0));int begin=block*512,used=std::min(512,length-begin);
     for(int j=0;j<used;j++){
      __fp16 key[256];
      if(!pack_row(k+b*s[3]+kh*s[4]+(begin+j)*s[5],key,256)||!pack_row(v+b*s[6]+kh*s[7]+(begin+j)*s[8],vb.data()+j*256,256))return -2;
      for(int d=0;d<256;d++)kb[d*512+j]=key[d];
     }
     if(la_set_b(qkh[block],kb.data())||la_set_b(pvh[block],vb.data()))return -1;
    }
    previous_b=b;previous_kh=kh;
   }
   for(int begin_q=0;begin_q<length;begin_q+=128){
    int rows=std::min(128,length-begin_q);
    for(int r=0;r<rows;r++)if(!pack_row(q+b*s[0]+h*s[1]+(begin_q+r)*s[2],qa.data()+r*256,256))return -2;
    std::fill(acc.begin(),acc.end(),0.f);std::fill(sums.begin(),sums.end(),0.f);std::fill(maximums.begin(),maximums.end(),-INFINITY);
    for(int block=0;block<blocks;block++){
     int begin_k=block*512,used=std::min(512,length-begin_k);
     if(la_run(qkh[block],qa.data(),scores.data(),rows))return -1;
     for(int r=0;r<rows;r++){
      float*row=scores.data()+r*512;float mx=maximums[r];
      for(int j=0;j<512;j++){row[j]=(j<used&&mask[b*length+begin_k+j])?row[j]*scale:-INFINITY;mx=std::max(mx,row[j]);}
      if(mx==-INFINITY){std::fill(prob.begin()+r*512,prob.begin()+(r+1)*512,__fp16(0));continue;}
      float old_scale=maximums[r]==-INFINITY?0.f:expf(maximums[r]-mx);float32x4_t sum=vdupq_n_f32(0);
      for(int j=0;j<512;j+=4){
       float32x4_t e=exp4(vsubq_f32(vld1q_f32(row+j),vdupq_n_f32(mx)));sum=vaddq_f32(sum,e);
       float values[4];vst1q_f32(values,e);
       for(int z=0;z<4;z++)prob[r*512+j+z]=static_cast<__fp16>(from_bf(to_bf(values[z]))*1024.f);
      }
      sums[r]=vaddvq_f32(sum)+old_scale*sums[r];maximums[r]=mx;
      for(int d=0;d<256;d++)acc[r*256+d]*=old_scale;
     }
     if(la_run(pvh[block],prob.data(),pv.data(),rows))return -1;
     for(int r=0;r<rows;r++)for(int d=0;d<256;d++)acc[r*256+d]+=pv[r*256+d]*(1.f/1024.f);
    }
    for(int r=0;r<rows;r++){
     float inverse=sums[r]>0?1.f/sums[r]:1.f;
     for(int d=0;d<256;d++){
      float result=acc[r*256+d]*inverse;if(!std::isfinite(result))return -2;
      out[((size_t(b)*length+begin_q+r)*heads+h)*256+d]=to_bf(result);
     }
    }
   }
  }
 }catch(const std::exception&ex){error=std::string("medium flash scratch: ")+ex.what();return -1;}
 return 0;
}
