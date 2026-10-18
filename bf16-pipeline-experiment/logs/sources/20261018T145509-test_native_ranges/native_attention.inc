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
extern "C" int dm_attention_worker(void*qkh,void*pvh,const uint16_t*q,const uint16_t*k,const uint16_t*v,const uint8_t*mask,uint16_t*out,const int64_t*s,float scale,int kvheads,int first,int step,int residual){
 auto*qp=static_cast<Matrix*>(qkh);auto*vp=static_cast<Matrix*>(pvh);
 if(!qp||!vp||!q||!k||!v||!mask||!out||!s||kvheads<=0||8%kvheads||first<0||first>=8||step<1||!std::isfinite(scale)||(residual!=0&&residual!=1))return -1;
 if(qp->m!=615||qp->k!=256||qp->n!=624||vp->m!=615||vp->k!=640||vp->n!=256)return -1;
 bool any=false;for(int t=0;t<615;t++)any|=mask[t];if(!any)return -2;
 // Per-worker scratch is overwritten for every invocation; only storage is reused.
 // It is released with the persistent worker thread at Owner.close().
 thread_local std::vector<__fp16> qa(615*256),kb(256*624),vb(640*256),prob(615*640);
 thread_local std::vector<float> scores(615*624),output(615*256),low(615*256);
 int previous=-1;
 for(int h=first;h<8;h+=step){int kh=h/(8/kvheads);
  if(previous!=kh){
   std::fill(kb.begin(),kb.end(),0);std::fill(vb.begin(),vb.end(),0);
   for(int t=0;t<615;t++){
    const uint16_t*ks=k+kh*s[4]+t*s[5];__fp16 row[256];if(!pack_row(ks,row,256)||!pack_row(v+kh*s[7]+t*s[8],vb.data()+t*256,256))return -2;
    for(int d=0;d<256;d++)kb[d*624+t]=row[d];
   }
   if(dm_set_b(qkh,kb.data())||dm_set_b(pvh,vb.data()))return -1;
   previous=kh;
  }
  for(int t=0;t<615;t++)if(!pack_row(q+h*s[1]+t*s[2],qa.data()+t*256,256))return -2;
  if(dm_run(qkh,qa.data(),scores.data(),615))return -1;
  std::fill(prob.begin(),prob.end(),0);
  for(int t=0;t<615;t++){
   float*row=scores.data()+t*624;float m=-INFINITY;
   for(int j=0;j<615;j++){row[j]=mask[j]?row[j]*scale:-INFINITY;if(row[j]>m)m=row[j];}
   float sum=0;int j=0;
   for(;j+4<=615;j+=4){float32x4_t e=exp4(vsubq_f32(vld1q_f32(row+j),vdupq_n_f32(m)));vst1q_f32(row+j,e);sum+=vaddvq_f32(e);}
   for(;j<615;j++){row[j]=expf(row[j]-m);sum+=row[j];}
   if(!(sum>0)||!std::isfinite(sum))return -2;
   for(j=0;j<615;j++){row[j]/=sum;prob[t*640+j]=static_cast<__fp16>(row[j]);}
  }
  if(dm_run(pvh,prob.data(),output.data(),615))return -1;
  if(residual){
   for(int t=0;t<615;t++)for(int j=0;j<615;j++)prob[t*640+j]=static_cast<__fp16>(scores[t*624+j]-static_cast<float>(prob[t*640+j]));
   if(dm_run(pvh,prob.data(),low.data(),615))return -1;
  }
  for(int t=0;t<615;t++)for(int d=0;d<256;d++){size_t i=size_t(t)*256+d;float value=output[i]+(residual?low[i]:0.f);if(!std::isfinite(value))return -2;out[(size_t(t)*8+h)*256+d]=to_bf(value);}
 }
 return 0;
}

// Coalesce two independent query heads sharing the same K/V into one MatMul.
extern "C" int dm_attention_group_worker(void*qkh,void*pvh,const uint16_t*q,const uint16_t*k,const uint16_t*v,const uint8_t*mask,uint16_t*out,const int64_t*s,float scale,int kvheads,int first,int step,int residual){
 auto*qp=static_cast<Matrix*>(qkh);auto*vp=static_cast<Matrix*>(pvh);
 if(!qp||!vp||!q||!k||!v||!mask||!out||!s||kvheads<=0||8%kvheads||first<0||first>=8||step<1||!std::isfinite(scale)||(residual!=0&&residual!=1))return -1;
 if(kvheads!=2)return -1;
 if(qp->m!=1230||qp->k!=256||qp->n!=624||vp->m!=1230||vp->k!=640||vp->n!=256)return -1;
 bool any=false;for(int t=0;t<615;t++)any|=mask[t];if(!any)return -2;
 // Per-worker scratch is overwritten for every invocation; only storage is reused.
 // It is released with the persistent worker thread at Owner.close().
 thread_local std::vector<__fp16> qa(1230*256),kb(256*624),vb(640*256),prob(1230*640);
 thread_local std::vector<float> scores(1230*624),output(1230*256),low(1230*256);
 int previous=-1;
 for(int kh=0;kh<kvheads;kh++){
  int heads[8],count=0;for(int h=kh*4;h<(kh+1)*4;h++)if(h%step==first)heads[count++]=h;
  if(count!=2)return -1;
  if(previous!=kh){
   std::fill(kb.begin(),kb.end(),0);std::fill(vb.begin(),vb.end(),0);
   for(int t=0;t<615;t++){
    const uint16_t*ks=k+kh*s[4]+t*s[5];__fp16 row[256];if(!pack_row(ks,row,256)||!pack_row(v+kh*s[7]+t*s[8],vb.data()+t*256,256))return -2;
    for(int d=0;d<256;d++)kb[d*624+t]=row[d];
   }
   if(dm_set_b(qkh,kb.data())||dm_set_b(pvh,vb.data()))return -1;
   previous=kh;
  }
  for(int row=0;row<1230;row++)if(!pack_row(q+heads[row/615]*s[1]+(row%615)*s[2],qa.data()+row*256,256))return -2;
  if(dm_run(qkh,qa.data(),scores.data(),1230))return -1;
  std::fill(prob.begin(),prob.end(),0);
  for(int t=0;t<1230;t++){
   float*row=scores.data()+t*624;float m=-INFINITY;
   for(int j=0;j<615;j++){row[j]=mask[j]?row[j]*scale:-INFINITY;if(row[j]>m)m=row[j];}
   float sum=0;int j=0;
   for(;j+4<=615;j+=4){float32x4_t e=exp4(vsubq_f32(vld1q_f32(row+j),vdupq_n_f32(m)));vst1q_f32(row+j,e);sum+=vaddvq_f32(e);}
   for(;j<615;j++){row[j]=expf(row[j]-m);sum+=row[j];}
   if(!(sum>0)||!std::isfinite(sum))return -2;
   for(j=0;j<615;j++){row[j]/=sum;prob[t*640+j]=static_cast<__fp16>(row[j]);}
  }
  if(dm_run(pvh,prob.data(),output.data(),1230))return -1;
  if(residual){
   for(int t=0;t<1230;t++)for(int j=0;j<615;j++)prob[t*640+j]=static_cast<__fp16>(scores[t*624+j]-static_cast<float>(prob[t*640+j]));
   if(dm_run(pvh,prob.data(),low.data(),1230))return -1;
  }
  for(int t=0;t<1230;t++)for(int d=0;d<256;d++){int h=heads[t/615];size_t i=size_t(t)*256+d;float value=output[i]+(residual?low[i]:0.f);if(!std::isfinite(value))return -2;out[(size_t(t%615)*8+h)*256+d]=to_bf(value);}
 }
 return 0;
}
