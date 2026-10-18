/* Own FP32 GQA attention using the public BLAS interface, no causal masking. */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef void(*sgemm_t)(int,int,int,int64_t,int64_t,int64_t,float,const float*,int64_t,const float*,int64_t,float,float*,int64_t);
static sgemm_t gemm=0;
void d3_set_sgemm(void*fn){gemm=(sgemm_t)fn;}
static inline float bf(uint16_t x){uint32_t u=(uint32_t)x<<16;float f;memcpy(&f,&u,4);return f;}
static inline uint16_t rb(float f){uint32_t u;memcpy(&u,&f,4);if((u&0x7fffffff)>0x7f800000)return (u>>16)|0x40;return (u+0x7fff+((u>>16)&1))>>16;}
static inline float32x4_t exp4(float32x4_t x){
 if(vmaxvq_f32(vabsq_f32(x))>80.0f){float a[4];vst1q_f32(a,x);for(int i=0;i<4;i++)a[i]=expf(a[i]);return vld1q_f32(a);}
 int32x4_t n=vcvtnq_s32_f32(vmulq_n_f32(x,1.4426950408889634f));float32x4_t nf=vcvtq_f32_s32(n);
 float32x4_t z=vfmsq_n_f32(x,nf,0.693145751953125f);z=vfmsq_n_f32(z,nf,1.428606765330187e-6f);
 float32x4_t p=vdupq_n_f32(1.0f/5040);p=vfmaq_f32(vdupq_n_f32(1.0f/720),p,z);p=vfmaq_f32(vdupq_n_f32(1.0f/120),p,z);p=vfmaq_f32(vdupq_n_f32(1.0f/24),p,z);p=vfmaq_f32(vdupq_n_f32(1.0f/6),p,z);p=vfmaq_f32(vdupq_n_f32(.5),p,z);p=vfmaq_f32(vdupq_n_f32(1),p,z);p=vfmaq_f32(vdupq_n_f32(1),p,z);
 return vmulq_f32(p,vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n,vdupq_n_s32(127)),23)));
}
/* Every query tile still attends to ALL keys. GQA K/V conversion is shared. */
int d3_attention_tiled(const uint16_t*q,const uint16_t*k,const uint16_t*v,const uint8_t*mask,uint16_t*out,const int64_t*strides,int batch,int heads,int kvheads,int length,int dim,float scale,int tile,int threads,int reciprocal){
 if(!gemm||batch<1||batch>8||heads<1||heads>64||batch*heads>128||kvheads<1||heads%kvheads||length<1||length>2048||dim<1||dim>256||tile<8||tile>2048||threads<1||threads>3||(reciprocal!=0&&reciprocal!=1))return -1;
 size_t size=(size_t)length*dim,qsize=(size_t)batch*heads*size,ksize=(size_t)batch*kvheads*size;
 float*packed=malloc((qsize+2*ksize)*sizeof(float));if(!packed)return -2;
 float *qq=packed,*kk=qq+qsize,*vv=kk+ksize;int bad=0;
 int allheads=heads+2*kvheads;
 #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:bad)
 for(int row=0;row<batch*allheads*length;row++){
  int t=row%length,ph=(row/length)%allheads,b=row/(length*allheads),h,base;
  const uint16_t*src;float*dst;
  if(ph<heads){h=ph;base=0;src=q;dst=qq+((size_t)b*heads+h)*size+(size_t)t*dim;}
  else if(ph<heads+kvheads){h=ph-heads;base=3;src=k;dst=kk+((size_t)b*kvheads+h)*size+(size_t)t*dim;}
  else{h=ph-heads-kvheads;base=6;src=v;dst=vv+((size_t)b*kvheads+h)*size+(size_t)t*dim;}
  src+=b*strides[base]+h*strides[base+1]+t*strides[base+2];int d=0;
  for(;d+4<=dim;d+=4){uint32x4_t bits=vshll_n_u16(vld1_u16(src+d),16);if(vmaxvq_u32(vcgeq_u32(vandq_u32(bits,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000))))bad|=1;vst1q_f32(dst+d,vreinterpretq_f32_u32(bits));}
  for(;d<dim;d++){dst[d]=bf(src[d]);if(!isfinite(dst[d]))bad|=1;}
 }
 if(bad){free(packed);return 1;}
 int blocks=(length+tile-1)/tile;
 #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:bad)
 for(int job=0;job<batch*heads*blocks;job++){
  int bh=job/blocks,b=bh/heads,h=bh%heads,kh=h/(heads/kvheads),start=(job%blocks)*tile,n=length-start;if(n>tile)n=tile;
  float*buf=malloc(((size_t)n*length+(size_t)n*dim)*sizeof(float));if(!buf){bad|=1;continue;}
  float*scores=buf,*yy=buf+(size_t)n*length;int invalid=0;
  gemm(101,111,112,n,length,dim,scale,qq+(size_t)bh*size+(size_t)start*dim,dim,kk+((size_t)b*kvheads+kh)*size,dim,0,scores,length);
  for(int i=0;i<n;i++){
   float*s=scores+(size_t)i*length,m=-INFINITY;
   for(int j=0;j<length;j++){if(mask&&!mask[b*length+j])s[j]=-INFINITY;if(s[j]>m)m=s[j];}
   float sum=0;int j=0;
   for(;j+4<=length;j+=4){float32x4_t e=exp4(vsubq_f32(vld1q_f32(s+j),vdupq_n_f32(m)));vst1q_f32(s+j,e);sum+=vaddvq_f32(e);}
   for(;j<length;j++){s[j]=expf(s[j]-m);sum+=s[j];}
   if(!(sum>0)||!isfinite(sum)){invalid=1;break;}
   if(reciprocal){float inv=1/sum;j=0;for(;j+4<=length;j+=4)vst1q_f32(s+j,vmulq_n_f32(vld1q_f32(s+j),inv));for(;j<length;j++)s[j]*=inv;}
   else for(j=0;j<length;j++)s[j]/=sum;
  }
  if(invalid){bad|=1;free(buf);continue;}
  gemm(101,111,111,n,dim,length,1,scores,length,vv+((size_t)b*kvheads+kh)*size,dim,0,yy,dim);
  for(int i=0;i<n;i++)for(int d=0;d<dim;d++){float f=yy[(size_t)i*dim+d];if(!isfinite(f))bad|=1;out[((size_t)b*length+start+i)*heads*dim+h*dim+d]=rb(f);}
  free(buf);
 }
 free(packed);return bad;
}
