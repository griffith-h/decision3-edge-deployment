/* Own FP32 GQA attention using the public BLAS interface, no causal masking. */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
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
int d3_attention(const uint16_t*q,const uint16_t*k,const uint16_t*v,const uint8_t*mask,uint16_t*out,const int64_t*strides,int batch,int heads,int kvheads,int length,int dim,float scale,int mode,double*profile){
 if(!gemm||batch<1||heads<1||kvheads<1||heads%kvheads||length<1||length>2048||dim<1||dim>256||batch>8||heads>64||batch*heads>128)return -1;
 int bad=0;
 #pragma omp parallel for num_threads(3) schedule(static) reduction(|:bad)
 for(int bh=0;bh<batch*heads;bh++){
  double t0=profile?omp_get_wtime():0,marks[6]={0};
  int b=bh/heads,h=bh%heads,kh=h/(heads/kvheads);size_t size=(size_t)length*dim;
  float*buf=(float*)malloc((4*size+(size_t)length*length)*sizeof(float));if(!buf){bad|=1;continue;}
  float*qq=buf,*kk=qq+size,*vv=kk+size,*yy=vv+size,*scores=yy+size;int invalid=0;
  for(int t=0;t<length;t++)for(int d=0;d<dim;d++){
   qq[t*dim+d]=bf(q[b*strides[0]+h*strides[1]+t*strides[2]+d]);
   kk[t*dim+d]=bf(k[b*strides[3]+kh*strides[4]+t*strides[5]+d]);
   vv[t*dim+d]=bf(v[b*strides[6]+kh*strides[7]+t*strides[8]+d]);
   if(!isfinite(qq[t*dim+d])||!isfinite(kk[t*dim+d])||!isfinite(vv[t*dim+d]))invalid=1;
  }
  if(invalid){bad|=1;free(buf);continue;}
  if(profile){marks[0]=omp_get_wtime()-t0;t0=omp_get_wtime();}
  gemm(101,111,112,length,length,dim,scale,qq,dim,kk,dim,0,scores,length);
  if(profile){marks[1]=omp_get_wtime()-t0;t0=omp_get_wtime();}
  for(int t=0;t<length;t++){
   float*s=scores+t*length,m=-INFINITY;
   for(int j=0;j<length;j++){if(mask&&!mask[b*length+j])s[j]=-INFINITY;if(s[j]>m)m=s[j];}
   float sum=0;int j=0;
   if(mode){for(;j+4<=length;j+=4){float32x4_t z=vsubq_f32(vld1q_f32(s+j),vdupq_n_f32(m));float32x4_t e=exp4(z);vst1q_f32(s+j,e);sum+=vaddvq_f32(e);}}
   for(;j<length;j++){s[j]=expf(s[j]-m);sum+=s[j];}
   if(!(sum>0)||!isfinite(sum)){invalid=1;break;}
   for(j=0;j<length;j++)s[j]/=sum;
  }
  if(invalid){bad|=1;free(buf);continue;}
  if(profile){marks[2]=omp_get_wtime()-t0;t0=omp_get_wtime();}
  gemm(101,111,111,length,dim,length,1,scores,length,vv,dim,0,yy,dim);
  if(profile){marks[3]=omp_get_wtime()-t0;t0=omp_get_wtime();}
  for(int t=0;t<length;t++)for(int d=0;d<dim;d++){float f=yy[t*dim+d];if(!isfinite(f))bad|=1;out[((size_t)b*length+t)*heads*dim+h*dim+d]=rb(f);}
  if(profile){marks[4]=omp_get_wtime()-t0;for(int i=0;i<5;i++)profile[bh*5+i]=marks[i];}
  free(buf);
 }
 return bad;
}
