/* Own NEON implementations of the same zero-state recurrent Delta equations. */
#include <arm_neon.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
static inline uint16x4_t rb(float32x4_t v){uint32x4_t u=vreinterpretq_u32_f32(v);uint32x4_t r=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16);r=vbslq_u32(vcgtq_u32(vandq_u32(u,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000)),vdupq_n_u32(0x7fc0),r);return vmovn_u32(r);}
int d3_scan_fast(const float *q,const float *k,const float *v,const float *decay,const float *beta,uint16_t *out,int heads,int length,int kd,int vd,int mode,int threads){
 if(heads<=0||length<=0||kd<=0||kd>128||vd<=0||vd>128||vd%4||(mode!=6&&mode!=7)||threads<1||threads>3)return -1;
 int tiles=(vd+15)/16;
 #pragma omp parallel for num_threads(threads) schedule(static)
 for(int job=0;job<heads*tiles;++job){
  int h=job/tiles,first=job%tiles*16,width=vd-first;if(width>16)width=16;
  float state[128*16] __attribute__((aligned(64)));memset(state,0,kd*width*sizeof(float));
  for(int t=0;t<length;++t){
   int row=h*length+t;const float *qt=q+row*kd,*kt=k+row*kd,*vt=v+row*vd+first;
   float d=decay[row],b=beta[row],qk=0;
   if(mode==7)for(int i=0;i<kd;++i)qk+=qt[i]*kt[i];
   float32x4_t p[4],o[4],delta[4];int groups=width/4;
   for(int j=0;j<groups;++j)p[j]=o[j]=vdupq_n_f32(0);
   for(int i=0;i<kd;++i)for(int j=0;j<groups;++j){float32x4_t ss=vld1q_f32(state+i*width+j*4);p[j]=vfmaq_n_f32(p[j],ss,kt[i]);if(mode==7)o[j]=vfmaq_n_f32(o[j],ss,qt[i]);}
   for(int j=0;j<groups;++j)delta[j]=vmulq_n_f32(vsubq_f32(vld1q_f32(vt+j*4),vmulq_n_f32(p[j],d)),b);
   for(int i=0;i<kd;++i)for(int j=0;j<groups;++j){float *s=state+i*width+j*4;float32x4_t ss=vfmaq_n_f32(vmulq_n_f32(vld1q_f32(s),d),delta[j],kt[i]);vst1q_f32(s,ss);if(mode==6)o[j]=vfmaq_n_f32(o[j],ss,qt[i]);}
   for(int j=0;j<groups;++j){if(mode==7)o[j]=vfmaq_n_f32(vmulq_n_f32(o[j],d),delta[j],qk);vst1_u16(out+row*vd+first+j*4,rb(o[j]));}
  }
 }
 return 0;
}
