/* Own CPU kernels: lookup exactly the finite BF16 input domain, original rounding. */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
static inline float bf(uint16_t x){uint32_t u=(uint32_t)x<<16;float f;memcpy(&f,&u,4);return f;}
static inline uint16_t rb(float f){uint32_t u;memcpy(&u,&f,4);if((u&0x7fffffff)>0x7f800000)return (u>>16)|0x40;return (u+0x7fff+((u>>16)&1))>>16;}
static inline float32x4_t bf4(const uint16_t*x){return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(x),16));}
static inline uint16x4_t rb4(float32x4_t v){uint32x4_t u=vreinterpretq_u32_f32(v);uint32x4_t r=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16);r=vbslq_u32(vcgtq_u32(vandq_u32(u,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000)),vorrq_u32(vshrq_n_u32(u,16),vdupq_n_u32(0x40)),r);return vmovn_u32(r);}
void d3_gate_table(float*out){for(int i=0;i<65536;i++){float g=bf(i);out[i]=g/(1.0f+expf(-g));}}
void d3_lut(const uint16_t*x,const uint16_t*up,uint16_t*out,const uint16_t*table,int n){
 #pragma omp parallel for num_threads(3) schedule(static) if(n>4096)
 for(int c=0;c<(n+2047)/2048;c++){
  int start=c*2048,end=start+2048;if(end>n)end=n;int i=start;
  if(!up){for(;i<end;i++)out[i]=table[x[i]];}
  else{
   for(;i+4<=end;i+=4){uint16_t v[4]={table[x[i]],table[x[i+1]],table[x[i+2]],table[x[i+3]]};vst1_u16(out+i,rb4(vmulq_f32(bf4(v),bf4(up+i))));}
   for(;i<end;i++)out[i]=rb(bf(table[x[i]])*bf(up[i]));
  }
 }
}
void d3_norm_lut(const uint16_t*x,const uint16_t*w,const uint16_t*gate,uint16_t*out,const float*table,int rows,int dim,float eps,int kind){
 #pragma omp parallel for num_threads(3) schedule(static) if(rows>16)
 for(int r=0;r<rows;r++){
  const uint16_t*in=x+(size_t)r*dim;uint16_t*dst=out+(size_t)r*dim;const uint16_t*g=gate?gate+(size_t)r*dim:0;
  float32x4_t acc=vdupq_n_f32(0);int i=0;
  for(;i+4<=dim;i+=4){float32x4_t v=bf4(in+i);acc=vaddq_f32(acc,vmulq_f32(v,v));}
  float sum=vaddvq_f32(acc);for(;i<dim;i++){float f=bf(in[i]);sum+=f*f;}
  float scale=1.0f/sqrtf(sum/dim+eps);i=0;
  for(;i+4<=dim;i+=4){float32x4_t v=vmulq_n_f32(bf4(in+i),scale);
   if(!kind)v=vmulq_f32(v,vaddq_f32(vdupq_n_f32(1),bf4(w+i)));
   else{v=vreinterpretq_f32_u32(vshll_n_u16(rb4(v),16));v=vmulq_f32(v,bf4(w+i));v=vreinterpretq_f32_u32(vshll_n_u16(rb4(v),16));float a[4]={table[g[i]],table[g[i+1]],table[g[i+2]],table[g[i+3]]};v=vmulq_f32(v,vld1q_f32(a));}
   vst1_u16(dst+i,rb4(v));
  }
  for(;i<dim;i++){float v=bf(in[i])*scale;if(!kind)v*=1.0f+bf(w[i]);else{v=bf(rb(v));v=bf(rb(v*bf(w[i])));v*=table[g[i]];}dst[i]=rb(v);}
 }
}
