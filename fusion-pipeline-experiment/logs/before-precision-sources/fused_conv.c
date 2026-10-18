/* Own exact 4-tap BTC convolution: reuse weights across four time rows and
 * apply the existing BF16 SiLU table after the original convolution rounding. */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
static inline float32x4_t bf4(const uint16_t*x){return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(x),16));}
static inline uint16x4_t rb(float32x4_t v){uint32x4_t u=vreinterpretq_u32_f32(v);uint32x4_t r=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16);r=vbslq_u32(vcgtq_u32(vandq_u32(u,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000)),vorrq_u32(vshrq_n_u32(u,16),vdupq_n_u32(0x40)),r);return vmovn_u32(r);}
int d3_conv_fused(const uint16_t*x,const uint16_t*w,const uint16_t*bias,uint16_t*out,const uint16_t*table,int batch,int length,int channels){
 if(!x||!w||!out||batch<=0||batch>8||length<=0||length>2048||channels<=0||channels>16384||channels%4)return -1;
 float*weights=malloc((size_t)channels*4*4);if(!weights)return -2;
 for(int c=0;c<channels;c+=4){uint16x4x4_t raw=vld4_u16(w+c*4);for(int tap=0;tap<4;tap++)vst1q_f32(weights+tap*channels+c,vreinterpretq_f32_u32(vshll_n_u16(raw.val[tap],16)));}
 int times=(length+3)/4,tiles=(channels+127)/128;
 #pragma omp parallel for num_threads(3) schedule(static)
 for(int job=0;job<batch*times*tiles;job++){
  int tile=job%tiles,bt=job/tiles,b=bt/times,first=(bt%times)*4,lo=tile*128,hi=lo+128;if(hi>channels)hi=channels;
  for(int c=lo;c<hi;c+=4){float32x4_t ww[4];for(int tap=0;tap<4;tap++)ww[tap]=vld1q_f32(weights+tap*channels+c);float32x4_t bb=bias?bf4(bias+c):vdupq_n_f32(0);
   for(int t=first;t<length&&t<first+4;t++){
    const uint16_t*in=x+((size_t)b*length+t)*channels+c;uint16_t*dst=out+((size_t)b*length+t)*channels+c;
    float32x4_t value=vmulq_f32(bf4(in),ww[3]);
    for(int tap=1;tap<4&&tap<=t;tap++)value=vaddq_f32(value,vmulq_f32(bf4(in-(size_t)tap*channels),ww[3-tap]));
    if(bias)value=vaddq_f32(value,bb);
    uint16x4_t rounded=rb(value);
    if(table){uint16_t a[4];vst1_u16(a,rounded);for(int j=0;j<4;j++)dst[j]=table[a[j]];}
    else vst1_u16(dst,rounded);
   }
  }
 }
 free(weights);return 0;
}
