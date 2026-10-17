/* Same 4-tap causal convolution and BF16 rounding; consume token-major input. */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static inline float32x4_t bf4(const uint16_t*x){return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(x),16));}
static inline uint16x4_t rb(float32x4_t v){uint32x4_t u=vreinterpretq_u32_f32(v);uint32x4_t r=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16);r=vbslq_u32(vcgtq_u32(vandq_u32(u,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000)),vorrq_u32(vshrq_n_u32(u,16),vdupq_n_u32(0x40)),r);return vmovn_u32(r);}
int d3_conv_btc(const uint16_t*x,const uint16_t*w,const uint16_t*bias,uint16_t*out,int batch,int length,int channels){
 if(batch<=0||length<=0||channels<=0||channels%4)return -1;
 #pragma omp parallel for num_threads(3) schedule(static)
 for(int bt=0;bt<batch*length;++bt){
  int t=bt%length;const uint16_t*in=x+(size_t)bt*channels;uint16_t*dst=out+(size_t)bt*channels;
  for(int c=0;c<channels;c+=4){
   uint16x4x4_t ww=vld4_u16(w+c*4);float32x4_t weights[4];for(int j=0;j<4;++j)weights[j]=vreinterpretq_f32_u32(vshll_n_u16(ww.val[j],16));
   float32x4_t v=vmulq_f32(bf4(in+c),weights[3]);
   for(int o=1;o<4&&o<=t;++o)v=vaddq_f32(v,vmulq_f32(bf4(in+c-(size_t)o*channels),weights[3-o]));
   if(bias) v=vaddq_f32(v,bf4(bias+c));
   vst1_u16(dst+c,rb(v));
  }
 }
 return 0;
}
