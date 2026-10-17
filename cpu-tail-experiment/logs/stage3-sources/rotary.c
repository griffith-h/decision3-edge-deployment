/* BF16 RoPE with both original intermediate product roundings preserved. */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static inline float32x4_t bf4(const uint16_t*x){return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(x),16));}
static inline uint16x4_t rb4(float32x4_t v){uint32x4_t u=vreinterpretq_u32_f32(v);uint32x4_t r=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16);r=vbslq_u32(vcgtq_u32(vandq_u32(u,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000)),vorrq_u32(vshrq_n_u32(u,16),vdupq_n_u32(0x40)),r);return vmovn_u32(r);}
static inline float32x4_t rounded(float32x4_t x){return vreinterpretq_f32_u32(vshll_n_u16(rb4(x),16));}
int d3_rope(const uint16_t*q,const uint16_t*k,const uint16_t*c,const uint16_t*s,uint16_t*qo,uint16_t*ko,const int64_t*strides,int batch,int qh,int kh,int len,int dim,int rot){
 if(batch<1||qh<1||kh<1||len<1||dim<rot||rot<8||rot%8)return -1;
 #pragma omp parallel for num_threads(3) schedule(static)
 for(int job=0;job<batch*(qh+kh)*len;job++){
  int t=job%len,hb=job/len,h=hb%(qh+kh),b=hb/(qh+kh),key=h>=qh;int heads=key?kh:qh;if(key)h-=qh;
  const int64_t*st=strides+(key?3:0);const uint16_t*in=(key?k:q)+b*st[0]+h*st[1]+t*st[2];uint16_t*out=(key?ko:qo)+((size_t)b*heads*len+h*len+t)*dim;
  const uint16_t*cc=c+((size_t)b*len+t)*rot;const uint16_t*ss=s+((size_t)b*len+t)*rot;
  for(int d=0;d<rot;d+=4){int other=d<rot/2?d+rot/2:d-rot/2;float32x4_t rv=bf4(in+other);if(d<rot/2)rv=vnegq_f32(rv);
   float32x4_t left=rounded(vmulq_f32(bf4(in+d),bf4(cc+d)));float32x4_t right=rounded(vmulq_f32(rv,bf4(ss+d)));vst1_u16(out+d,rb4(vaddq_f32(left,right)));
  }
  memcpy(out+rot,in+rot,(dim-rot)*2);
 }
 return 0;
}
