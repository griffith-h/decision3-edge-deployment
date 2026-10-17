#include <arm_neon.h>
#include <cstdint>
#include <cstring>
#include <cmath>
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
static int mlp_convert_half(const uint16_t* src,uint16_t* dst,size_t count) {
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
static int mlp_convert_input(const uint16_t* src,uint16_t* dst,size_t count) {
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
static void mlp_convert_float(const float* src, uint16_t* dst, size_t count) {
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
