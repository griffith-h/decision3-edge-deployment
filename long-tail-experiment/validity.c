/* Validation only: preserve rejection checks without allocating Torch masks. */
#include <arm_neon.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
static float bf(uint16_t x){uint32_t u=(uint32_t)x<<16;float f;memcpy(&f,&u,4);return f;}
int d3_finite_bf16(const uint16_t*x,int64_t n){
 if(n<0)return -1;
 int64_t i=0;
 for(;i+8<=n;i+=8)if(vmaxvq_u16(vcgeq_u16(vandq_u16(vld1q_u16(x+i),vdupq_n_u16(0x7fff)),vdupq_n_u16(0x7f80))))return 1;
 for(;i<n;i++)if((x[i]&0x7fff)>=0x7f80)return 1;
 return 0;
}
int d3_finite_f32(const float*x,int64_t n){
 if(n<0)return -1;
 int64_t i=0;
 for(;i+4<=n;i+=4)if(vmaxvq_u32(vcgeq_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(x+i)),vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000))))return 1;
 for(;i<n;i++)if(!isfinite(x[i]))return 1;
 return 0;
}
int d3_validate_scan(const uint16_t*q,const uint16_t*k,const uint16_t*v,const void*g,const void*beta,const int64_t*s,int batch,int length,int heads,int kd,int vd,int gf,int beta_f32){
 if(batch<1||batch>8||length<1||length>2048||heads<1||batch*heads>128||kd<1||kd>128||vd<1||vd>128||(gf!=0&&gf!=1)||(beta_f32!=0&&beta_f32!=1))return -1;
 int bad=0;
 #pragma omp parallel for num_threads(3) schedule(static) reduction(|:bad)
 for(int row=0;row<batch*length*heads;row++){
  int h=row%heads,t=(row/heads)%length,b=row/(length*heads);
  const uint16_t*qs=q+b*s[0]+t*s[1]+h*s[2],*ks=k+b*s[3]+t*s[4]+h*s[5],*vs=v+b*s[6]+t*s[7]+h*s[8];
  if(d3_finite_bf16(qs,kd)||d3_finite_bf16(ks,kd)||d3_finite_bf16(vs,vd))bad=1;
  int64_t gi=b*s[9]+t*s[10]+h*s[11],bi=b*s[12]+t*s[13]+h*s[14];
  float gv=gf?((const float*)g)[gi]:bf(((const uint16_t*)g)[gi]),bv=beta_f32?((const float*)beta)[bi]:bf(((const uint16_t*)beta)[bi]);
  if(!isfinite(gv)||gv>0||!isfinite(bv)||bv<0||bv>1)bad=1;
 }
 return bad;
}
