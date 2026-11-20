/* Same per-output sequential K FMA and final BF16 boundary; full16/8 then original4 tails. */
#include <arm_neon.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
static uint16_t bf16(float v){union{float f;uint32_t u;}b={.f=v};b.u+=0x7fff+((b.u>>16)&1);return (uint16_t)(b.u>>16);}
static void transpose(float32x4_t a,float32x4_t b,float32x4_t c,float32x4_t d,float32x4_t*out){
 float32x4x2_t ab=vtrnq_f32(a,b),cd=vtrnq_f32(c,d);
 out[0]=vcombine_f32(vget_low_f32(ab.val[0]),vget_low_f32(cd.val[0]));
 out[1]=vcombine_f32(vget_low_f32(ab.val[1]),vget_low_f32(cd.val[1]));
 out[2]=vcombine_f32(vget_high_f32(ab.val[0]),vget_high_f32(cd.val[0]));
 out[3]=vcombine_f32(vget_high_f32(ab.val[1]),vget_high_f32(cd.val[1]));
}

#define DEFINE_COMPUTE(NAME,GROUPS) \
static __attribute__((noinline)) void NAME(const uint16_t*xx,const uint16_t*const*ww,int k,float*values){ \
 float32x4_t acc[GROUPS];for(int g=0;g<(GROUPS);g++)acc[g]=vdupq_n_f32(0); \
 for(int at=0;at<k;at+=4){ \
  float32x4_t xv=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(xx+at),16)),t[GROUPS][4]; \
  for(int g=0;g<(GROUPS);g++){ \
   float32x4_t a=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ww[4*g]+at),16)); \
   float32x4_t b=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ww[4*g+1]+at),16)); \
   float32x4_t c=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ww[4*g+2]+at),16)); \
   float32x4_t d=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ww[4*g+3]+at),16)); \
   transpose(a,b,c,d,t[g]); \
  } \
  for(int g=0;g<(GROUPS);g++)acc[g]=vfmaq_laneq_f32(acc[g],t[g][0],xv,0); \
  for(int g=0;g<(GROUPS);g++)acc[g]=vfmaq_laneq_f32(acc[g],t[g][1],xv,1); \
  for(int g=0;g<(GROUPS);g++)acc[g]=vfmaq_laneq_f32(acc[g],t[g][2],xv,2); \
  for(int g=0;g<(GROUPS);g++)acc[g]=vfmaq_laneq_f32(acc[g],t[g][3],xv,3); \
 } \
 for(int g=0;g<(GROUPS);g++)vst1q_f32(values+4*g,vaddq_f32(vdupq_n_f32(0),acc[g])); \
}
DEFINE_COMPUTE(compute4,1)
DEFINE_COMPUTE(compute8,2)
DEFINE_COMPUTE(compute16,4)
int d3_layout_tail(const uint16_t*x,int64_t sx,const uint16_t*w,int64_t sw,
 int rows,int k,int columns,const int64_t*pairs,int count,uint16_t*out,int mode,float*debug){
 if(!x||!w||!out||rows<=0||rows>615||k<=0||k>10240||k%4||columns<=0||columns>16384||sx<k||sw<k||count<0||count>65536||(!pairs&&count)||(mode!=4&&mode!=16))return -1;
 int offsets[616];for(int r=0;r<=rows;r++)offsets[r]=0;
 for(int i=0;i<count;i++){
  if(pairs[2*i]<0||pairs[2*i]>=rows||pairs[2*i+1]<0||pairs[2*i+1]>=columns||(i&&pairs[2*i]<pairs[2*i-2]))return -2;
  offsets[pairs[2*i]+1]++;
 }
 for(int r=0;r<rows;r++)offsets[r+1]+=offsets[r];
 int bad=0;
 #pragma omp parallel for num_threads(3) schedule(static) reduction(|:bad)
 for(int row=0;row<rows;row++){
  const uint16_t*xx=x+(size_t)row*sx;
  for(int first=offsets[row];first<offsets[row+1];){
   int remaining=offsets[row+1]-first;
   int width=mode==4?4:(remaining>=16?16:(remaining>=8?8:4));
   int used=remaining<width?remaining:width;const uint16_t*ww[16];float values[16];
   for(int j=0;j<width;j++)ww[j]=w+pairs[2*(first+(j<used?j:used-1))+1]*sw;
   if(width==16)compute16(xx,ww,k,values);else if(width==8)compute8(xx,ww,k,values);else compute4(xx,ww,k,values);
   for(int j=0;j<used;j++){
    if(debug)debug[first+j]=values[j];
    if(!isfinite(values[j])){bad=1;continue;}
    out[(size_t)row*columns+pairs[2*(first+j)+1]]=bf16(values[j]);
   }
   first+=used;
  }
 }
 return bad?-3:0;
}
