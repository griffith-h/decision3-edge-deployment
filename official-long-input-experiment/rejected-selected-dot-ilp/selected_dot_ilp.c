/* Independent-output instruction parallelism. Each output retains the original
 * continuous K FMA order and final +0 / BF16 rounding. No partial K sums.
 * Optional FP32 debug writes are outside the K loop and disabled in inference.
 */
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
#define DEFINE_DOTS(NAME,GROUPS) \
static int NAME(const uint16_t*x,int64_t sx,const uint16_t*w,int64_t sw,int rows,int k,int columns,const int64_t*pairs,const int*offsets,uint16_t*out,float*debug){ \
 int bad=0; \
 _Pragma("omp parallel for num_threads(3) schedule(static) reduction(|:bad)") \
 for(int row=0;row<rows;row++){ \
  const uint16_t*xx=x+(size_t)row*sx; \
  for(int first=offsets[row];first<offsets[row+1];first+=4*(GROUPS)){ \
   int used=offsets[row+1]-first;if(used>4*(GROUPS))used=4*(GROUPS);const uint16_t*ww[4*(GROUPS)]; \
   for(int j=0;j<4*(GROUPS);j++)ww[j]=w+pairs[2*(first+(j<used?j:used-1))+1]*sw; \
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
   float values[4*(GROUPS)];for(int g=0;g<(GROUPS);g++)vst1q_f32(values+4*g,vaddq_f32(vdupq_n_f32(0),acc[g])); \
   for(int j=0;j<used;j++){ \
    if(debug)debug[first+j]=values[j]; \
    if(!isfinite(values[j])){bad=1;continue;} \
    out[(size_t)row*columns+pairs[2*(first+j)+1]]=bf16(values[j]); \
   } \
  } \
 } \
 return bad?-3:0; \
}
DEFINE_DOTS(dots4,1)
DEFINE_DOTS(dots8,2)
DEFINE_DOTS(dots16,4)
int d3_selected_dot_ilp(const uint16_t*x,int64_t sx,const uint16_t*w,int64_t sw,
 int rows,int k,int columns,const int64_t*pairs,int count,uint16_t*out,int width,float*debug){
 if(!x||!w||!out||rows<=0||rows>615||k<=0||k>10240||k%4||columns<=0||columns>16384||sx<k||sw<k||count<0||count>65536||(!pairs&&count)||(width!=4&&width!=8&&width!=16))return -1;
 int offsets[616];for(int r=0;r<=rows;r++)offsets[r]=0;
 for(int i=0;i<count;i++){
  if(pairs[2*i]<0||pairs[2*i]>=rows||pairs[2*i+1]<0||pairs[2*i+1]>=columns||(i&&pairs[2*i]<pairs[2*i-2]))return -2;
  offsets[pairs[2*i]+1]++;
 }
 for(int r=0;r<rows;r++)offsets[r+1]+=offsets[r];
 if(width==4)return dots4(x,sx,w,sw,rows,k,columns,pairs,offsets,out,debug);
 if(width==8)return dots8(x,sx,w,sw,rows,k,columns,pairs,offsets,out,debug);
 return dots16(x,sx,w,sw,rows,k,columns,pairs,offsets,out,debug);
}
