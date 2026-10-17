/* Fuse BF16 layout conversion and FP32 q/k normalization, with original boundaries. */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
static inline float32x4_t bf4(const uint16_t*x){return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(x),16));}
int d3_prepare_scan(const uint16_t*q,const uint16_t*k,const uint16_t*v,float*qo,float*ko,float*vo,const int64_t*strides,int batch,int length,int heads,int kd,int vd,int normalize){
 if(batch<=0||length<=0||heads<=0||kd<=0||kd>128||kd%4||vd<=0||vd>128||vd%4)return -1;
 float scaling=powf((float)kd,-0.5f);
 #pragma omp parallel for num_threads(3) schedule(static)
 for(int row=0;row<batch*heads*length;++row){
  int t=row%length,h=(row/length)%heads,b=row/(length*heads);
  const uint16_t*qs=q+b*strides[0]+t*strides[1]+h*strides[2];
  const uint16_t*ks=k+b*strides[3]+t*strides[4]+h*strides[5];
  const uint16_t*vs=v+b*strides[6]+t*strides[7]+h*strides[8];
  float *qd=qo+(size_t)row*kd,*kdout=ko+(size_t)row*kd,*vdout=vo+(size_t)row*vd;
  float qscale=1,kmscale=1;
  if(normalize){float32x4_t qsum=vdupq_n_f32(0),ksum=qsum;for(int i=0;i<kd;i+=4){float32x4_t qq=bf4(qs+i),kk=bf4(ks+i);qsum=vaddq_f32(qsum,vmulq_f32(qq,qq));ksum=vaddq_f32(ksum,vmulq_f32(kk,kk));}qscale=1.0f/sqrtf(vaddvq_f32(qsum)+1e-6f);kmscale=1.0f/sqrtf(vaddvq_f32(ksum)+1e-6f);}
  for(int i=0;i<kd;i+=4){vst1q_f32(qd+i,vmulq_n_f32(vmulq_n_f32(bf4(qs+i),qscale),scaling));vst1q_f32(kdout+i,vmulq_n_f32(bf4(ks+i),kmscale));}
  for(int i=0;i<vd;i+=4)vst1q_f32(vdout+i,bf4(vs+i));
 }
 return 0;
}
