/* Own block form of the public zero-state causal gated-delta equations. */
#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
static inline uint16x4_t rb(float32x4_t v){uint32x4_t u=vreinterpretq_u32_f32(v),r=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16);r=vbslq_u32(vcgtq_u32(vandq_u32(u,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000)),vdupq_n_u32(0x7fc0),r);return vmovn_u32(r);}
/* Four query rows, eight value columns: independent accumulators fit NEON registers. */
static void product(const float*a,const float*s,float*out,int n,int kd,int width){
 float zeros[128]={0};
 for(int row=0;row<n;row+=4){
  const float *a0=a+row*kd,*a1=row+1<n?a+(row+1)*kd:zeros,*a2=row+2<n?a+(row+2)*kd:zeros,*a3=row+3<n?a+(row+3)*kd:zeros;
  for(int j=0;j<width;j+=8){
   float32x4_t p0=vdupq_n_f32(0),p1=p0,p2=p0,p3=p0,r0=p0,r1=p0,r2=p0,r3=p0;
   for(int k=0;k<kd;++k){float32x4_t v0=vld1q_f32(s+k*width+j);p0=vfmaq_n_f32(p0,v0,a0[k]);p1=vfmaq_n_f32(p1,v0,a1[k]);p2=vfmaq_n_f32(p2,v0,a2[k]);p3=vfmaq_n_f32(p3,v0,a3[k]);if(j+4<width){float32x4_t v1=vld1q_f32(s+k*width+j+4);r0=vfmaq_n_f32(r0,v1,a0[k]);r1=vfmaq_n_f32(r1,v1,a1[k]);r2=vfmaq_n_f32(r2,v1,a2[k]);r3=vfmaq_n_f32(r3,v1,a3[k]);}}
   vst1q_f32(out+row*width+j,p0);if(row+1<n)vst1q_f32(out+(row+1)*width+j,p1);if(row+2<n)vst1q_f32(out+(row+2)*width+j,p2);if(row+3<n)vst1q_f32(out+(row+3)*width+j,p3);
   if(j+4<width){vst1q_f32(out+row*width+j+4,r0);if(row+1<n)vst1q_f32(out+(row+1)*width+j+4,r1);if(row+2<n)vst1q_f32(out+(row+2)*width+j+4,r2);if(row+3<n)vst1q_f32(out+(row+3)*width+j+4,r3);}
  }
 }
}
int d3_chunk_scan(const float*q,const float*k,const float*v,const float*decay,const float*beta,uint16_t*out,int heads,int length,int kd,int vd,int block,int tile){
 if(heads<=0||length<=0||length>512||kd<=0||kd>128||kd%4||vd<=0||vd>128||vd%4||block<2||block>32||(tile!=8&&tile!=16&&tile!=32))return -1;
 int chunks=(length+block-1)/block,coeffsize=2*block*block+2*block;
 float*coeff=calloc((size_t)heads*chunks*coeffsize,sizeof(float));if(!coeff)return -2;
 #pragma omp parallel for num_threads(3) schedule(static)
 for(int h=0;h<heads;++h)for(int cb=0;cb<chunks;++cb){
  int start=cb*block,n=length-start;if(n>block)n=block;
  float *c=coeff+((size_t)h*chunks+cb)*coeffsize,*u=c,*a=c+block*block,*d=c+2*block*block,*tail=d+block;
  int padded=(n+7)/8*8;float kt[128*32],qq[32*32],kk[32*32];
  for(int ki=0;ki<kd;++ki)for(int j=0;j<padded;++j)kt[ki*padded+j]=j<n?k[((size_t)h*length+start+j)*kd+ki]:0;
  product(q+((size_t)h*length+start)*kd,kt,qq,n,kd,padded);
  product(k+((size_t)h*length+start)*kd,kt,kk,n,kd,padded);
  float accum=1;
  for(int i=0;i<n;++i){size_t row=(size_t)h*length+start+i;accum*=decay[row];d[i]=accum;float pair=1;
   for(int j=i;j>=0;--j){size_t prev=(size_t)h*length+start+j;a[i*block+j]=pair*qq[i*padded+j];if(j<i)u[i*block+j]=(beta[row]*pair)*kk[i*padded+j];pair*=decay[prev];}
  }
  accum=1;for(int j=n-1;j>=0;--j){tail[j]=accum;accum*=decay[(size_t)h*length+start+j];}
 }
 int tiles=(vd+tile-1)/tile;
 #pragma omp parallel for num_threads(3) schedule(static)
 for(int job=0;job<heads*tiles;++job){
  int h=job/tiles,first=job%tiles*tile,width=vd-first;if(width>tile)width=tile;
  float state[128*32] __attribute__((aligned(64)));memset(state,0,(size_t)kd*width*4);
  float pred[32*32] __attribute__((aligned(64))),readout[32*32] __attribute__((aligned(64))),delta[32*32] __attribute__((aligned(64)));
  for(int cb=0;cb<chunks;++cb){
   int start=cb*block,n=length-start;if(n>block)n=block;
   const float*c=coeff+((size_t)h*chunks+cb)*coeffsize,*u=c,*a=c+block*block,*d=c+2*block*block,*tail=d+block;
   const float *ks=k+((size_t)h*length+start)*kd,*qs=q+((size_t)h*length+start)*kd;
   if(cb==0){memset(pred,0,n*width*4);memset(readout,0,n*width*4);}
   else {product(ks,state,pred,n,kd,width);product(qs,state,readout,n,kd,width);}
   for(int i=0;i<n;++i){size_t row=(size_t)h*length+start+i;
    for(int j=0;j<width;j+=4){float32x4_t dd=vmulq_n_f32(vsubq_f32(vld1q_f32(v+row*vd+first+j),vmulq_n_f32(vld1q_f32(pred+i*width+j),d[i])),beta[row]);
     for(int z=0;z<i;++z)dd=vfmaq_n_f32(dd,vld1q_f32(delta+z*width+j),-u[i*block+z]);
     vst1q_f32(delta+i*width+j,dd);
     float32x4_t oo=vmulq_n_f32(vld1q_f32(readout+i*width+j),d[i]);for(int z=0;z<=i;++z)oo=vfmaq_n_f32(oo,vld1q_f32(delta+z*width+j),a[i*block+z]);vst1_u16(out+row*vd+first+j,rb(oo));
    }
   }
   /* No state is needed after the final block. */
   if(cb+1<chunks)for(int ki=0;ki<kd;++ki)for(int j=0;j<width;j+=4){float32x4_t ss=vmulq_n_f32(vld1q_f32(state+ki*width+j),d[n-1]);for(int z=0;z<n;++z)ss=vfmaq_n_f32(ss,vld1q_f32(delta+z*width+j),tail[z]*ks[z*kd+ki]);vst1q_f32(state+ki*width+j,ss);}
  }
 }
 free(coeff);return 0;
}
