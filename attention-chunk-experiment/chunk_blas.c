/* Own block form of the public zero-state causal gated-delta equations. */
#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
static inline uint16x4_t rb(float32x4_t v){uint32x4_t u=vreinterpretq_u32_f32(v),r=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16);r=vbslq_u32(vcgtq_u32(vandq_u32(u,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000)),vdupq_n_u32(0x7fc0),r);return vmovn_u32(r);}
/* NumPy's bundled OpenBLAS public ILP64 CBLAS API. One BLAS thread per OMP job. */
typedef void (*sgemm_fn)(int,int,int,int64_t,int64_t,int64_t,float,const float*,int64_t,const float*,int64_t,float,float*,int64_t);
static sgemm_fn gemm=0;
void d3_set_sgemm(void*fn){gemm=(sgemm_fn)fn;}
static void product(const float*a,const float*s,float*out,int n,int kd,int width){gemm(101,111,111,n,width,kd,1,a,kd,s,width,0,out,width);}
int d3_chunk_scan(const float*q,const float*k,const float*v,const float*decay,const float*beta,uint16_t*out,int heads,int length,int kd,int vd,int block,int tile){
 if(!gemm||heads<=0||length<=0||length>512||kd<=0||kd>128||kd%4||vd<=0||vd>128||vd%4||block<2||block>32||(tile!=8&&tile!=16&&tile!=32&&tile!=64&&tile!=128))return -1;
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
  float state[128*128] __attribute__((aligned(64)));memset(state,0,(size_t)kd*width*4);
  float pred[32*128] __attribute__((aligned(64))),readout[32*128] __attribute__((aligned(64))),delta[32*128] __attribute__((aligned(64)));
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
   if(cb+1<chunks){
    float ktail[128*32];for(int ki=0;ki<kd;++ki)for(int z=0;z<n;++z)ktail[ki*n+z]=tail[z]*ks[z*kd+ki];
    gemm(101,111,111,kd,width,n,1,ktail,n,delta,width,d[n-1],state,width);
   }

  }
 }
 free(coeff);return 0;
}
