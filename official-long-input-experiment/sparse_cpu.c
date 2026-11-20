/* Own bounded BF16-to-FP32 row gather; arithmetic stays in original PyTorch. */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static void convert(const uint16_t*src,float*dst,int k){
 int j=0;for(;j+4<=k;j+=4)vst1q_f32(dst+j,vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(src+j),16)));
 for(;j<k;j++){uint32_t bits=(uint32_t)src[j]<<16;memcpy(dst+j,&bits,4);}
}
int d3_sparse_gather(const uint16_t*x,int64_t sx,const uint16_t*w,int64_t sw,
 int rows,int k,int columns,const int64_t*pairs,int count,float*a,float*b){
 if(!x||!w||rows<=0||rows>615||k<=0||k>10240||columns<=0||columns>16384||sx<k||sw<k||count<0||count>4096)return -1;
 if(!count)return 0;
 if(!pairs||!a||!b)return -1;
 for(int i=0;i<count;i++)if(pairs[2*i]<0||pairs[2*i]>=rows||pairs[2*i+1]<0||pairs[2*i+1]>=columns)return -2;
 int padded=(count+3)/4*4;
 #pragma omp parallel for num_threads(3) schedule(static)
 for(int row=0;row<padded;row++){
  int i=row<count?row:count-1;
  convert(x+pairs[2*i]*sx,a+(size_t)row*k,k);
  convert(w+pairs[2*i+1]*sw,b+(size_t)row*k,k);
 }
 return 0;
}

/* Optional approximate CPU repair experiment; full-model gates required. */
#include <math.h>
typedef void (*sgemm_fn)(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
static sgemm_fn gemm=0;
int d3_sparse_set_sgemm(void*fn){if(!fn)return -1;gemm=(sgemm_fn)fn;return 0;}
static uint16_t rb(float f){uint32_t u;memcpy(&u,&f,4);return (u+0x7fff+((u>>16)&1))>>16;}
int d3_sparse_repair(const uint16_t*x,int64_t sx,const uint16_t*w,int64_t sw,
 int rows,int k,int columns,const int64_t*pairs,int count,uint16_t*out){
 if(!gemm||!x||!w||!out||rows<=0||rows>615||k<=0||k>10240||columns<=0||columns>16384||sx<k||sw<k||count<0||count>65536||(!pairs&&count))return -1;
 for(int i=0;i<count;i++)if(pairs[2*i]<0||pairs[2*i]>=rows||pairs[2*i+1]<0||pairs[2*i+1]>=columns)return -2;
 int bad=0;
 #pragma omp parallel for num_threads(3) schedule(static) reduction(|:bad)
 for(int first=0;first<count;first+=4){
  float a[4*10240],b[4*10240],c[16];int used=count-first;if(used>4)used=4;
  for(int r=0;r<4;r++){
   int i=first+(r<used?r:used-1);
   convert(x+pairs[2*i]*sx,a+r*k,k);convert(w+pairs[2*i+1]*sw,b+r*k,k);
  }
  gemm(101,111,112,4,4,k,1,a,k,b,k,0,c,4);
  for(int r=0;r<used;r++){
   float value=c[r*4+r];if(!isfinite(value)){bad=1;continue;}
   int i=first+r;out[pairs[2*i]*columns+pairs[2*i+1]]=rb(value);
  }
 }
 return bad?-3:0;
}

/* Fused uncertainty-mask preparation; arithmetic ordering matches Torch FP32. */
static int selected(float raw,float a,float b,float c,float wn,float wl){
 uint32_t u;memcpy(&u,&raw,4);u&=0x7fffffffU;
 uint32_t low=u&0xffff0000U,high=low+65536U;float lo,hi,mag;
 memcpy(&lo,&low,4);memcpy(&hi,&high,4);memcpy(&mag,&u,4);
 float midpoint=(lo+hi)*.5f;
 float band=(a*wn+b*wn)+c*wl;
 return fabsf(mag-midpoint)<=band;
}
int d3_sparse_mask(const float*raw,int rows,int columns,const float*xn,const float*xl,const float*xhn,
 const float*wn,const float*wl,float noise,double max_fraction,int capacity,int64_t*pairs,int*count_out){
 if(!raw||!xn||!xl||!xhn||!wn||!wl||!pairs||!count_out||rows<=0||rows>615||columns<=0||columns>16384||capacity<0||capacity>65536||!(max_fraction>0&&max_fraction<=1)||!(noise>0))return -1;
 int counts[615],offsets[616];int bad=0;
 #pragma omp parallel for num_threads(3) schedule(static) reduction(|:bad)
 for(int r=0;r<rows;r++){
  int n=0;float a=noise*xn[r];
  for(int col=0;col<columns;col++){
   float value=raw[(size_t)r*columns+col];bad|=!isfinite(value);
   n+=selected(value,a,xl[r],xhn[r],wn[col],wl[col]);
  }
  counts[r]=n;
 }
 if(bad)return -3;
 offsets[0]=0;for(int r=0;r<rows;r++)offsets[r+1]=offsets[r]+counts[r];
 int total=offsets[rows];*count_out=total;
 if(total>capacity||(double)total/((double)rows*columns)>max_fraction)return -4;
 #pragma omp parallel for num_threads(3) schedule(static)
 for(int r=0;r<rows;r++){
  int next=offsets[r];float a=noise*xn[r];
  for(int col=0;col<columns;col++)if(selected(raw[(size_t)r*columns+col],a,xl[r],xhn[r],wn[col],wl[col])){
   pairs[2*next]=r;pairs[2*next+1]=col;next++;
  }
 }
 return 0;
}
static float bf_scalar(uint16_t bits){uint32_t u=(uint32_t)bits<<16;float f;memcpy(&f,&u,4);return f;}
int d3_sparse_norms(const uint16_t*original,int64_t stride,const uint16_t*half,int rows,int width,
 float*norm,float*loss,float*half_norm){
 if(!original||!half||!norm||!loss||rows<=0||rows>16384||width<=0||width>10240||stride<width)return -1;
 int bad=0;
 #pragma omp parallel for num_threads(3) schedule(static) reduction(|:bad)
 for(int row=0;row<rows;row++){
  double sum=0,lsum=0,hsum=0;
  for(int col=0;col<width;col++){
   uint16_t hbits=half[(size_t)row*width+col];bad|=(hbits&0x7c00)==0x7c00;
   __fp16 hh;memcpy(&hh,&hbits,2);float h=hh;float value=bf_scalar(original[(size_t)row*stride+col]);float delta=value-h;
   sum+=(double)value*value;lsum+=(double)delta*delta;hsum+=(double)h*h;
  }
  norm[row]=(float)sqrt(sum);loss[row]=(float)sqrt(lsum);if(half_norm)half_norm[row]=(float)sqrt(hsum);
 }
 return bad?-3:0;
}

/* Group the same selected positions by input row. Public FP32 BLAS only.
 * mode1:1x4; mode4:4x16 with duplicate input rows, no GEMV shortcut.
 * Exactness against the previous repair is a preflight gate, not assumed.
 */
int d3_sparse_repair_rows(const uint16_t*x,int64_t sx,const uint16_t*w,int64_t sw,
 int rows,int k,int columns,const int64_t*pairs,int count,uint16_t*out,int mode){
 if(!gemm||!x||!w||!out||rows<=0||rows>615||k<=0||k>10240||columns<=0||columns>16384||sx<k||sw<k||count<0||count>65536||(!pairs&&count)||(mode!=1&&mode!=2&&mode!=4))return -1;
 int offsets[616];for(int r=0;r<=rows;r++)offsets[r]=0;
 for(int i=0;i<count;i++){
  if(pairs[2*i]<0||pairs[2*i]>=rows||pairs[2*i+1]<0||pairs[2*i+1]>=columns||(i&&pairs[2*i]<pairs[2*i-2]))return -2;
  offsets[pairs[2*i]+1]++;
 }
 for(int r=0;r<rows;r++)offsets[r+1]+=offsets[r];
 int bad=0,width=mode==4?16:4,matrix_rows=mode==1?1:4;
 #pragma omp parallel for num_threads(3) schedule(static) reduction(|:bad)
 for(int row=0;row<rows;row++){
  if(offsets[row]==offsets[row+1])continue;
  float a[4*10240],b[16*10240],c[64];convert(x+(size_t)row*sx,a,k);
  for(int r=1;r<matrix_rows;r++)memcpy(a+r*k,a,(size_t)k*4);
  for(int first=offsets[row];first<offsets[row+1];first+=width){
   int used=offsets[row+1]-first;if(used>width)used=width;
   for(int r=0;r<width;r++)convert(w+pairs[2*(first+(r<used?r:used-1))+1]*sw,b+r*k,k);
   gemm(101,111,112,matrix_rows,width,k,1,a,k,b,k,0,c,width);
   for(int r=0;r<used;r++){
    float value=c[r];if(!isfinite(value)){bad=1;continue;}
    out[(size_t)row*columns+pairs[2*(first+r)+1]]=rb(value);
   }
  }
 }
 return bad?-3:0;
}

/* Exact four-lane form of selected(): same FP32 operations, no contraction. */
static uint32x4_t selected4(float32x4_t raw,float a,float b,float c,float32x4_t wn,float32x4_t wl){
 uint32x4_t bits=vandq_u32(vreinterpretq_u32_f32(raw),vdupq_n_u32(0x7fffffff));
 uint32x4_t low=vandq_u32(bits,vdupq_n_u32(0xffff0000));
 float32x4_t midpoint=vmulq_n_f32(vaddq_f32(vreinterpretq_f32_u32(low),vreinterpretq_f32_u32(vaddq_u32(low,vdupq_n_u32(65536)))),.5f);
 float32x4_t band=vaddq_f32(vaddq_f32(vmulq_n_f32(wn,a),vmulq_n_f32(wn,b)),vmulq_n_f32(wl,c));
 return vcleq_f32(vabsq_f32(vsubq_f32(vreinterpretq_f32_u32(bits),midpoint)),band);
}
int d3_sparse_mask_neon(const float*raw,int rows,int columns,const float*xn,const float*xl,const float*xhn,
 const float*wn,const float*wl,float noise,double max_fraction,int capacity,int64_t*pairs,int*count_out){
 if(!raw||!xn||!xl||!xhn||!wn||!wl||!pairs||!count_out||rows<=0||rows>615||columns<=0||columns>16384||capacity<0||capacity>65536||!(max_fraction>0&&max_fraction<=1)||!(noise>0))return -1;
 int counts[615],offsets[616],bad=0;
 #pragma omp parallel for num_threads(3) schedule(static) reduction(|:bad)
 for(int r=0;r<rows;r++){
  int n=0,j=0;float a=noise*xn[r];const float*values=raw+(size_t)r*columns;
  for(;j+4<=columns;j+=4){
   float32x4_t v=vld1q_f32(values+j);
   uint32x4_t finite=vcltq_u32(vandq_u32(vreinterpretq_u32_f32(v),vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000));
   bad|=vminvq_u32(finite)==0;
   n+=vaddvq_u32(vshrq_n_u32(selected4(v,a,xl[r],xhn[r],vld1q_f32(wn+j),vld1q_f32(wl+j)),31));
  }
  for(;j<columns;j++){bad|=!isfinite(values[j]);n+=selected(values[j],a,xl[r],xhn[r],wn[j],wl[j]);}
  counts[r]=n;
 }
 if(bad)return -3;
 offsets[0]=0;for(int r=0;r<rows;r++)offsets[r+1]=offsets[r]+counts[r];
 int total=offsets[rows];*count_out=total;
 if(total>capacity||(double)total/((double)rows*columns)>max_fraction)return -4;
 #pragma omp parallel for num_threads(3) schedule(static)
 for(int r=0;r<rows;r++){
  int next=offsets[r],j=0;float a=noise*xn[r];const float*values=raw+(size_t)r*columns;
  for(;j+4<=columns;j+=4){
   uint32x4_t hit=selected4(vld1q_f32(values+j),a,xl[r],xhn[r],vld1q_f32(wn+j),vld1q_f32(wl+j));
   if(vmaxvq_u32(hit)){
    uint32_t lanes[4];vst1q_u32(lanes,hit);
    for(int lane=0;lane<4;lane++)if(lanes[lane]){pairs[2*next]=r;pairs[2*next+1]=j+lane;next++;}
   }
  }
  for(;j<columns;j++)if(selected(values[j],a,xl[r],xhn[r],wn[j],wl[j])){pairs[2*next]=r;pairs[2*next+1]=j;next++;}
 }
 return 0;
}

/* Sequential-block diagonal experiment. Each lane is an independent selected
 * dot, not a reduction over K. Q must be qualified against pinned public BLAS.
 * No private BLAS/SDK entrypoints. Reference algorithm: OpenBLAS v0.3.30
 * public SGEMM driver and A57 kernel; mathematical order only, own NEON code.
 */
static void transpose4(float32x4_t a,float32x4_t b,float32x4_t c,float32x4_t d,float32x4_t*out){
 float32x4x2_t ab=vtrnq_f32(a,b),cd=vtrnq_f32(c,d);
 out[0]=vcombine_f32(vget_low_f32(ab.val[0]),vget_low_f32(cd.val[0]));
 out[1]=vcombine_f32(vget_low_f32(ab.val[1]),vget_low_f32(cd.val[1]));
 out[2]=vcombine_f32(vget_high_f32(ab.val[0]),vget_high_f32(cd.val[0]));
 out[3]=vcombine_f32(vget_high_f32(ab.val[1]),vget_high_f32(cd.val[1]));
}
int d3_sparse_repair_diagonal(const uint16_t*x,int64_t sx,const uint16_t*w,int64_t sw,
 int rows,int k,int columns,const int64_t*pairs,int count,uint16_t*out,int qblock){
 if(!x||!w||!out||rows<=0||rows>615||k<=0||k>10240||k%4||columns<=0||columns>16384||sx<k||sw<k||count<0||count>65536||(!pairs&&count)||(qblock!=352&&qblock!=1024&&qblock!=16384))return -1;
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
  for(int first=offsets[row];first<offsets[row+1];first+=4){
   int used=offsets[row+1]-first;if(used>4)used=4;const uint16_t*ww[4];
   for(int j=0;j<4;j++)ww[j]=w+pairs[2*(first+(j<used?j:used-1))+1]*sw;
   float32x4_t total=vdupq_n_f32(0);
   for(int start=0;start<k;){
    int remaining=k-start,n=remaining;
    if(remaining>=2*qblock)n=qblock;else if(remaining>qblock)n=((remaining/2+3)/4)*4;
    float32x4_t acc=vdupq_n_f32(0);
    for(int at=start;at<start+n;at+=4){
     float32x4_t xv=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(xx+at),16)),wv[4],t[4];
     for(int j=0;j<4;j++)wv[j]=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ww[j]+at),16));
     transpose4(wv[0],wv[1],wv[2],wv[3],t);
     acc=vfmaq_laneq_f32(acc,t[0],xv,0);acc=vfmaq_laneq_f32(acc,t[1],xv,1);
     acc=vfmaq_laneq_f32(acc,t[2],xv,2);acc=vfmaq_laneq_f32(acc,t[3],xv,3);
    }
    total=vaddq_f32(total,acc);start+=n;
   }
   float values[4];vst1q_f32(values,total);
   for(int j=0;j<used;j++){
    if(!isfinite(values[j])){bad=1;continue;}
    out[(size_t)row*columns+pairs[2*(first+j)+1]]=rb(values[j]);
   }
  }
 }
 return bad?-3:0;
}
