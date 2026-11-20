// Include our immutable native-layout implementation, never vendor internals.
#include "native_split_reference.inc"
extern "C" int d3_probe_copy_fp32(void*handle,int rows,float*output){
 auto*p=static_cast<Matrix*>(handle);
 if(!p||!output||p->prepared!=rows||rows<=0||p->selected<0)return -1;
 int m=p->shapes[p->selected].M;
 if(rows>m)return -1;
 auto*source=static_cast<const float*>(p->c->virt_addr);
 // Keep both sides of the native-column4 -> row-major copy in a small
 // working set. Every FP32 value and the finite/range contract are unchanged;
 // no extra thread, arithmetic or persistent buffer is introduced.
 for(int row_begin=0;row_begin<rows;row_begin+=16)
  for(int column_begin=0;column_begin<p->n;column_begin+=64)
   for(int column=column_begin;column<std::min(column_begin+64,p->n);column+=4)
    for(int row=row_begin;row<std::min(row_begin+16,rows);row++){
     const float*src=source+(size_t(column/4)*m+row)*4;
     for(int j=0;j<4;j++){if(!std::isfinite(src[j]))return -3;output[size_t(row)*p->n+column+j]=src[j];}
    }
 return 0;
}

// Same owned native Matrix ABI as the immutable parent, with corrected BF16
// gate/up inputs. Preserve original SiLU/product rounding and native A layout.
extern "C" int d3_native_product_bf16(void*handle,const uint16_t*gate,const uint16_t*up,
 int start,int rows,int total,int width,const uint16_t*table,uint16_t*product){
 auto*d=static_cast<Matrix*>(handle);
 if(!d||!gate||!up||!table||!product||start<0||rows<=0||total<=0||total>615||start>total||rows>total-start||width!=d->k||width%8)return -1;
 if(!bind(d,rows))return -1;
 d->prepared=0;int m=d->shapes[d->selected].M;auto*out=static_cast<uint16_t*>(d->a->virt_addr);int bad=0;
 #pragma omp parallel for num_threads(3) schedule(static) reduction(|:bad)
 for(int kb=0;kb<width/8;kb++)for(int row=0;row<m;row++){
  uint16x8_t packed=vdupq_n_u16(0);
  if(row<rows){
   size_t at=(size_t(start+row)*width+kb*8);uint16_t activated[8];
   for(int j=0;j<8;j++)activated[j]=table[gate[at+j]];
   uint16x4_t p[2];
   for(int half=0;half<2;half++){
    float32x4_t av=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(activated+half*4),16));
    float32x4_t uv=vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(up+at+half*4),16));
    p[half]=rb(vmulq_f32(av,uv));
   }
   uint16x8_t bf=vcombine_u16(p[0],p[1]);vst1q_u16(product+at,bf);
   packed=vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(vreinterpretq_f32_u32(vshll_n_u16(p[0],16))),vcvt_f16_f32(vreinterpretq_f32_u32(vshll_n_u16(p[1],16)))));
   bad|=vmaxvq_u16(vceqq_u16(vandq_u16(packed,vdupq_n_u16(0x7c00)),vdupq_n_u16(0x7c00)))!=0;
  }
  vst1q_u16(out+(size_t(kb)*m+row)*8,packed);
 }
 if(bad)return -2;
 d->prepared=rows;return 0;
}
