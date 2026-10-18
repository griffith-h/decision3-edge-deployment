// Own FP32 native-layout MatMul MLP; public SDK only. No graph/weight changes.
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "rknn_matmul_api.h"
static thread_local std::string error;
struct Matrix {
 rknn_matmul_ctx ctx=0;rknn_tensor_mem *a=nullptr,*b=nullptr,*c=nullptr;
 int k,n,selected=-1,prepared=0;std::vector<rknn_matmul_shape> shapes;std::vector<rknn_matmul_io_attr> attrs;
 ~Matrix(){if(a)rknn_destroy_mem(ctx,a);if(b)rknn_destroy_mem(ctx,b);if(c)rknn_destroy_mem(ctx,c);if(ctx)rknn_matmul_destroy(ctx);}
};
static bool ok(int rc,const char*op){if(!rc)return true;error=std::string(op)+":"+std::to_string(rc);return false;}
extern "C" const char* nm_error(){return error.c_str();}
extern "C" void* nm_create(int k,int n,const void*weights,int core){
 if(!weights||k<=0||k%16||n<=0||n%8){error="Invalid dimensions";return nullptr;}
 auto p=std::make_unique<Matrix>();p->k=k;p->n=n;for(int m:{128,256,320,512,640})p->shapes.push_back({m,k,n});p->attrs.resize(5);
 rknn_matmul_info info{};info.M=640;info.K=k;info.N=n;info.type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;info.AC_layout=RKNN_MM_LAYOUT_NATIVE;info.B_layout=RKNN_MM_LAYOUT_NATIVE;
 if(!ok(rknn_matmul_create_dynamic_shape(&p->ctx,&info,5,p->shapes.data(),p->attrs.data()),"create"))return nullptr;
 if(!ok(rknn_matmul_set_core_mask(p->ctx,static_cast<rknn_core_mask>(core)),"core"))return nullptr;
 uint32_t as=0,bs=0,cs=0;
 for(int i=0;i<5;i++){
  auto&a=p->attrs[i].A;auto&c=p->attrs[i].C;int m=p->shapes[i].M;
  // Validate actual returned contract instead of assuming a generic header.
  if(a.n_dims!=3||a.dims[0]!=uint32_t(k/8)||a.dims[1]!=uint32_t(m)||a.dims[2]!=8||c.n_dims!=3||c.dims[0]!=uint32_t(n/4)||c.dims[1]!=uint32_t(m)||c.dims[2]!=4||a.size!=uint32_t(k*m*2)||c.size!=uint32_t(n*m*4)){error="Unexpected native layout";return nullptr;}
  as=std::max(as,a.size);bs=std::max(bs,p->attrs[i].B.size);cs=std::max(cs,c.size);
 }
 p->a=rknn_create_mem(p->ctx,as);p->b=rknn_create_mem(p->ctx,bs);p->c=rknn_create_mem(p->ctx,cs);if(!p->a||!p->b||!p->c){error="Allocation failed";return nullptr;}
 if(!ok(rknn_B_normal_layout_to_native_layout(const_cast<void*>(weights),p->b->virt_addr,k,n,&info),"pack_B"))return nullptr;
 return p.release();
}
extern "C" void nm_destroy(void*h){delete static_cast<Matrix*>(h);}
static bool bind(Matrix*p,int rows){
 int index=0;while(index<5&&p->shapes[index].M<rows)index++;if(rows<=0||index==5){error="Row bound";return false;}
 if(p->selected!=index){
  if(!ok(rknn_matmul_set_dynamic_shape(p->ctx,&p->shapes[index]),"shape")||!ok(rknn_matmul_set_io_mem(p->ctx,p->a,&p->attrs[index].A),"bindA")||!ok(rknn_matmul_set_io_mem(p->ctx,p->b,&p->attrs[index].B),"bindB")||!ok(rknn_matmul_set_io_mem(p->ctx,p->c,&p->attrs[index].C),"bindC"))return false;
  p->selected=index;
 }
 return true;
}
static uint16x4_t rb(float32x4_t f){uint32x4_t u=vreinterpretq_u32_f32(f);return vmovn_u32(vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1))),16));}
static bool bad4(float32x4_t f){return vmaxvq_u32(vorrq_u32(vmvnq_u32(vceqq_f32(f,f)),vcgeq_f32(vabsq_f32(f),vdupq_n_f32(65488.f))))!=0;}
extern "C" int nm_prepare(void*h,int rows,const uint16_t*input){
 auto*p=static_cast<Matrix*>(h);if(!p||!input||!bind(p,rows))return -1;p->prepared=0;int m=p->shapes[p->selected].M;auto*dst=static_cast<uint16_t*>(p->a->virt_addr);int bad=0;
 #pragma omp parallel for num_threads(3) reduction(|:bad) schedule(static)
 for(int kb=0;kb<p->k/8;kb++)for(int row=0;row<m;row++){
  uint16x8_t v=vdupq_n_u16(0);
  if(row<rows){uint16x8_t b=vld1q_u16(input+size_t(row)*p->k+kb*8);v=vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b),16))),vcvt_f16_f32(vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b),16)))));bad|=vmaxvq_u16(vceqq_u16(vandq_u16(v,vdupq_n_u16(0x7c00)),vdupq_n_u16(0x7c00)))!=0;}
  vst1q_u16(dst+(size_t(kb)*m+row)*8,v);
 }
 if(bad)return -2;p->prepared=rows;return 0;
}
extern "C" int nm_copy(void*to,void*from,int rows){auto*d=static_cast<Matrix*>(to);auto*s=static_cast<Matrix*>(from);if(!d||!s||s->prepared!=rows||d->k!=s->k||!bind(d,rows))return -1;memcpy(d->a->virt_addr,s->a->virt_addr,d->attrs[d->selected].A.size);d->prepared=rows;return 0;}
extern "C" int nm_raw(void*h,int rows){auto*p=static_cast<Matrix*>(h);if(!p||p->prepared!=rows)return -1;return ok(rknn_matmul_run(p->ctx),"run")?0:-1;}
extern "C" int nm_fuse_slice(void*down,void*gate,void*up,int start,int rows,const uint16_t*table){
 auto*d=static_cast<Matrix*>(down);auto*g=static_cast<Matrix*>(gate);auto*u=static_cast<Matrix*>(up);if(!d||!g||!u||!table||g->n!=d->k||u->n!=d->k||start<0||g->prepared<start+rows||u->prepared<start+rows||!bind(d,rows))return -1;
 d->prepared=0;int m=d->shapes[d->selected].M;int gm=g->shapes[g->selected].M;if(gm!=u->shapes[u->selected].M)return -1;
 auto*ga=static_cast<float*>(g->c->virt_addr);auto*ua=static_cast<float*>(u->c->virt_addr);auto*out=static_cast<uint16_t*>(d->a->virt_addr);int rawbad=0,inputbad=0;
 #pragma omp parallel for num_threads(3) reduction(|:rawbad,inputbad) schedule(static)
 for(int kb=0;kb<d->k/8;kb++)for(int row=0;row<m;row++){
  uint16x8_t packed=vdupq_n_u16(0);
  if(row<rows){uint16x4_t products[2];
   for(int half=0;half<2;half++){
    size_t index=(size_t(kb*2+half)*gm+row+start)*4;float32x4_t gv=vld1q_f32(ga+index),uv=vld1q_f32(ua+index);rawbad|=bad4(gv)||bad4(uv);uint16_t indices[4],act[4];vst1_u16(indices,rb(gv));for(int z=0;z<4;z++)act[z]=table[indices[z]];
    products[half]=rb(vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(act),16)),vreinterpretq_f32_u32(vshll_n_u16(rb(uv),16))));
   }
   packed=vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(vreinterpretq_f32_u32(vshll_n_u16(products[0],16))),vcvt_f16_f32(vreinterpretq_f32_u32(vshll_n_u16(products[1],16)))));inputbad|=vmaxvq_u16(vceqq_u16(vandq_u16(packed,vdupq_n_u16(0x7c00)),vdupq_n_u16(0x7c00)))!=0;
  }
  vst1q_u16(out+(size_t(kb)*m+row)*8,packed);
 }
 if(rawbad)return -3;if(inputbad)return -2;d->prepared=rows;return 0;
}
extern "C" int nm_output(void*h,int rows,uint16_t*out){
 auto*p=static_cast<Matrix*>(h);if(!p||!out||p->prepared!=rows)return -1;int rc=nm_raw(h,rows);if(rc)return rc;int m=p->shapes[p->selected].M;auto*src=static_cast<float*>(p->c->virt_addr);int bad=0;
 #pragma omp parallel for num_threads(3) reduction(|:bad) schedule(static)
 for(int nb=0;nb<p->n/4;nb++)for(int row=0;row<rows;row++){float32x4_t f=vld1q_f32(src+(size_t(nb)*m+row)*4);bad|=bad4(f);vst1_u16(out+size_t(row)*p->n+nb*4,rb(f));}
 return bad?-3:0;
}

extern "C" int nm_fuse(void*d,void*g,void*u,int rows,const uint16_t*table){return nm_fuse_slice(d,g,u,0,rows,table);}
