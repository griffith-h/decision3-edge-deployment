// Own activation-to-activation MatMul wrapper, using only public RKNN API.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include "rknn_matmul_api.h"
static thread_local std::string error;
struct Matrix {
 rknn_matmul_ctx ctx=0; rknn_tensor_mem *a=nullptr,*b=nullptr,*c=nullptr;
 rknn_matmul_info info{}; rknn_matmul_io_attr attr{};
 int m,k,n;bool ready=false;
 ~Matrix(){if(a)rknn_destroy_mem(ctx,a);if(b)rknn_destroy_mem(ctx,b);if(c)rknn_destroy_mem(ctx,c);if(ctx)rknn_matmul_destroy(ctx);}
};
static bool check(int code,const char*op){if(!code)return true;error=std::string(op)+" returned "+std::to_string(code);return false;}
extern "C" const char* dm_error(){return error.c_str();}
extern "C" void* dm_create(int m,int k,int n,int core){
 error.clear();if(m<=0||m>2048||k<=0||k>10240||k%16||n<=0||n%8){error="Invalid aligned matrix shape";return nullptr;}
 auto p=std::make_unique<Matrix>();p->m=m;p->k=k;p->n=n;
 p->info.M=m;p->info.K=k;p->info.N=n;p->info.type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;p->info.B_layout=RKNN_MM_LAYOUT_NATIVE;p->info.AC_layout=RKNN_MM_LAYOUT_NORM;
 if(!check(rknn_matmul_create(&p->ctx,&p->info,&p->attr),"create"))return nullptr;
 if(!check(rknn_matmul_set_core_mask(p->ctx,static_cast<rknn_core_mask>(core)),"set_core"))return nullptr;
 p->a=rknn_create_mem(p->ctx,p->attr.A.size);p->b=rknn_create_mem(p->ctx,p->attr.B.size);p->c=rknn_create_mem(p->ctx,p->attr.C.size);
 if(!p->a||!p->b||!p->c){error="allocation failed";return nullptr;}
 if(!check(rknn_matmul_set_io_mem(p->ctx,p->a,&p->attr.A),"bind A")||!check(rknn_matmul_set_io_mem(p->ctx,p->b,&p->attr.B),"bind B")||!check(rknn_matmul_set_io_mem(p->ctx,p->c,&p->attr.C),"bind C"))return nullptr;
 return p.release();
}
extern "C" int dm_set_b(void*handle,const void*data){
 auto*p=static_cast<Matrix*>(handle);if(!p||!data)return -1;p->ready=false;
 if(!check(rknn_B_normal_layout_to_native_layout(const_cast<void*>(data),p->b->virt_addr,p->k,p->n,&p->info),"pack B"))return -1;
 p->ready=true;return 0;
}
extern "C" int dm_run(void*handle,const void*data,float*out,int rows){
 auto*p=static_cast<Matrix*>(handle);if(!p||!data||!out||!p->ready||rows<=0||rows>p->m)return -1;
 std::memcpy(p->a->virt_addr,data,size_t(rows)*p->k*2);
 std::memset(static_cast<char*>(p->a->virt_addr)+size_t(rows)*p->k*2,0,size_t(p->m-rows)*p->k*2);
 if(!check(rknn_matmul_run(p->ctx),"run"))return -1;
 std::memcpy(out,p->c->virt_addr,size_t(rows)*p->n*4);return 0;
}
extern "C" void dm_destroy(void*handle){delete static_cast<Matrix*>(handle);}
