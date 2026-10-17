#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "rknn_matmul_api.h"

// Experimental, synchronous, single-owner handle. Callers must serialize calls.
static thread_local std::string last_error;
static thread_local int requested_core = 0;
extern "C" void mm_test_core(int mask) { requested_core = mask; }
struct Matrix {
  rknn_matmul_ctx ctx = 0;
  rknn_tensor_mem *a = nullptr, *b = nullptr, *c = nullptr;
  int k, n, selected = -1;
  std::vector<rknn_matmul_shape> shapes;
  std::vector<rknn_matmul_io_attr> attrs;
  ~Matrix() {
    if (a) rknn_destroy_mem(ctx, a);
    if (b) rknn_destroy_mem(ctx, b);
    if (c) rknn_destroy_mem(ctx, c);
    if (ctx) rknn_matmul_destroy(ctx);
  }
};
static bool check(int code, const char* operation) {
  if (code == 0) return true;
  last_error = std::string(operation) + " returned " + std::to_string(code);
  return false;
}
extern "C" const char* mm_error() { return last_error.c_str(); }
extern "C" void* mm_create(int k, int n, const void* half_kn) {
  last_error.clear();
  if (k <= 0 || k > 10240 || k % 16 || n <= 0 || n % 8 || !half_kn) {
    last_error = "Unsupported K/N alignment or missing weights"; return nullptr;
  }
  auto p = std::make_unique<Matrix>();
  p->k = k; p->n = n;
  for (int m : {128, 256, 512}) p->shapes.push_back({m,k,n});
  p->attrs.resize(p->shapes.size());
  rknn_matmul_info info{};
  info.M=512; info.K=k; info.N=n;
  info.type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
  info.B_layout=RKNN_MM_LAYOUT_NATIVE;
  info.AC_layout=RKNN_MM_LAYOUT_NORM;
  if (!check(rknn_matmul_create_dynamic_shape(&p->ctx, &info,
      p->shapes.size(), p->shapes.data(), p->attrs.data()), "create_dynamic")) return nullptr;
  // RK3576 matmul rejects combined core mask 3; use the supported auto mode.
  if (!check(rknn_matmul_set_core_mask(p->ctx, static_cast<rknn_core_mask>(requested_core)), "set_cores")) return nullptr;
  uint32_t as=0, bs=0, cs=0;
  for (auto& attr : p->attrs) {
    as=std::max(as,attr.A.size); bs=std::max(bs,attr.B.size); cs=std::max(cs,attr.C.size);
  }
  p->a=rknn_create_mem(p->ctx,as); p->b=rknn_create_mem(p->ctx,bs); p->c=rknn_create_mem(p->ctx,cs);
  if (!p->a || !p->b || !p->c) { last_error="rknn_create_mem failed"; return nullptr; }
  if (!check(rknn_B_normal_layout_to_native_layout(const_cast<void*>(half_kn),p->b->virt_addr,k,n,&info), "pack_B")) return nullptr;
  if (!check(rknn_matmul_set_dynamic_shape(p->ctx,&p->shapes[0]),"initial_shape")) return nullptr;
  if (!check(rknn_matmul_set_io_mem(p->ctx,p->b,&p->attrs[0].B),"bind_B")) return nullptr;
  std::memset(p->a->virt_addr,0,as);
  return p.release();
}
extern "C" int mm_run(void* handle, int rows, const void* half_mk, float* out_mn) {
  auto* p=static_cast<Matrix*>(handle);
  if (!p || rows<=0 || rows>512 || !half_mk || !out_mn) { last_error="Invalid run argument"; return -1; }
  int index=rows<=128 ? 0 : rows<=256 ? 1 : 2;
  if (p->selected!=index) {
    if (!check(rknn_matmul_set_dynamic_shape(p->ctx,&p->shapes[index]),"set_shape")) return -1;
    if (!check(rknn_matmul_set_io_mem(p->ctx,p->a,&p->attrs[index].A),"bind_A")) return -1;
    if (!check(rknn_matmul_set_io_mem(p->ctx,p->c,&p->attrs[index].C),"bind_C")) return -1;
    p->selected=index;
  }
  size_t bytes=size_t(rows)*p->k*2;
  std::memcpy(p->a->virt_addr,half_mk,bytes);
  std::memset(static_cast<char*>(p->a->virt_addr)+bytes,0,size_t(p->shapes[index].M-rows)*p->k*2);
  if (!check(rknn_matmul_run(p->ctx),"run")) return -1;
  std::memcpy(out_mn,p->c->virt_addr,size_t(rows)*p->n*sizeof(float));
  return 0;
}
extern "C" void mm_destroy(void* handle) { delete static_cast<Matrix*>(handle); }
