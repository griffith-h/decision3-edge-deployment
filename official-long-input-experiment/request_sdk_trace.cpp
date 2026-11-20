// Diagnostic interposition of the public synchronous SDK call. No math changes.
#include "rknn_matmul_api.h"
#include <atomic>
#include <cstdint>
#include <dlfcn.h>
#include <mutex>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
struct Event {uint64_t start_ns,end_ns,cpu_start_ns,cpu_end_ns,thread,context;int64_t status;};
static constexpr uint64_t limit=32768;
static Event events[limit];
static std::atomic<uint64_t> used{0};
static std::atomic<bool> enabled{false};
static uint64_t now(clockid_t clock){timespec t{};clock_gettime(clock,&t);return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;}
using Run=int(*)(rknn_matmul_ctx);
static Run real_run(){
 static std::once_flag once;static Run fn=nullptr;
 std::call_once(once,[]{void* h=dlopen("/base/sdk/librknnrt.so",RTLD_NOW|RTLD_LOCAL);if(h)fn=reinterpret_cast<Run>(dlsym(h,"rknn_matmul_run"));});
 return fn;
}
extern "C" int rknn_matmul_run(rknn_matmul_ctx ctx){
 auto fn=real_run();if(!fn)return -1;
 if(!enabled.load(std::memory_order_relaxed))return fn(ctx);
 uint64_t index=used.fetch_add(1);Event local{};local.thread=syscall(SYS_gettid);local.context=ctx;
 local.start_ns=now(CLOCK_MONOTONIC);local.cpu_start_ns=now(CLOCK_THREAD_CPUTIME_ID);
 int rc=fn(ctx);local.end_ns=now(CLOCK_MONOTONIC);local.cpu_end_ns=now(CLOCK_THREAD_CPUTIME_ID);local.status=rc;
 if(index<limit)events[index]=local;
 return rc;
}
extern "C" void ledger_sdk_begin(){used=0;enabled=true;}
extern "C" uint64_t ledger_sdk_end(){enabled=false;return used.load();}
extern "C" const Event* ledger_sdk_events(){return events;}
extern "C" uint64_t ledger_sdk_capacity(){return limit;}
