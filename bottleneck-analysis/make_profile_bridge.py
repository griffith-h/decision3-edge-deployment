"""Add wall clocks to our public-API shim without touching SDK binaries."""
from pathlib import Path
root=Path(__file__).resolve().parent
s=(root.parent/'rknn-matmul-experiment/matmul_bridge.cpp').read_text()
s=s.replace('#include <algorithm>', '#include <algorithm>\n#include <chrono>')
s=s.replace('// Experimental, synchronous', '''static thread_local double mm_times[6] = {};
static double clock_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
extern "C" void mm_profile_reset() { std::fill(mm_times, mm_times+6, 0.0); }
extern "C" void mm_profile_read(double* values) { std::copy(mm_times, mm_times+6, values); }
// Experimental, synchronous''',1)
s=s.replace('  int index=rows<=128', '  double stage=clock_seconds();\n  mm_times[0] += 1;\n  int index=rows<=128',1)
s=s.replace('  size_t bytes=size_t(rows)*p->k*2;', '  mm_times[1] += clock_seconds()-stage; stage=clock_seconds();\n  size_t bytes=size_t(rows)*p->k*2;',1)
s=s.replace('  std::memset(static_cast<char*>(p->a->virt_addr)+bytes', '  mm_times[2] += clock_seconds()-stage; stage=clock_seconds();\n  std::memset(static_cast<char*>(p->a->virt_addr)+bytes',1)
s=s.replace('  if (!check(rknn_matmul_run(p->ctx)', '  mm_times[3] += clock_seconds()-stage; stage=clock_seconds();\n  if (!check(rknn_matmul_run(p->ctx)',1)
s=s.replace('  std::memcpy(out_mn,p->c->virt_addr', '  mm_times[4] += clock_seconds()-stage; stage=clock_seconds();\n  std::memcpy(out_mn,p->c->virt_addr',1)
s=s.replace('  return 0;\n}', '  mm_times[5] += clock_seconds()-stage;\n  return 0;\n}',1)
(root/'matmul_profile.cpp').write_text(s)
