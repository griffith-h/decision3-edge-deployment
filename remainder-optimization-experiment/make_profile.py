"""Add clocks only to our own public-API bridge, never a vendor binary."""
from pathlib import Path
import sys
half="--half" in sys.argv
root=Path(__file__).resolve().parent
s=(root/'half_bridge.cpp').read_text() if half else (root.parent/'bridge-parallel-experiment/fused_bridge.cpp').read_text()
s=s.replace('#include <algorithm>','#include <algorithm>\n#include <chrono>\nstatic double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }')
s=s.replace('int k, n, selected = -1, prepared_rows = 0;','int k, n, selected = -1, prepared_rows = 0;\n  double profile[8] = {};')
s=s.replace('extern "C" int mm_prepare_bf16', '''extern "C" void mm_profile_read(void* handle,double* out,int reset) {
  auto* p=static_cast<Matrix*>(handle);
  std::memcpy(out,p->profile,sizeof(p->profile));
  if (reset) std::memset(p->profile,0,sizeof(p->profile));
}
extern "C" int mm_prepare_bf16''')
s=s.replace('if (!input || !bind_shape(p,rows)) return -1;', 'double stamp=now();\n  if (!input || !bind_shape(p,rows)) return -1;\n  p->profile[1]+=now()-stamp;stamp=now();')
s=s.replace('if (status) { last_error="FP16 activation overflow"; return status; }', 'p->profile[2]+=now()-stamp;stamp=now();\n  if (status) { last_error="FP16 activation overflow"; return status; }')
s=s.replace('p->prepared_rows=rows;\n  return 0;', 'p->profile[3]+=now()-stamp;\n  p->prepared_rows=rows;\n  return 0;',1)
s=s.replace('if (!src || !dst || src->k!=dst->k', 'double stamp=now();\n  if (!src || !dst || src->k!=dst->k')
s=s.replace('dst->prepared_rows=rows;', 'dst->profile[3]+=now()-stamp;\n  dst->prepared_rows=rows;')
needle='if (!check(rknn_matmul_run(p->ctx),"run")) return -1;'
# Only the fused prepared path, not the unused legacy entry point.
before,after=s.rsplit(needle,1)
s=before+'''double stamp=now();p->profile[6]=stamp;
  if (!check(rknn_matmul_run(p->ctx),"run")) return -1;
  p->profile[7]=now();p->profile[4]+=p->profile[7]-stamp;p->profile[0]+=1;
  stamp=now();'''+after
s=s.replace('''  return 0;
}
extern "C" int mm_run_bf16''','''  p->profile[5]+=now()-stamp;
  return 0;
}
extern "C" int mm_run_bf16''')
(root/('half_profile.cpp' if half else 'fused_profile.cpp')).write_text(s)
