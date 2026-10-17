import sys,json,time,ctypes,statistics
sys.path[:0]=['/base','/scan','/release']
import torch
from scan import recurrent
from pathlib import Path
torch.set_num_threads(3)
lib=ctypes.CDLL('/work/libscan_fast.so');lib.d3_scan_fast.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*6;lib.d3_scan_fast.restype=ctypes.c_int

def fast(q,k,v,g,beta,use_qk_l2norm_in_kernel=True,mode=6):
    q,k,v,beta,g=[x.transpose(1,2).float().contiguous() for x in (q,k,v,beta,g)]
    if use_qk_l2norm_in_kernel:
        q=q*torch.rsqrt((q*q).sum(-1,keepdim=True)+1e-6);k=k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
    q=q*q.shape[-1]**-.5;decay=g.exp();out=torch.empty(v.shape,dtype=torch.bfloat16)
    rc=lib.d3_scan_fast(*(x.data_ptr() for x in (q,k,v,decay,beta,out)),q.shape[0]*q.shape[1],q.shape[2],q.shape[3],v.shape[3],mode,3)
    assert rc==0
    return out.transpose(1,2).contiguous(),None
if __name__=='__main__':
    captures=torch.load('/work/scan-inputs.pt',weights_only=False);records=[]
    funcs={'baseline':lambda *xs:recurrent(*xs,use_qk_l2norm_in_kernel=True,tile=16,fused=5),'decay-late':lambda *xs:fast(*xs,mode=6),'readout-identity':lambda *xs:fast(*xs,mode=7)}
    for c in captures:
        xs=c['tensors'];ref=funcs['baseline'](*xs)[0];row={'shapes':[list(x.shape) for x in xs],'strides':[list(x.stride()) for x in xs],'times':{k:[] for k in funcs},'errors':{}}
        for name,fn in funcs.items():
            out=fn(*xs)[0];row['errors'][name]={'max_abs':(out.float()-ref.float()).abs().max().item(),'mean_abs':(out.float()-ref.float()).abs().mean().item()}
        for i in range(15):
            for name in (list(funcs) if i%2==0 else list(reversed(funcs))):
                t=time.perf_counter();out=funcs[name](*xs)[0];row['times'][name].append(time.perf_counter()-t)
        records.append(row);print(row,flush=True)
    Path('/work/logs/replay-scan.json').write_text(json.dumps(records,indent=2)+'\n')
