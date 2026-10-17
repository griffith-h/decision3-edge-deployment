import baseline
import ctypes,json,time,statistics,torch
from pathlib import Path
from prepared_scan import prepare
from scan import lib as original
lib=ctypes.CDLL('/work/libchunk_scan_v2.so');lib.d3_chunk_scan.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*6;lib.d3_chunk_scan.restype=ctypes.c_int
torch.set_num_threads(3);torch.manual_seed(629)
def execute(xs,block=8,tile=16):
    q,k,v,d,b=xs;h,t,kd=q.shape;vd=v.shape[-1];out=torch.empty(h,t,vd,dtype=torch.bfloat16)
    rc=lib.d3_chunk_scan(*(x.data_ptr() for x in (*xs,out)),h,t,kd,vd,block,tile);assert rc==0,rc
    return out
def reference(xs):
    q,k,v,d,b=[x.double() for x in xs];state=torch.zeros(q.shape[0],q.shape[2],v.shape[2],dtype=torch.float64);outs=[]
    for i in range(q.shape[1]):
        state=state*d[:,i,None,None];delta=(v[:,i]-(state*k[:,i,:,None]).sum(1))*b[:,i,None];state+=k[:,i,:,None]*delta[:,None,:];outs.append((state*q[:,i,:,None]).sum(1))
    return torch.stack(outs,1).bfloat16()
def baseline_kernel(xs):
    q,k,v,d,b=xs;out=torch.empty_like(v,dtype=torch.bfloat16);rc=original.d3_scan(*(x.data_ptr() for x in (*xs,out)),q.shape[0],q.shape[1],q.shape[2],v.shape[2],16,3,5);assert rc==0;return out
variants=[(b,t) for b in (4,8,16,32) for t in (8,16,32)]
units=[]
with torch.inference_mode():
    for h,t,kd,vd in [(1,1,4,4),(2,3,12,20),(1,4,8,8),(2,5,128,128),(1,7,64,12),(1,8,16,16),(2,9,12,20),(1,16,128,128),(1,17,64,32),(1,115,128,128),(1,129,32,16),(1,512,128,128)]:
        q,k=[torch.nn.functional.normalize(torch.randn(h,t,kd),dim=-1) for _ in range(2)];q*=kd**-.5;v=torch.randn(h,t,vd);d=torch.rand(h,t)*.8+.15;b=torch.rand(h,t)
        if t>3:d[:,1]=0;d[:,2]=1e-35;b[:,3]=0
        xs=(q,k,v,d,b);saved=[x.clone() for x in xs];ref=reference(xs)
        for block,tile in variants:
            out=execute(xs,block,tile);assert torch.allclose(out.float(),ref.float(),atol=.001,rtol=.008),(h,t,kd,vd,block,tile,(out.float()-ref.float()).abs().max())
            if t in (9,17,115):
                n=min(t-1,block+1);prefix=tuple(x[:,:n].contiguous() for x in xs);assert torch.equal(out[:,:n],execute(prefix,block,tile))
            assert all(torch.equal(x,y) for x,y in zip(xs,saved));units.append([h,t,kd,vd,block,tile])
    # Different intervening requests cannot change a repeated request.
    first=execute(xs);execute(tuple(torch.zeros_like(x) for x in xs));assert torch.equal(first,execute(xs))
    records=[]
    captured=torch.load('/previous/profile-candidate-inputs.pt',weights_only=False)
    for c in captured:
        q,k,v,g,beta=c['tensors'];q,k,v=prepare(q,k,v,True);d=g.transpose(1,2).float().contiguous().exp();b=beta.transpose(1,2).float().contiguous()
        xs=tuple(x.flatten(0,1) for x in (q,k,v,d,b));ref=baseline_kernel(xs)
        funcs={'baseline':baseline_kernel,**{f'b{block}t{tile}':lambda xs,block=block,tile=tile:execute(xs,block,tile) for block,tile in variants}}
        row={'times':{name:[] for name in funcs},'max_abs':{}}
        for name,fn in funcs.items():
            out=fn(xs);row['max_abs'][name]=(out.float()-ref.float()).abs().max().item();assert torch.allclose(out.float(),ref.float(),atol=.001,rtol=.008),(name,row['max_abs'][name])
        for i in range(10):
            names=list(funcs);names=names[i%len(names):]+names[:i%len(names)]
            for name in names:
                t=time.perf_counter();out=funcs[name](xs);row['times'][name].append(time.perf_counter()-t)
        records.append(row);print('REPLAY',len(records),{name:statistics.median(v)*1000 for name,v in row['times'].items()},flush=True)
result={'passed':True,'unit_cases':len(units),'records':records,'sum_layer_medians_ms':{name:sum(statistics.median(r['times'][name]) for r in records)*1000 for name in records[0]['times']}}
Path('/work/logs/chunk-v2-unit-replay.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',result['sum_layer_medians_ms'],flush=True)
