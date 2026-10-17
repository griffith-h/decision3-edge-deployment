"""Replay actual SiLU calls: BF16 Torch, FP32 Torch, own SIMD polynomial."""
import argparse,ctypes,json,statistics,time
from pathlib import Path
import torch
from common import load,sources,stats,compare
def run_probe(model):
    from cpu_math import activation
    import torch.nn.functional as F
    original=F.silu;captured=[]
    def capture(x,inplace=False):
        out=original(x,inplace=inplace)
        if x.device.type=='cpu' and x.dtype==torch.bfloat16:captured.append((x,out))
        return out
    body=json.loads(Path('/base/request.json').read_text())
    result={'sources':sources(),'variants':{}}
    try:
        F.silu=capture
        response=model.system_one(state=body['state'],questions=body['questions'])
        F.silu=original
        assert len(captured)==18,len(captured)
        torchlib=ctypes.CDLL(str(Path(torch.__file__).parent/'lib/libtorch_cpu.so'))
        result['public_sleef_symbols']={name:hasattr(torchlib,name) for name in ('Sleef_expf4_u10advsimd','Sleef_expf4_u10','Sleef_expf8_u10')}
        choices={'torch_bf16':original,'torch_fp32':lambda x:original(x.float()).bfloat16(),'native_poly':activation}
        with torch.inference_mode():
            # A strictly numeric check, without model or SDK work in the timing.
            values=torch.arange(65536,dtype=torch.int32).short().view(torch.bfloat16)
            expected=original(values);actual=choices['torch_fp32'](values)
            finite=expected.isfinite();result['fp32_vs_bf16_exhaustive']={'nan_classes_equal':torch.equal(actual.isnan(),expected.isnan()),
                'finite_bitwise_equal':torch.equal(actual[finite].view(torch.int16),expected[finite].view(torch.int16)),
                'max_finite_bf16_ulp':(actual[finite].view(torch.int16).int()-expected[finite].view(torch.int16).int()).abs().max().item()}
            for name,fn in choices.items():
                rows=[]
                for x,ref in captured:
                    times=[]
                    for _ in range(15):
                        t=time.perf_counter();out=fn(x);times.append(time.perf_counter()-t)
                    rows.append({'shape':list(x.shape),**stats(times),'bitwise_equal':torch.equal(out,ref),
                                 'max_delta':(out.float()-ref.float()).abs().max().item(),
                                 'fraction_abs_over80':(x.float().abs()>80).float().mean().item()})
                result['variants'][name]={'calls':rows,'sum_medians':sum(r['median'] for r in rows)}
                print('ACTIVATION',name,result['variants'][name]['sum_medians'],flush=True)
        result['passed']=True
        Path('/work/logs/activation-probe.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:F.silu=original

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);a=p.parse_args()
    model,engine,native,close=load(a.model)
    try:run_probe(model)
    finally:close()
