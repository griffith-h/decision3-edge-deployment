import sys,json,time
sys.path[:0]=['/base','/release']
import torch
from cpu_attention_optimized import install_conv
from native_norm import install_native_conv
from pathlib import Path
torch.set_num_threads(3)
install_conv();install_native_conv()
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
original=qwen.causal_conv1d_fn
from conv_btc import install
restore,counts=install();candidate=qwen.causal_conv1d_fn
rows=[];torch.manual_seed(527)
with torch.inference_mode():
    for batch,length,channels in [(1,1,4),(1,2,12),(2,3,16),(2,5,28),(1,115,6144),(1,129,128)]:
        x=torch.randn(batch,length,channels).bfloat16().transpose(1,2);w=torch.randn(channels,4).bfloat16()
        for hasbias in [False,True]:
            bias=torch.randn(channels).bfloat16() if hasbias else None
            for activation in [None,'silu']:
                saved=x.clone();saved_w=w.clone()
                actual=candidate(x,w,bias,activation=activation);expected=original(x,w,bias,activation=activation)
                assert torch.equal(actual,expected),(batch,length,channels,hasbias,activation,(actual.float()-expected.float()).abs().max())
                assert torch.equal(x,saved) and torch.equal(w,saved_w)
                rows.append([batch,length,channels,hasbias,activation])
    # Non-BTC layout/FP32/gradient/kwargs all follow the frozen implementation.
    for x,w,kw in [(torch.randn(1,8,5).bfloat16(),torch.randn(8,4).bfloat16(),{}),(torch.randn(1,8,5),torch.randn(8,4),{}),(torch.randn(1,5,8).bfloat16().transpose(1,2),torch.randn(8,4).bfloat16(),{'use_cache':False})]:
        assert torch.equal(candidate(x,w,**kw),original(x,w,**kw))
    records=[]
    for c in torch.load('/work/conv-inputs.pt',weights_only=False):
        x,w,bias,activation=c['x'],c['w'],c['bias'],c['activation'];assert tuple(x.stride())==tuple(c['stride'])
        assert torch.equal(candidate(x,w,bias,activation=activation),original(x,w,bias,activation=activation))
        times={'baseline':[],'btc':[]}
        for i in range(20):
            for name,fn in ([('baseline',original),('btc',candidate)] if i%2==0 else [('btc',candidate),('baseline',original)]):
                t=time.perf_counter();fn(x,w,bias,activation=activation);times[name].append(time.perf_counter()-t)
        records.append(times)
with torch.enable_grad():
    x=torch.randn(1,3,8).bfloat16().transpose(1,2).requires_grad_(True);w=torch.randn(8,4).bfloat16()
    a=candidate(x,w,activation='silu');b=original(x,w,activation='silu')
    assert torch.equal(a,b) and a.requires_grad
restore();assert qwen.causal_conv1d_fn is original
Path('/work/logs/conv-unit-replay.json').write_text(json.dumps({'passed':True,'unit_cases':rows,'counts':counts,'replay':records},indent=2)+'\n')
print('PASS',len(rows),counts,flush=True)
