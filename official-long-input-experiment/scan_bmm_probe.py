"""Actual scan BMM operands, preserved strides, CPU thread component preflight."""
import json,math,statistics,time
import torch

def preserved(x):
    if any(s<0 for s in x.stride()):raise ValueError('Negative diagnostic stride')
    span=1+sum((n-1)*s for n,s in zip(x.shape,x.stride()))
    storage=x.as_strided((span,),(1,),x.storage_offset()).detach().clone()
    result=storage.as_strided(x.shape,x.stride(),0)
    if not torch.equal(result,x):raise AssertionError('Captured input changed values')
    return result

def run(model,owner,request,emit,out):
    captures={};original=owner.long.original_scan;visited=False
    original_matmul=torch.Tensor.__matmul__
    def matmul(a,b):
        key=(math.prod(a.shape[:-2]),*a.shape[-2:]) if isinstance(a,torch.Tensor) and a.ndim>=3 else None
        capture=key in ((32,128,128),(320,128,128)) and key not in captures
        if capture:
            if not isinstance(b,torch.Tensor) or any(x.dtype!=torch.float32 for x in (a,b)):raise ValueError('Unexpected actual matmul contract')
            inputs=[preserved(x) for x in (a,b)]
        value=original_matmul(a,b)
        if capture:captures[key]={'inputs':inputs,'expected':value.detach().clone(),'input_strides':[list(x.stride()) for x in (a,b)],'actual_shapes':[list(x.shape) for x in (a,b)]}
        return value
    def scan(*args,**kwargs):
        nonlocal visited
        if visited:return original(*args,**kwargs)
        visited=True;torch.Tensor.__matmul__=matmul
        try:return original(*args,**kwargs)
        finally:torch.Tensor.__matmul__=original_matmul
    owner.long.original_scan=scan;threads=torch.get_num_threads();records=[];response=None
    target=out/'scan-bmm-preflight.json'
    def save(completed=False):
        target.write_text(json.dumps({'completed':completed,'response':response,'records':records,'thread_setting_restored':torch.get_num_threads()==threads,
            'scope':'Actual CPU scan component only, thread switching excluded from op timing and reported separately; not an engine speed result'},indent=2)+'\n')
    try:
        emit({'event':'ready','case':'scan-bmm-preflight'});emit({'event':'request_begin','phase':'capture_complete_request','budget_seconds':120})
        response=model.system_one(**request);save()
        if set(captures)!={(32,128,128),(320,128,128)}:raise AssertionError('Required actual scan shapes not captured')
        emit({'event':'request_begin','phase':'cpu_component_pairs','budget_seconds':120})
        with torch.inference_mode():
            for shape,entry in captures.items():
                a,b=entry['inputs'];expected=entry['expected'];times={1:[],3:[]};switches=[];parity=True
                replay=original_matmul(a,b)
                if not torch.equal(replay.view(torch.int32),expected.view(torch.int32)):raise AssertionError('Original3-thread replay did not reproduce live FP32 BMM')
                for pair in range(7):
                    for count in ((3,1) if pair%2==0 else (1,3)):
                        begin=time.perf_counter();torch.set_num_threads(count);switches.append(time.perf_counter()-begin)
                        begin=time.perf_counter();value=original_matmul(a,b);times[count].append(time.perf_counter()-begin)
                        equal=torch.equal(value.view(torch.int32),expected.view(torch.int32));parity=parity and equal
                        if not equal:break
                    if not parity:break
                torch.set_num_threads(threads)
                records.append({'shape':list(shape),'input_strides':entry['input_strides'],'actual_input_shapes':entry['actual_shapes'],'original_live_replay_equal':True,'bitwise_equal':parity,
                    'times':times,'thread_switch_seconds':switches,'median_pair_ratio_1_over_3':statistics.median(b/a for a,b in zip(times[3],times[1])) if parity else None})
                save()
        save(True);emit({'event':'complete','components':len(records),'all_fp32_bitwise_equal':all(r['bitwise_equal'] for r in records)})
    finally:
        torch.set_num_threads(threads)
        if owner.long.original_scan is scan:owner.long.original_scan=original
