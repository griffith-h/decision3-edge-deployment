"""Public SDK constant matrices: inspect small FP16 inputs and power-of-two scales."""
import json

def run(out):
    import torch
    from long_npu_attention_candidate import lib
    rows=[];handle=None;complete=False
    try:
        handle=lib.la_create(128,256,640,1)
        if not handle:raise RuntimeError(lib.la_error().decode())
        for name,av,bv in (('normal',1.,2**-8),('small_weight',1.,2**-15),
                           ('small_activation',2**-15,1.),('smaller_weight',1.,2**-20)):
            for sa,sb in ((1,1),(1,16),(16,16),(1,256)):
                a=torch.full((128,256),av*sa,dtype=torch.float16)
                b=torch.full((256,640),bv*sb,dtype=torch.float16)
                result=torch.empty((128,640),dtype=torch.float32)
                if lib.la_set_b(handle,b.data_ptr()) or lib.la_run(handle,a.data_ptr(),result.data_ptr(),128):raise RuntimeError(lib.la_error().decode())
                corrected=result/(sa*sb);expected=256*av*bv
                rows.append({'case':name,'activation_scale':sa,'weight_scale':sb,
                    'activation_bits':int(a.view(torch.int16)[0,0]),'weight_bits':int(b.view(torch.int16)[0,0]),
                    'expected':expected,'actual':float(corrected[0,0]),
                    'max_absolute_error':float((corrected-expected).abs().max())})
        complete=True
    finally:
        if handle:lib.la_destroy(handle)
        (out/'fp16-range-probe.json').write_text(json.dumps({'completed':complete,'handle_closed':True,
            'scope':'Synthetic public SDK MatMul, no model or accuracy/performance conclusion','records':rows})+'\n')
