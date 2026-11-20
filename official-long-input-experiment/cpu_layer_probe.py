"""Counterfactual CPU-fusion reference on identical real layer inputs.

Keep accelerated linears/MLP identical in both calls. Stop at first difference
or after3layers. Not a full model request or performance/accuracy benchmark.
"""
import json
import torch
from medium_cpu_original import references,restored_methods

class ProbeComplete(Exception):pass

def run(model,owner,request,emit,out):
    states=references(model.backbone);ref_by_module={m:ref for _,m,_,ref,_ in states};restores=[];records=[]
    if not states:raise ValueError('No pre-fusion CPU references')
    def save(reason,completed=True):
        (out/'cpu-layer-observations.json').write_text(json.dumps({'completed':completed,'full_request_completed':False,
            'scope':'Same real B8 layer inputs; same accelerated linears/MLP in both; pre-fusion CPU method ablation only',
            'stop_reason':reason,'reference_methods':len(states),'records':records})+'\n')
    try:
        for index,(layer,_) in enumerate(model.encoder.stack[:3]):
            native=layer.forward;reference=ref_by_module.get(layer)
            if reference is None:raise ValueError('Missing original decoder method')
            def observe(*args,_index=index,_native=native,_reference=reference,**kwargs):
                actual=_native(*args,**kwargs)
                with restored_methods(states):expected=_reference(*args,**kwargs)
                if not isinstance(actual,torch.Tensor) or actual.shape!=expected.shape:raise RuntimeError('Unexpected decoder output contract')
                a=actual.reshape(-1,actual.shape[-1]);b=expected.reshape_as(a);mismatches=0;maximum=0.
                for start in range(0,len(a),512):
                    aa=a[start:start+512];bb=b[start:start+512];mismatches+=int((aa!=bb).sum());maximum=max(maximum,float((aa.float()-bb.float()).abs().max()))
                records.append({'layer':_index,'shape':list(actual.shape),'bf16_mismatches':mismatches,'maximum_delta':maximum})
                save('continue',False)
                if mismatches:save('first_layer_difference');raise ProbeComplete
                if _index==2:save('first3layers_equal');raise ProbeComplete
                return actual
            layer.forward=observe;restores.append((layer,native))
        emit({'event':'request_begin','phase':'same_input_original_CPU_layer_ablation'})
        try:model.system_one(**request)
        except ProbeComplete:pass
        else:raise AssertionError('Diagnostic failed to stop')
    finally:
        for layer,native in reversed(restores):layer.forward=native
    emit({'event':'complete','layers':len(records),'full_request_completed':False})
