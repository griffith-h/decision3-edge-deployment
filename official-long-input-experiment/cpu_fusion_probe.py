"""Find the first CPU-fusion mismatch on the original full-B8 input.

Native outputs remain unchanged; stop on the first difference, or after the
first decoder layer. Partial model diagnostic, not an accuracy measurement.
"""
import inspect,json
import torch

class ProbeComplete(Exception):pass

def run(model,owner,request,emit,out):
    restores=[];records=[];seen=set();reason='first_layer_complete'
    def save():
        (out/'cpu-fusion-observations.json').write_text(json.dumps({'completed':True,'full_request_completed':False,
            'scope':'First real8x1181 CPU fusions; reference is captured pre-fusion current CPU method, not raw Torch BF16',
            'stop_reason':reason,'records':records})+'\n')
    try:
        for name,module in model.backbone.named_modules():
            if type(module).__name__ not in ('Qwen3_5RMSNorm','Qwen3_5RMSNormGated','Linear'):continue
            old=module.forward;function=getattr(old,'__func__',old)
            if getattr(function,'__module__','') not in ('lut_ops','small_linear'):continue
            parameter=inspect.signature(old).parameters.get('_old')
            if parameter is None or parameter.default is inspect.Parameter.empty or not callable(parameter.default):continue
            reference=parameter.default
            def observe(*args,_old=old,_reference=reference,_name=name,_kind=function.__module__,**kwargs):
                nonlocal reason
                actual=_old(*args,**kwargs)
                if _name in seen:return actual
                expected=_reference(*args,**kwargs);seen.add(_name)
                if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise RuntimeError('Fusion reference changed output contract')
                delta=(actual.float()-expected.float()).abs();mismatches=int((actual!=expected).sum())
                records.append({'module':_name,'kind':_kind,'input_shape':list(args[0].shape),'output_shape':list(actual.shape),
                    'bf16_mismatches':mismatches,'maximum_delta':float(delta.max())})
                if mismatches:reason='first_fusion_difference';save();raise ProbeComplete
                return actual
            module.forward=observe;restores.append((module,old))
        layer=owner.model.encoder.stack[0][0];old=layer.forward
        def stop_after_first(*args,**kwargs):
            value=old(*args,**kwargs);save();raise ProbeComplete
        layer.forward=stop_after_first;restores.append((layer,old))
        emit({'event':'request_begin','phase':'first_original_B8_CPU_fusions'})
        try:model.system_one(**request)
        except ProbeComplete:pass
        else:raise AssertionError('Diagnostic failed to stop in first layer')
        if not records:raise RuntimeError('No captured CPU fusion reference was observed')
    finally:
        for module,old in reversed(restores):module.forward=old
    emit({'event':'complete','fusion_modules':len(records),'full_request_completed':False,'stop_reason':reason})
