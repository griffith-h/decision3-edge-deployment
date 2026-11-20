"""Two frozen questions: isolate existing components, then real full-batch CPU.

Diagnostics only: modes share one process and their timings are not a final
performance comparison. Labels and expected outputs never reach this module.
"""
import copy,json,time
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def perturb(owner,kinds):
    if set(kinds)-{'scan','attention','mlp','projection'}:raise ValueError('Unknown ablation component')
    changed=[]
    def replace(obj,name,value):changed.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    try:
        replace(owner.long,'enabled',set(owner.long.enabled)-set(kinds))
        if 'mlp' in kinds:replace(owner.long_mlp,'disabled',set(owner.long_mlp.disabled)|{id(s[0]) for s in owner.long_mlp.states})
        if 'projection' in kinds:replace(owner.projections,'disabled',set(owner.projections.disabled)|{id(m) for m in owner.projections.modules.values()})
        yield
    finally:
        for obj,name,value in reversed(changed):setattr(obj,name,value)

def run(model,owner,get_counts,cases,emit,out,restores,suite='disable_one'):
    from board import operating_state
    request=cases['diagnostic-home-pair'];source=model.prepare(**cases['coverage-0000']);pair=model.prepare(**request)
    assert len(pair.runnable)==2 and max(len(pair.sequences[k]) for k in pair.runnable)==1193
    assert all(pair.sequences[k]==source.sequences[k] for k in pair.runnable)
    records=[];observer=None
    def save(completed=False):
        (out/'ablation.json').write_text(json.dumps({'completed':completed,'suite':suite,'scope':'Two-question diagnostic subset, not a complete benchmark group or performance suite','records':records})+'\n')
    def call(mode):
        if observer is not None:observer.mode=mode
        condition=operating_state();before=copy.deepcopy(get_counts())
        emit({'event':'request_begin','mode':mode});start=time.perf_counter()
        response=model.system_one(**request);seconds=time.perf_counter()-start
        records.append({'mode':mode,'seconds':seconds,'response':response,'counts_before':before,
                        'counts_after':get_counts(),'operating_state':condition,
                        'encoder_restored':model.encoder is owner.last_readout.encoder})
        save();emit({'event':'case_complete','mode':mode,'seconds':seconds})
    modes=(('candidate',()),('cpu_scan',('scan',)),('cpu_attention',('attention',)),
                       ('cpu_mlp',('mlp',)),('cpu_projection',('projection',)),
                       ('all_four_cpu',('scan','attention','mlp','projection')))
    if suite in ('single_enable','single_observe'):
        if suite=='single_enable':
            from fp16_range_probe import run as probe_range
            emit({'event':'request_begin','mode':'synthetic_fp16_range'});probe_range(out)
        else:
            from component_observer import Observer
            observer=Observer(owner,out);restores.append(observer.close)
        modes=tuple(('only_'+kind,tuple(k for k in ('scan','attention','mlp','projection') if k!=kind))
                    for kind in ('scan','attention','mlp','projection'))
    elif suite!='disable_one':raise ValueError('Unknown ablation suite')
    for mode,kinds in modes:
        with perturb(owner,kinds):call(mode)
        errors=list(owner.long_mlp.counts['runtime_errors'])+list(owner.projections.counts['errors'])
        errors += [error for item in owner.long.counts.values() for error in item['errors']]
        if errors:raise RuntimeError('Ablation stopped on runtime fallback; evidence preserved')
    if observer is not None:assert observer.done=={'scan','attention','mlp','projection'}
    # Retire every development adapter before the independent canonical model
    # call. Same original D3.logits batch, B2/T1193; no row-wise oracle here.
    for restore in reversed(restores):restore()
    restores.clear();owner.use('current')
    original_eligible=owner.engine.eligible;owner.engine.eligible=lambda module,x:False
    restores.append(lambda:setattr(owner.engine,'eligible',original_eligible))
    import torch
    for module in model.backbone.modules():
        if isinstance(module,torch.nn.Linear) and hasattr(module,'_bp_original'):
            old=module.forward;module.forward=module._bp_original
            restores.append(lambda m=module,f=old:setattr(m,'forward',f))
    before=(owner.engine.calls,owner.native_mlp.calls,owner.npu_calls)
    call('original_current_cpu_full_pair')
    after=(owner.engine.calls,owner.native_mlp.calls,owner.npu_calls)
    assert before==after,'Canonical CPU pair unexpectedly called NPU'
    records[-1]['npu_calls_delta']=[a-b for a,b in zip(after,before)]
    save(True);emit({'event':'complete','diagnostic_modes':len(records)})
