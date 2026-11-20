"""Compare first3candidate layers to actual current CPU at original B8 geometry.

Two intentionally partial model calls. Retire candidate adapters before the CPU
call; assert no NPU/graph counter changes. Never score this as a full request.
"""
import json,time
import torch

class PrefixComplete(Exception):pass

def run(model,owner,request,emit,out,restores,depth=3):
    if not 1<=depth<=6:raise ValueError('Invalid bounded diagnostic depth')
    candidate={};raw_input=None;records=[]
    def save(completed=False):
        (out/'cpu-prefix-comparison.json').write_text(json.dumps({'completed':completed,'full_request_completed':False,
            'scope':f'Original8x1181 first{depth}layers; candidate versus actual owner.current CPU, not just fusion ablation',
            'records':records})+'\n')
    def compare(a,b):
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Prefix tensor contract changed')
        aa=a.reshape(-1,a.shape[-1]);bb=b.reshape_as(aa);count=0;maximum=0.
        for start in range(0,len(aa),512):
            x,y=aa[start:start+512],bb[start:start+512];count+=int((x!=y).sum());maximum=max(maximum,float((x.float()-y.float()).abs().max()))
        return {'shape':list(a.shape),'bf16_mismatches':count,'maximum_delta':maximum}
    def call(mode):
        nonlocal raw_input
        changed=[]
        try:
            for index,(layer,_) in enumerate(model.encoder.stack[:depth]):
                old=layer.forward
                def capture(*args,_old=old,_index=index,**kwargs):
                    nonlocal raw_input
                    h=args[0] if args else kwargs['hidden_states']
                    if _index==0:
                        if tuple(h.shape)!=(8,1181,1024):raise ValueError('Original frozen prefix geometry changed')
                        if mode=='candidate':raw_input=h.detach().clone()
                        else:records.append({'kind':'initial_hidden',**compare(raw_input,h)});save()
                    value=_old(*args,**kwargs)
                    if mode=='candidate':candidate[_index]=value.detach().clone()
                    else:records.append({'kind':'layer','layer':_index,**compare(candidate[_index],value)});save()
                    if _index==depth-1:raise PrefixComplete
                    return value
                layer.forward=capture;changed.append((layer,old))
            emit({'event':'request_begin','phase':mode+'_prefix'+str(depth)});start=time.perf_counter()
            try:model.system_one(**request)
            except PrefixComplete:pass
            else:raise AssertionError('Prefix observer failed to stop')
            emit({'event':'case_complete','phase':mode+'_prefix'+str(depth),'seconds':time.perf_counter()-start,'full_request_completed':False})
        finally:
            for layer,old in reversed(changed):layer.forward=old
    call('candidate')
    for restore in reversed(restores):restore()
    restores.clear();owner.use('current')
    old_eligible=owner.engine.eligible;owner.engine.eligible=lambda module,x:False
    restores.append(lambda:setattr(owner.engine,'eligible',old_eligible))
    for module in model.backbone.modules():
        if isinstance(module,torch.nn.Linear) and hasattr(module,'_bp_original'):
            old=module.forward;module.forward=module._bp_original;restores.append(lambda m=module,f=old:setattr(m,'forward',f))
    # Contexts are no longer used by current CPU; free this candidate's own
    # banks before allocating the canonical CPU's larger full-batch workspaces.
    owner.native_mlp.close();owner.native_attention.close()
    before=(owner.engine.calls,owner.native_mlp.calls,owner.npu_calls,owner.bank.counts.get('graph_calls',0))
    call('cpu_current')
    after=(owner.engine.calls,owner.native_mlp.calls,owner.npu_calls,owner.bank.counts.get('graph_calls',0))
    if before!=after:raise RuntimeError('Canonical CPU prefix used an NPU/graph path')
    records.append({'kind':'cpu_counter_delta','npu_graph_calls':[a-b for a,b in zip(after,before)]});save(True)
    emit({'event':'complete','layers':depth,'full_request_completed':False})
