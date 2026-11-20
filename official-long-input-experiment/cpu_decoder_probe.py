"""First original B8 decoder versus actual CPU, with bounded replay segments.

Candidate computes the entire first batch's twelve decoder layers, then stops
before the final head/remaining batches. CPU replays those same layers in order,
passing its own preceding hidden output and the captured original masks/position
arguments. Three-layer segments are diagnostic budgets, not request timings.
No cached hidden values are ever used by production inference.
"""
import json,time
import torch

class DecoderComplete(Exception):pass

def run(model,owner,request,emit,out,restores,depth=12):
    if not 1<=depth<=12:raise ValueError('Invalid decoder diagnostic depth')
    records=[];captured=[];candidate={};first_input=None
    layers=[layer for layer,_ in model.encoder.stack]
    if len(layers)!=12:raise ValueError('Frozen decoder layer count changed')
    def save(completed=False,reason='running'):
        (out/'cpu-decoder-comparison.json').write_text(json.dumps({'completed':completed,'stop_reason':reason,
            'full_request_completed':False,'scope':f'Original first B8/T1181 first{depth} decoder layers; CPU replay retains its own preceding hidden state and original masks/positions; partial model diagnostic only',
            'records':records})+'\n')
    def compare(a,b):
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Decoder tensor contract changed')
        aa=a.reshape(-1,a.shape[-1]);bb=b.reshape_as(aa);count=0;maximum=0.;examples=[]
        for start in range(0,len(aa),512):
            x,y=aa[start:start+512],bb[start:start+512]
            count+=int((x!=y).sum());maximum=max(maximum,float((x.float()-y.float()).abs().max()))
            if len(examples)<8:
                for row,col in (x!=y).nonzero()[:8-len(examples)].tolist():
                    examples.append({'flat_row':start+row,'column':col,'candidate':float(x[row,col]),'cpu':float(y[row,col]),'consumed_last_position':(start+row)%a.shape[1]==a.shape[1]-1})
        return {'shape':list(a.shape),'bf16_mismatches':count,'maximum_delta':maximum,'examples':examples}
    changed=[]
    try:
        for index,layer in enumerate(layers[:depth]):
            old=layer.forward
            def capture(*args,_old=old,_index=index,**kwargs):
                nonlocal first_input
                h=args[0] if args else kwargs['hidden_states']
                if _index==0:
                    if tuple(h.shape)!=(8,1181,1024):raise ValueError('Original frozen decoder geometry changed')
                    first_input=h.detach().clone()
                # Hidden states are not retained in the argument snapshot: the
                # CPU replay must use its own preceding output.
                tail=args[1:] if args else ()
                kw={k:v for k,v in kwargs.items() if k!='hidden_states'}
                captured.append((bool(args),tail,kw))
                value=_old(*args,**kwargs);candidate[_index]=value.detach().clone()
                emit({'event':'decoder_layer_complete','phase':'candidate','layer':_index})
                if _index==depth-1:raise DecoderComplete
                return value
            layer.forward=capture;changed.append((layer,old))
        emit({'event':'request_begin','phase':'candidate_first_batch_decoder'+str(depth)});start=time.perf_counter()
        try:model.system_one(**request)
        except DecoderComplete:pass
        else:raise AssertionError('Decoder observer failed to stop')
        emit({'event':'case_complete','phase':'candidate_first_batch_decoder'+str(depth),'seconds':time.perf_counter()-start,'full_request_completed':False})
    finally:
        for layer,old in reversed(changed):layer.forward=old
    if len(captured)!=depth:raise AssertionError('Incomplete candidate decoder capture')
    for restore in reversed(restores):restore()
    restores.clear();owner.use('current')
    old_eligible=owner.engine.eligible;owner.engine.eligible=lambda module,x:False
    restores.append(lambda:setattr(owner.engine,'eligible',old_eligible))
    for module in model.backbone.modules():
        if isinstance(module,torch.nn.Linear) and hasattr(module,'_bp_original'):
            old=module.forward;module.forward=module._bp_original
            restores.append(lambda m=module,f=old:setattr(m,'forward',f))
    owner.native_mlp.close();owner.native_attention.close()
    def counters():return (owner.engine.calls,owner.native_mlp.calls,owner.npu_calls,owner.bank.counts.get('graph_calls',0))
    before=counters();hidden=first_input;found_difference=False
    with torch.inference_mode():
        for index,layer in enumerate(layers[:depth]):
            if index%3==0:emit({'event':'request_begin','phase':'cpu_decoder_segment','start_layer':index})
            positional,tail,kw=captured[index];start=time.perf_counter();layer_input=hidden
            hidden=layer(hidden,*tail,**kw) if positional else layer(hidden_states=hidden,**kw)
            actual=candidate.pop(index)
            records.append({'kind':'layer','layer':index,'cpu_seconds_correctness_only':time.perf_counter()-start,**compare(actual,hidden)})
            if counters()!=before:raise RuntimeError('CPU decoder replay unexpectedly used an NPU/graph path')
            save();emit({'event':'decoder_layer_complete','phase':'cpu_current','layer':index,'bf16_mismatches':records[-1]['bf16_mismatches']})
            if records[-1]['bf16_mismatches']:
                torch.save({'layer_index':index,'input':layer_input,'candidate_output':actual,'cpu_output':hidden,
                            'positional':positional,'tail':tail,'kwargs':kw},out/'first-decoder-difference.pt')
                found_difference=True
                if index!=11 or depth!=12:
                    save(True,'first_layer_difference');emit({'event':'complete','layers':index+1,'stop_reason':'first_layer_difference','full_request_completed':False});return
    if depth==12:
        if model.readout_dtype!='float32' or model.permutation_average or model.batch_size!=8:
            raise ValueError('Unsupported pinned readout diagnostic configuration')
        # Preserve a diagnostic CPU first-batch response after all12 real CPU
        # layers; no producer output cache or production inference shortcut.
        prepared=model.prepare(**request);keys=prepared.runnable[:8]
        subset=dict(request,questions={k:request['questions'][k] for k in keys})
        part=model.prepare(**subset)
        if part.runnable!=keys or any(part.sequences[k]!=prepared.sequences[k] for k in keys):raise ValueError('Readout diagnostic changed original first batch')
        counts=[len(prepared.questions[k].keys) for k in keys]
        def readout(value):
            with torch.inference_mode():
                pooled=model.encoder.language_model.norm(value)[:,-1]
                logits=pooled.float()@model.readout.T
                invalid=torch.arange(255,device=model.device)[None]>=torch.as_tensor(counts,device=model.device)[:,None]
                rows=torch.softmax(logits.masked_fill(invalid,float('-inf'))/model.temperature,dim=-1).cpu().tolist()
            return model.respond(part,{key:row[:count] for key,row,count in zip(keys,rows,counts)},sum(len(part.sequences[k]) for k in keys))
        response=readout(hidden);candidate_response=readout(actual)
        torch.save({'cpu_decoder_output':hidden,'candidate_decoder_output':actual,'batch_keys':keys,'counts':counts},out/'cpu-final-decoder-hidden.pt')
        (out/'cpu-replayed-first-batch.json').write_text(json.dumps({'completed':True,'full_request_completed':False,
            'scope':'Original first B8 only; real CPU decoder replay and captured candidate decoder output use original norm/FP32 readout/softmax/respond in the same CPU context',
            'response':response,'candidate_decoder_response':candidate_response})+'\n')
        if counters()!=before:raise RuntimeError('CPU readout diagnostic unexpectedly used NPU/graph')
    records.append({'kind':'cpu_counter_delta','npu_graph_calls':[a-b for a,b in zip(counters(),before)]})
    reason='first_layer_difference' if found_difference else 'all'+str(depth)+'_decoder_layers_equal'
    save(True,reason);emit({'event':'complete','layers':depth,'stop_reason':reason,'full_request_completed':False})
