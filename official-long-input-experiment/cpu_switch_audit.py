"""Audit normal first-batch calls before/after candidate->CPU switch.

No encoder or decoder method is wrapped, and no hidden state is replayed.
Only normal D3 probabilities are recorded/intercepted to stop the first batch.
"""
import hashlib,inspect,json
import torch

class FirstBatchComplete(Exception):pass

def run(model,owner,request,emit,out,restores):
    prepared=model.prepare(**request);keys=prepared.runnable[:8]
    subset=dict(request,questions={k:request['questions'][k] for k in keys});part=model.prepare(**subset)
    assert model.batch_size==8 and model.max_length is None and part.runnable==keys
    assert all(part.sequences[k]==prepared.sequences[k] for k in keys)
    records={};phase='candidate';original_encoder=model.encoder;original_probabilities=model.probabilities
    if model.fast is not None:raise ValueError('Audit requires the pinned ordinary CPU probabilities path')
    def metadata():
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        def describe(fn,depth=0):
            value={'module':getattr(fn,'__module__',None),'qualname':getattr(fn,'__qualname__',type(fn).__name__)}
            try:value['source_sha256']=hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
            except (OSError,TypeError):pass
            instance=getattr(fn,'__self__',None)
            if instance is not None and depth<3:
                for name in ('original_attention','original_scan','previous'):
                    prior=getattr(instance,name,None)
                    if callable(prior):value[name]=describe(prior,depth+1)
            return value
        return {'temperature':model.temperature,'readout_dtype':model.readout_dtype,'permutation_average':model.permutation_average,
                'threads':torch.get_num_threads(),'fast_type':type(model.fast).__name__ if model.fast is not None else None,
                'cpu_silu':describe(torch.nn.functional.silu),'cpu_linear':describe(torch.nn.functional.linear),'mkldnn':torch.backends.mkldnn.enabled,'float32_precision':torch.get_float32_matmul_precision(),
                'sdpa':describe(qwen.ALL_ATTENTION_FUNCTIONS['sdpa']),'scan':describe(qwen.torch_chunk_gated_delta_rule),
                'compiled_layer_calls':[getattr(layer,'_compiled_call_impl',None) is not None for layer,_ in original_encoder.stack]}
    def probabilities(sequences,counts):
        if len(sequences)!=8 or max(map(len,sequences))!=1181:raise ValueError('Original first sequence group changed')
        row=records.setdefault(phase,{})
        row.update(metadata=metadata(),sequences_sha256=hashlib.sha256(json.dumps(sequences,separators=(',',':')).encode()).hexdigest(),counts=list(counts))
        probs=original_probabilities(sequences,counts)
        row['response']=model.respond(part,{key:value for key,value in zip(keys,probs)},sum(map(len,sequences)))
        if phase=='candidate':raise FirstBatchComplete
        return probs
    def save(completed=False):
        (out/'cpu-switch-audit.json').write_text(json.dumps({'completed':completed,'full_request_completed':False,
            'scope':'Normal first original B8 only; encoder and decoder methods untouched; no hidden-state replay; numerical audit only','records':records})+'\n')
    model.probabilities=probabilities
    try:
        emit({'event':'request_begin','phase':'candidate_normal_first_batch','budget_seconds':120})
        try:model.system_one(**request)
        except FirstBatchComplete:pass
        else:raise AssertionError('Candidate first-batch interceptor did not stop')
    finally:model.probabilities=original_probabilities
    save()
    for restore in reversed(restores):restore()
    restores.clear();owner.use('current')
    old_eligible=owner.engine.eligible;owner.engine.eligible=lambda module,x:False
    restores.append(lambda:setattr(owner.engine,'eligible',old_eligible))
    for module in model.backbone.modules():
        if isinstance(module,torch.nn.Linear) and hasattr(module,'_bp_original'):
            old=module.forward;module.forward=module._bp_original;restores.append(lambda m=module,f=old:setattr(m,'forward',f))
    # Free ONLY unused engine-owned SDK matrices through the same mixed ABI.
    # Keep all CPU wrappers and native MLP/attention objects/pools open. No CPU
    # weight or parameter buffer is changed; original CPU forwards remain live.
    owned=set(owner.engine.handles)
    for module in owner.registry():
        if getattr(module,'_bp_handle',None) in owned:module._bp_handle=None
    owner.float_handles.clear();owner.engine.handles.clear()
    for handle in owned:owner.mixed.mm_destroy(handle)
    records['retired_unused_normal_npu_handles']=len(owned)
    phase='cpu_current';before=(owner.engine.calls,owner.native_mlp.calls,owner.npu_calls,owner.bank.counts.get('graph_calls',0))
    # Match the original reference's explicit full/first-batch token checks.
    current=model.prepare(**request);cpu_part=model.prepare(**subset)
    assert cpu_part.runnable==keys and all(cpu_part.sequences[k]==current.sequences[k] for k in keys)
    model.probabilities=probabilities
    try:
        emit({'event':'request_begin','phase':'cpu_current_normal_first_batch_correctness_only'})
        response=model.system_one(**subset)
        after=(owner.engine.calls,owner.native_mlp.calls,owner.npu_calls,owner.bank.counts.get('graph_calls',0))
        if before!=after:raise RuntimeError('Normal CPU audit unexpectedly used NPU/graph')
        records['cpu_current']['actual_system_one_response']=response
        records['cpu_current']['npu_graph_delta']=[a-b for a,b in zip(after,before)]
        save(True);emit({'event':'complete','full_request_completed':False,'actual_cpu_model_call_completed':True})
    finally:
        model.probabilities=original_probabilities
