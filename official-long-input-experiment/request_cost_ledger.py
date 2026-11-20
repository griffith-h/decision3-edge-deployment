"""Diagnostic intervals only; main-thread leaves partition request wall time.

Worker and SDK intervals are separate views, never added to the main ledger.
All hooks return the exact original result and restore on scope exit.
"""
import contextlib,itertools,threading,time


def union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if end<start:raise ValueError('Reversed interval')
        if result and start<=result[-1][1]:result[-1][1]=max(end,result[-1][1])
        else:result.append([start,end])
    return result


def overlap(start,end,intervals):
    return sum(max(0,min(end,b)-max(start,a)) for a,b in intervals)


def partition(events,root,sdk):
    """Partition the owner thread; use union, never sum, for concurrent SDK calls."""
    sdk_union=union((r['start_ns'],r['end_ns']) for r in sdk)
    main=[e for e in events if e['thread']==root['thread']]
    children={}
    for e in main:
        if e['parent'] is not None:children.setdefault(e['parent'],[]).append(e)
    rows={};segments=[]
    for e in main:
        if not root['start_ns']<=e['start_ns']<=e['end_ns']<=root['end_ns']:raise ValueError('Interval outside request')
        direct=sorted(children.get(e['id'],[]),key=lambda r:r['start_ns']);cursor=e['start_ns']
        for child in direct+[{'start_ns':e['end_ns'],'end_ns':e['end_ns']}]:
            if child['start_ns']<cursor:raise ValueError('Overlapping siblings')
            start,end=cursor,child['start_ns'];cursor=child['end_ns']
            if end==start:continue
            category=e['category'];sdk_ns=0
            if category in ('projection_native_call','mlp_native_call','worker_join'):
                sdk_ns=overlap(start,end,sdk_union)
                if sdk_ns:rows['sdk_runtime_sync']=rows.get('sdk_runtime_sync',0)+sdk_ns
            rows[category]=rows.get(category,0)+(end-start-sdk_ns)
            segments.append({'start_ns':start,'end_ns':end,'category':category,'tag':e['tag'],'layer':e['layer'],'sdk_overlap_ns':sdk_ns})
    duration=root['end_ns']-root['start_ns']
    if sum(rows.values())!=duration:raise ValueError('Wall ledger does not close')
    return {'seconds':duration/1e9,'sum_seconds':sum(rows.values())/1e9,'closure_error_ns':0,
            'exclusive_seconds':{k:v/1e9 for k,v in sorted(rows.items(),key=lambda x:-x[1])},
            'segments':segments,'sdk_union_seconds':sum(b-a for a,b in sdk_union)/1e9}


class Recorder:
    def __init__(self):
        self.events=[];self.local=threading.local();self.ids=itertools.count();self.restores=[];self.active=False
    @contextlib.contextmanager
    def span(self,tag,category,layer=None):
        stack=getattr(self.local,'stack',[])
        if not hasattr(self.local,'stack'):self.local.stack=stack
        row={'id':next(self.ids),'parent':stack[-1]['id'] if stack else None,
             'thread':threading.get_native_id(),'tag':tag,'category':category,
             'layer':layer if layer is not None else (stack[-1]['layer'] if stack else getattr(self.local,'layer',None)),
             'start_ns':time.perf_counter_ns(),'cpu_start_ns':time.thread_time_ns()}
        stack.append(row)
        try:yield row
        except BaseException as exc:row['error']=type(exc).__name__;raise
        finally:
            row['end_ns']=time.perf_counter_ns();row['cpu_end_ns']=time.thread_time_ns();stack.pop();self.events.append(row)
    def replace(self,obj,name,new):
        # Restore the original instance attribute or class descriptor exactly.
        existed=name in vars(obj);previous=vars(obj).get(name);old=getattr(obj,name)
        setattr(obj,name,new)
        def restore():
            if getattr(obj,name) is not new:raise RuntimeError('Diagnostic hook ownership changed: '+name)
            if existed:setattr(obj,name,previous)
            else:delattr(obj,name)
        self.restores.append(restore);return old
    def wrap(self,obj,name,category,tag=None,layer=None):
        old=getattr(obj,name)
        def call(*args,**kwargs):
            if not self.active:return old(*args,**kwargs)
            with self.span(tag or name,category,layer):return old(*args,**kwargs)
        self.replace(obj,name,call)
    def close(self):
        self.active=False
        for restore in reversed(self.restores):restore()
        self.restores.clear()


def install(model,owner):
    """No Linear/MLP forward guards or model algorithms are replaced."""
    import concurrent.futures,sys,torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    rec=Recorder()
    try:
        for name,category in [('prepare','request_prepare'),('respond','request_respond'),('logits','model_other')]:rec.wrap(model,name,category)
        guarded=set(getattr(owner.last_readout,'guarded_modules',[]))
        for index,(layer,_) in enumerate(model.encoder.stack):
            if layer not in guarded:rec.wrap(layer,'forward','decoder_other','layer.'+str(index),index)
        for name,module in model.backbone.named_modules():
            if module in guarded:continue
            typ=type(module).__name__
            if 'GatedDeltaNet' in typ:rec.wrap(module,'forward','scan_other',name)
            elif typ=='Qwen3_5Attention':rec.wrap(module,'forward','attention_other',name)
            elif 'Norm' in typ:rec.wrap(module,'forward','normalization',name)
        rec.wrap(owner.long,'original_scan','cpu_scan')
        # Refine all six actual scans, rather than extrapolating the first scan.
        # Outside that scope these operators are exact pass-throughs.
        def scan_operator(obj,name,category):
            previous=getattr(obj,name)
            def call(*args,**kwargs):
                if not rec.active or not any(e['category']=='cpu_scan' for e in getattr(rec.local,'stack',[])):
                    return previous(*args,**kwargs)
                with rec.span('scan.'+name,category) as row:
                    row['shapes']=[list(v.shape) for v in args if isinstance(v,torch.Tensor)]
                    return previous(*args,**kwargs)
            rec.replace(obj,name,call)
        scan_operator(torch.Tensor,'__matmul__','cpu_scan_matmul')
        scan_operator(torch.linalg,'solve_triangular','cpu_scan_triangular_solve')
        for name in ('contiguous','float','to'):
            scan_operator(torch.Tensor,name,'cpu_scan_copy_convert')
        if hasattr(qwen,'causal_conv1d_fn'):rec.wrap(qwen,'causal_conv1d_fn','cpu_convolution')
        registry=qwen.ALL_ATTENTION_FUNCTIONS;old_sdpa=registry['sdpa']
        def sdpa(*args,**kwargs):
            if not rec.active:return old_sdpa(*args,**kwargs)
            with rec.span('registered_sdpa','cpu_attention'):return old_sdpa(*args,**kwargs)
        registry.register('sdpa',sdpa)
        rec.restores.append(lambda:registry.register('sdpa',old_sdpa))
        rec.wrap(owner.projections,'run','projection_other')
        for name in ('group','execute','ensure'):rec.wrap(owner.engine,name,'projection_other','projection.'+name)
        rec.wrap(owner.long_mlp,'forward','mlp_other','mlp_tiling')
        for name in ('run','group','execute','ensure','ensure_down'): 
            if hasattr(owner.native_mlp,name):rec.wrap(owner.native_mlp,name,'mlp_other','mlp.'+name)
        state=owner.medium_sparse;repair=state.repair
        rec.wrap(state,'correct_norm','fp32_read_other');rec.wrap(state,'correct_native','fp32_read_other')
        rec.wrap(repair,'repair','repair_other');rec.wrap(repair.preparation,'norms','repair_norms')
        rec.wrap(repair.preparation,'mask','repair_mask');rec.wrap(repair.blas_repair,'run','repair_dots')
        for lib,name,category in [(state.norm,'d3_projection_read_fp32','fp32_read'),(state.raw_native,'d3_probe_copy_fp32','fp32_read'),
                                  (state.raw_native,'d3_native_product_bf16','mlp_product_prepare')]:rec.wrap(lib,name,category)
        for lib,names,prefix in [(owner.engine.lib,('mm_prepare_bf16','mm_copy_prepared','mm_run_bf16','mm_run_prepared_bf16','mm_run_prepared_scatter_bf16','mm_run_raw','mm_prepare_mlp'),'projection'),
                                 (owner.native_mlp.lib,('nm_prepare','nm_copy','nm_raw','nm_output','nm_fuse'),'mlp')]:
            for name in names:
                if hasattr(lib,name):rec.wrap(lib,name,prefix+'_native_call',prefix+'.'+name)
        # Preserve Future objects, queue order and every original join.
        pool=owner.engine.pool;submit=pool.submit
        def traced_submit(fn,*args,**kwargs):
            if not rec.active:return submit(fn,*args,**kwargs)
            stack=getattr(rec.local,'stack',[]);layer=stack[-1]['layer'] if stack else None
            def task():
                previous=getattr(rec.local,'layer',None);rec.local.layer=layer
                try:
                    with rec.span('worker_task','worker_task'):return fn(*args,**kwargs)
                finally:rec.local.layer=previous
            return submit(task)
        rec.replace(pool,'submit',traced_submit)
        rec.wrap(concurrent.futures.Future,'result','worker_join','Future.result')
        for module_name in ('native_mlp','fused_sparse_mlp_candidate'):
            module=sys.modules.get(module_name)
            if module and hasattr(module,'wait'):rec.wrap(module,'wait','worker_join',module_name+'.wait')
        return rec
    except BaseException:rec.close();raise
