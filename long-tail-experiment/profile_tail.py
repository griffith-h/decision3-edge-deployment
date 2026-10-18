from runner import *
from pathlib import Path
from types import MethodType
import long_kernels as lk
import prepared_scan
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

app=Owner();case=json.loads(Path('/long/logs/formal.json').read_text())['cases']['words-512']['request']
result={'settings':state(),'request':case,'records':[],'capture':{},'scope':'Separate instrumented profile; capture/cloning/disk IO excluded from profile records'}
restores=[];totals={};stack=[]
def wrap(obj,name,tag):
    old=getattr(obj,name)
    def wrapped(*args,**kwargs):
        frame={'child':0.};stack.append(frame);t=time.perf_counter()
        try:return old(*args,**kwargs)
        finally:
            dt=time.perf_counter()-t;stack.pop();r=totals.setdefault(tag,{'inclusive':0.,'exclusive':0.,'calls':0});r['inclusive']+=dt;r['exclusive']+=dt-frame['child'];r['calls']+=1
            if stack:stack[-1]['child']+=dt
    setattr(obj,name,wrapped);restores.append(lambda:setattr(obj,name,old))

try:
    app.use('baseline');app.invoke(case);app.invoke(case)
    att_original=qwen.ALL_ATTENTION_FUNCTIONS['sdpa']
    class Holder:pass
    holder=Holder();holder.att=att_original
    wrap(holder,'att','attention_entry');qwen.ALL_ATTENTION_FUNCTIONS.register('sdpa',holder.att)
    restores.append(lambda:qwen.ALL_ATTENTION_FUNCTIONS.register('sdpa',att_original))
    wrap(qwen,'torch_chunk_gated_delta_rule','scan_entry')
    wrap(qwen,'causal_conv1d_fn','conv')
    wrap(app.projections,'run','projection_rows')
    wrap(lk,'attention','attention_native_wrapper')
    wrap(lk.att,'d3_attention','attention_c')
    wrap(prepared_scan,'prepare','scan_prepare')
    wrap(lk,'scan_prepared','scan_native_wrapper')
    wrap(lk.scan,'d3_chunk_scan','scan_c')
    selected=set(id(m) for m in app.projections.modules.values())
    for name,m in app.model.backbone.named_modules():
        typ=type(m).__name__
        if typ=='Qwen3_5MLP':wrap(m,'forward','mlp')
        elif 'Norm' in typ:wrap(m,'forward','norm_modules')
        elif isinstance(m,torch.nn.Linear) and id(m) not in selected and '.mlp.' not in name:wrap(m,'forward','small_linear')
    for g in app.bank.graphs+app.bank.extra:wrap(g,'run_bf16','mlp_graph')
    handles=len(app.engine.handles)
    for i in range(5):
        app.use('baseline');totals={};t=time.perf_counter();out=app.invoke(case);elapsed=time.perf_counter()-t
        record={'seconds':elapsed,'components':copy.deepcopy(totals),**app.counts()}
        exclusive=sum(v['exclusive'] for v in totals.values());record['unwrapped']=elapsed-exclusive
        assert record['unwrapped']>=0 and len(app.engine.handles)==handles
        result['records'].append(record)
    for r in reversed(restores):r()
    restores=[]
    # A separate real request captures component inputs and reference outputs.
    captured={'attention':[],'scan':[],'mlp':[]}
    original_att,original_scan=lk.attention,lk.scan_prepared
    def capture_att(*args):
        out=original_att(*args);captured['attention'].append({'args':tuple(x.clone() if isinstance(x,torch.Tensor) else x for x in args),'reference':out.clone()});return out
    def capture_scan(*args):
        out=original_scan(*args);captured['scan'].append({'args':tuple(x.clone() for x in args),'reference':out.clone()});return out
    lk.attention=capture_att;lk.scan_prepared=capture_scan
    old_run=app.bank.run_rows
    def capture_mlp(g,x):
        out=old_run(g,x);captured['mlp'].append({'layer':app.bank.graphs.index(g),'input':x.clone(),'reference':out.clone()});return out
    app.bank.run_rows=capture_mlp
    try:app.use('baseline');app.invoke(case)
    finally:lk.attention=original_att;lk.scan_prepared=original_scan;app.bank.run_rows=old_run
    assert len(captured['attention'])==6 and len(captured['scan'])==6 and len(captured['mlp'])==12
    torch.save(captured,P/'captured.pt')
    result['capture']={k:[list(v['args'][0].shape) if 'args' in v else list(v['input'].shape) for v in values] for k,values in captured.items()}
    result['handles']=handles;result['completed']=True;result['final_settings']=state()
    (P/'logs/profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PROFILE PASS',{k:sum(r['components'].get(k,{}).get('inclusive',0) for r in result['records'])/5 for k in totals},flush=True)
finally:
    for r in reversed(restores):r()
    app.close()
