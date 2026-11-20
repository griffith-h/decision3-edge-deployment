"""Bounded official-input profiling in the main inference thread; private results only."""
import argparse,copy,hashlib,json,os,resource,sys,threading,time
from pathlib import Path
sys.path.insert(0,'/evaluation')
from board import setup,identity,operating_state

FRAME='@@D3OPT@@'
def memory_snapshot():
    result={}
    for name in ('memory.current','memory.peak','memory.max','memory.events'):
        try:result[name]=Path('/sys/fs/cgroup',name).read_text().strip()
        except OSError:pass
    try:
        result['process_status']={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmRSS:','VmHWM:','RssAnon:','RssFile:','RssShmem:'))}
    except OSError:pass
    return result
def emit(value):print(FRAME+json.dumps(value),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--repeat',type=int,default=1);p.add_argument('--profile',action='store_true');p.add_argument('--variant',choices=['baseline','batch','mlp','batch-mlp','batch-mlp-projection','batch-mlp-projection-fp32','batch-mlp-projection-fp32-attention','batch-mlp-projection-fp32-scan','batch-mlp-projection-fp32-scan-npu-attention','batch-mlp-projection-fp32-scan-npu-attention-readout','medium-npu-attention','full-batch','full-batch-precise-medium','full-batch-sparse','full-batch-sparse-cblas','full-batch-sparse-native','full-batch-sparse-fused','full-batch-sparse-flash','full-batch-dense-medium','full-batch-sparse-vector','full-batch-sparse-diagonal','full-batch-sparse-compact','full-batch-sparse-band2','full-batch-sparse-sequential','full-batch-sparse-band25','full-batch-sparse-native-scan','full-batch-sparse-visual-scope','full-batch-sparse-request-scope','reference','cpu-full'],default='baseline');a=p.parse_args()
 out=Path('/results');out.mkdir(exist_ok=True)
 cases={c['id']:c['payload'] for c in json.loads(Path('/inputs/inputs.json').read_text())}
 extra=Path('/inputs/coverage-inputs.json')
 if extra.exists():
  for row in json.loads(extra.read_text()):
   if row['id'] in cases:raise ValueError('Coverage case ID collision')
   cases[row['id']]=row['payload']
 request=cases.get(a.case)
 backend_states=[]
 def read_backend_state(stage):
  import torch
  value={'stage':stage,'mkldnn':torch.backends.mkldnn.enabled,'threads':torch.get_num_threads(),'float32_precision':torch.get_float32_matmul_precision(),'autocast_cpu':torch.is_autocast_enabled('cpu'),'grad_enabled':torch.is_grad_enabled(),'inference_mode':torch.is_inference_mode_enabled()}
  return value
 def backend_state(stage):
  value=read_backend_state(stage)
  backend_states.append(value);(out/'backend-state-audit.json').write_text(json.dumps({'completed':False,'states':backend_states})+'\n')
 model=close=get_counts=owner=None;restores=[];trace=(out/'trace.jsonl').open('w');totals={};stack=[];main_thread=threading.get_ident()
 def log(event,**fields):
  trace.write(json.dumps({'event':event,'at':time.perf_counter(),**fields})+'\n');trace.flush()
 def wrap(obj,name,tag,kind):
  old=getattr(obj,name)
  def call(*args,**kwargs):
   if threading.get_ident()!=main_thread:return old(*args,**kwargs)
   shapes=[list(x.shape) for x in args if hasattr(x,'shape')]
   if kind=='batch' and args:
    lengths=[len(s) for s in args[0]]
    log('batch',lengths=lengths,padded_tokens=len(lengths)*max(lengths),effective_tokens=sum(lengths))
   start=time.perf_counter();frame={'child':0};stack.append(frame);log('begin',tag=tag,kind=kind,shapes=shapes)
   try:return old(*args,**kwargs)
   finally:
    elapsed=time.perf_counter()-start;stack.pop();row=totals.setdefault(tag,{'kind':kind,'calls':0,'inclusive':0.,'exclusive':0.})
    row['calls']+=1;row['inclusive']+=elapsed;row['exclusive']+=elapsed-frame['child']
    if stack:stack[-1]['child']+=elapsed
    log('end',tag=tag,kind=kind,seconds=elapsed,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  setattr(obj,name,call);restores.append(lambda:setattr(obj,name,old))
 try:
  if a.case=='native-copy-preflight':
   from native_copy_preflight import run
   run(emit,out);return
  if a.case=='enabled-cpu-reduction-preflight':
   from enabled_reduction_preflight import run
   run(emit,out);return
  if a.case=='projection-single-point':
   from cpu_point_probe import run
   run(emit,out);return
  model,close,get_counts=setup('native-bf16');owner=close.__self__;backend_state('after_setup')
  if a.case=='projection-runtime-point':
   from cpu_point_probe import run
   counters_before=(owner.engine.calls,owner.native_mlp.calls,owner.npu_calls)
   run(emit,out,owner)
   assert counters_before==(owner.engine.calls,owner.native_mlp.calls,owner.npu_calls)
   return
  (out/'identity.json').write_text(json.dumps(identity('native-bf16'))+'\n')
  (out/'lab-sources.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('/lab').iterdir() if f.is_file()})+'\n')
  # Same two protection warmups, never sampled labels.
  for name,tokens in (('protect615',615),('protect115',115)):
   warm=model.system_one(**cases[name]);assert warm['usage']['input_tokens']==tokens;backend_state('after_warm_'+name)
  baseline_logits=model.logits
  if a.variant in ('reference','cpu-full'):
   import torch
   owner.use('current')
   original_eligible=owner.engine.eligible
   owner.engine.eligible=lambda module,x:False
   restores.append(lambda:setattr(owner.engine,'eligible',original_eligible))
   for module in model.backbone.modules():
    if isinstance(module,torch.nn.Linear) and hasattr(module,'_bp_original'):
     old=module.forward;module.forward=module._bp_original
     restores.append(lambda m=module,f=old:setattr(m,'forward',f))
  if a.variant in ('batch','mlp','batch-mlp','batch-mlp-projection','batch-mlp-projection-fp32','batch-mlp-projection-fp32-attention','batch-mlp-projection-fp32-scan','batch-mlp-projection-fp32-scan-npu-attention','batch-mlp-projection-fp32-scan-npu-attention-readout','medium-npu-attention','full-batch','full-batch-precise-medium','full-batch-sparse','full-batch-sparse-cblas','full-batch-sparse-native','full-batch-sparse-fused','full-batch-sparse-flash','full-batch-dense-medium','full-batch-sparse-vector','full-batch-sparse-diagonal','full-batch-sparse-compact','full-batch-sparse-band2','full-batch-sparse-sequential','full-batch-sparse-band25','full-batch-sparse-native-scan','full-batch-sparse-visual-scope','full-batch-sparse-request-scope'):
   import unittest,importlib
   # MLP must be inside the batch wrapper so it sees each actual subbatch.
   names={'batch':['batch_candidate'],'mlp':['long_mlp_candidate'],
          'batch-mlp':['long_mlp_candidate','batch_candidate'],
          'batch-mlp-projection':['long_projection_candidate','long_mlp_candidate','batch_candidate'],
          'batch-mlp-projection-fp32':['long_projection_fp32_candidate','long_mlp_candidate','batch_candidate'],
          'batch-mlp-projection-fp32-attention':['long_projection_fp32_candidate','long_mlp_candidate','batch_candidate','long_attention_candidate'],
          'batch-mlp-projection-fp32-scan':['long_projection_fp32_candidate','long_mlp_candidate','batch_candidate','long_scan_candidate'],
          'batch-mlp-projection-fp32-scan-npu-attention':['long_projection_fp32_candidate','long_mlp_candidate','batch_candidate','long_scan_candidate','long_npu_attention_candidate'],
          'batch-mlp-projection-fp32-scan-npu-attention-readout':['long_projection_fp32_candidate','long_mlp_candidate','long_readout_candidate','batch_candidate','long_scan_candidate','long_npu_attention_candidate'],
          'medium-npu-attention':['long_projection_fp32_candidate','long_mlp_candidate','long_readout_candidate','batch_candidate','long_scan_candidate','medium_npu_attention_candidate'],
          'full-batch':['full_batch_projection_candidate','long_mlp_candidate','long_readout_candidate','long_scan_candidate','long_npu_attention_candidate'],
          'full-batch-precise-medium':['full_batch_projection_candidate','long_mlp_candidate','long_readout_candidate','long_scan_candidate','long_npu_attention_candidate','medium_precision_candidate'],
          'full-batch-sparse':['full_batch_projection_candidate','long_mlp_candidate','long_readout_candidate','long_scan_candidate','long_npu_attention_candidate','medium_precision_candidate','medium_sparse_candidate'],
          'full-batch-sparse-cblas':['full_batch_projection_candidate','long_mlp_candidate','long_readout_candidate','long_scan_candidate','long_npu_attention_candidate','medium_precision_candidate','medium_sparse_candidate','cblas_repair_candidate'],
          'full-batch-sparse-native':['full_batch_projection_candidate','long_mlp_candidate','long_readout_candidate','long_scan_candidate','long_npu_attention_candidate','medium_precision_candidate','medium_sparse_candidate','cblas_repair_candidate','native_preparation_candidate'],
          'full-batch-sparse-fused':['full_batch_projection_candidate','long_mlp_candidate','long_readout_candidate','long_scan_candidate','long_npu_attention_candidate','medium_precision_candidate','medium_sparse_candidate','cblas_repair_candidate','native_preparation_candidate','fused_sparse_mlp_candidate'],
          'full-batch-sparse-flash':['full_batch_projection_candidate','long_mlp_candidate','long_readout_candidate','long_scan_candidate','long_npu_attention_candidate','medium_precision_candidate','medium_sparse_candidate','cblas_repair_candidate','native_preparation_candidate','medium_flash_candidate'],
          'full-batch-dense-medium':['full_batch_projection_candidate','long_mlp_candidate','long_readout_candidate','long_scan_candidate','long_npu_attention_candidate','medium_precision_candidate','medium_dense_candidate'],
          'full-batch-sparse-vector':['full_batch_projection_candidate','long_mlp_candidate','long_readout_candidate','long_scan_candidate','long_npu_attention_candidate','medium_precision_candidate','medium_sparse_candidate','cblas_repair_candidate','native_preparation_candidate','vector_preparation_candidate'],
          'full-batch-sparse-diagonal':['full_batch_projection_candidate','long_mlp_candidate','long_readout_candidate','long_scan_candidate','long_npu_attention_candidate','medium_precision_candidate','medium_sparse_candidate','cblas_repair_candidate','native_preparation_candidate','vector_preparation_candidate','diagonal_repair_candidate'],
          'full-batch-sparse-compact':['full_batch_projection_candidate','long_mlp_candidate','long_readout_candidate','long_scan_candidate','long_npu_attention_candidate','medium_precision_candidate','medium_sparse_candidate','cblas_repair_candidate','native_preparation_candidate','vector_preparation_candidate','diagonal_repair_candidate','fused_sparse_mlp_candidate','packed_cache_policy'],
          'full-batch-sparse-band2':['full_batch_projection_candidate','long_mlp_candidate','long_readout_candidate','long_scan_candidate','long_npu_attention_candidate','medium_precision_candidate','medium_sparse_candidate','cblas_repair_candidate','native_preparation_candidate','vector_preparation_candidate','diagonal_repair_candidate','fused_sparse_mlp_candidate','packed_cache_policy','rounding_band_candidate']}
   names['full-batch-sparse-sequential']=[*names['full-batch-sparse-band2'],'enabled_sequential_repair_candidate']
   names['full-batch-sparse-band25']=[*names['full-batch-sparse-sequential'],'rounding_band25_candidate']
   names['full-batch-sparse-native-scan']=[*names['full-batch-sparse-band25'],'medium_scan_restore_candidate']
   names['full-batch-sparse-visual-scope']=[*names['full-batch-sparse-band25'],'visual_backend_scope_candidate']
   names['full-batch-sparse-request-scope']=[*names['full-batch-sparse-band25'],'request_backend_scope_candidate']
   names=names[a.variant]
   test_modules=[]
   for module_name in names:
    test_names=['test_'+module_name]
    if module_name=='long_projection_fp32_candidate':test_names.insert(0,'test_long_projection_candidate')
    if module_name=='full_batch_projection_candidate':test_names[:0]=['test_long_projection_candidate','test_long_projection_fp32_candidate']
    if module_name=='medium_npu_attention_candidate':test_names.insert(0,'test_long_npu_attention_candidate')
    if module_name=='medium_sparse_candidate':test_names[:0]=['test_projection_read_bridge','test_sparse_cpu_backend','test_sparse_rounding']
    for test_name in test_names:
     if test_name not in test_modules:test_modules.append(test_name)
   # No test module is imported in this live owner process. The first old
   # projection toy can disable MKLDNN globally on this CPU; child exit keeps
   # that fallback and every other test-only global change out of inference.
   from isolated_checks import run_checks
   import torch
   expected_mkldnn=torch.backends.mkldnn.enabled
   backend_state('before_isolated_boundary_checks')
   if a.case=='selected-layout-preflight':test_modules+=['test_selected_layout_tail']
   run_checks(test_modules,out/'isolated-boundary-checks.json')
   backend_state('after_isolated_boundary_checks')
   if torch.backends.mkldnn.enabled!=expected_mkldnn:raise RuntimeError('Isolated checks changed live CPU backend')
   for module_name in names:
    restores.append(importlib.import_module(module_name).install(owner));backend_state('after_install_'+module_name)
    if torch.backends.mkldnn.enabled!=expected_mkldnn:raise RuntimeError('Candidate installation changed original CPU backend')
  backend_state('after_variant_selection')
  if a.case=='backend-state-audit':
   emit({'event':'ready','case':a.case,'variant':a.variant})
   (out/'backend-state-audit.json').write_text(json.dumps({'completed':True,'full_request_completed':False,'states':backend_states})+'\n')
   emit({'event':'complete','backend_snapshots':len(backend_states),'full_request_completed':False});return
  if hasattr(owner,'medium_sparse'):owner.medium_sparse.repair.profile=a.profile
  (out/'memory-before-request.json').write_text(json.dumps(memory_snapshot())+'\n')
  before=copy.deepcopy(get_counts())
  if a.case=='context-regression34':
   if a.variant not in ('reference','baseline','full-batch-sparse-request-scope'):raise ValueError('Unsupported context regression role')
   if a.variant=='reference':
    previous_bank_mode=owner.bank.mode;owner.bank.mode='cpu'
    restores.append(lambda:setattr(owner.bank,'mode',previous_bank_mode))
   if not hasattr(owner,'request_backend_scope_counts'):
    from isolated_checks import run_checks
    run_checks(['test_request_backend_scope_candidate'],out/'isolated-request-scope-checks.json')
    from request_backend_scope_candidate import install
    restores.append(install(owner))
   from context_regression import run
   from cpu_batch_reference import counters
   content=Path('/inputs/regression-inputs.json').read_bytes()
   run(model,owner,json.loads(content),get_counts,read_backend_state,emit,out,a.variant,content,counters);return
  if a.variant=='cpu-full':
   from cpu_batch_reference import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases[a.case],emit,out,a.case);return
  if a.case=='selected-layout-preflight':
   from selected_layout_preflight import run
   run(model,owner,cases['diagnostic-home-pair'],emit,out,memory_snapshot);return
  if a.case=='request-cost-ledger':
   from request_cost_probe import run
   run(model,owner,cases['diagnostic-home-pair'],get_counts,read_backend_state,operating_state,emit,out);return
  if a.case=='scan-bmm-preflight':
   from scan_bmm_probe import run
   run(model,owner,cases['diagnostic-home-pair'],emit,out);return
  if a.case=='scan-operator-profile':
   from scan_operator_probe import run
   run(model,owner,cases['diagnostic-home-pair'],get_counts,emit,out);return
  if a.case=='copy-layout-pairing':
   from native_copy_pairing import run
   run(model,owner,cases['diagnostic-home-pair'],emit,out);return
  if a.case=='cpu-switch-audit':
   from cpu_switch_audit import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases['coverage-0000'],emit,out,restores);return
  if a.case=='projection-repair-point':
   from projection_repair_probe import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases['coverage-0000'],emit,out);return
  if a.case=='last-layer-replay-components':
   from layer_component_probe import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases['coverage-0000'],emit,out,replay_path='/inputs/first-decoder-difference.pt');return
  import re
  component_case=re.fullmatch(r'layer([0-9]|1[01])-components',a.case)
  if component_case:
   from layer_component_probe import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases['coverage-0000'],emit,out,layer_index=int(component_case.group(1)));return
  if a.case in ('cpu-decoder3-reference','cpu-decoder9-reference'):
   from cpu_decoder_probe import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases['coverage-0000'],emit,out,restores,
       depth=3 if a.case=='cpu-decoder3-reference' else 9);return
  if a.case=='cpu-decoder-reference':
   from cpu_decoder_probe import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases['coverage-0000'],emit,out,restores);return
  if a.case=='cpu-prefix5-reference':
   from cpu_prefix_probe import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases['coverage-0000'],emit,out,restores,depth=5);return
  if a.case=='cpu-prefix-reference':
   from cpu_prefix_probe import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases['coverage-0000'],emit,out,restores);return
  if a.case=='first-cpu-layers':
   from cpu_layer_probe import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases['coverage-0000'],emit,out);return
  if a.case=='first-cpu-fusions':
   from cpu_fusion_probe import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases['coverage-0000'],emit,out);return
  if a.case in ('first-full-projection','first-cpu-reduction-geometry'):
   from first_projection_probe import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases['coverage-0000'],emit,out,cpu_geometry=a.case=='first-cpu-reduction-geometry');return
  if a.case=='linear-precision':
   from linear_precision_probe import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases,emit,out);return
  if a.case in ('ablation-pair','ablation-single','ablation-components'):
   from ablation_suite import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   suite={'ablation-pair':'disable_one','ablation-single':'single_enable','ablation-components':'single_observe'}[a.case]
   if a.case=='ablation-components':
    import unittest,test_component_observer
    tests=unittest.defaultTestLoader.loadTestsFromModule(test_component_observer)
    if not unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(tests).wasSuccessful():raise RuntimeError('Observation lifecycle checks failed')
   run(model,owner,get_counts,cases,emit,out,restores,suite=suite);return
  if a.case=='coverage':
   from coverage_scan import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,get_counts,emit,out);return
  if a.case=='attention-sweep':
   from attention_tile_sweep import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(owner,emit,out);return
  if a.case=='validation':
   from validation_suite import run
   emit({'event':'ready','case':a.case,'variant':a.variant})
   run(model,owner,cases,get_counts,baseline_logits,emit,out);return
  if a.case in ('visual-transition','visual-transition-protection','visual-transition-protection-audit'):
   image=next(row['payload'] for row in json.loads(Path('/inputs/regression-inputs.json').read_text()) if row['id']=='image')
   sequence=[('image',image)]+[(name,cases[name]) for name in ('diagnostic-home-pair','protect615','protect115')]
   transitions=[];emit({'event':'ready','case':a.case,'variant':a.variant})
   for name,payload in sequence:
    before_backend=read_backend_state('before_'+name)
    emit({'event':'request_begin','case':name});start=time.perf_counter()
    response=model.system_one(**payload);elapsed=time.perf_counter()-start
    transitions.append({'id':name,'seconds':elapsed,'response':response,
        'backend_before':before_backend,'backend_after':read_backend_state('after_'+name),
        'visual_backend_scope_counts':dict(getattr(owner,'visual_backend_scope_counts',{})),'request_backend_scope_counts':dict(getattr(owner,'request_backend_scope_counts',{}))})
    (out/'visual-transition.json').write_text(json.dumps({'completed':False,'records':transitions})+'\n')
    emit({'event':'case_complete','case':name,'seconds':elapsed})
   (out/'visual-transition.json').write_text(json.dumps({'completed':True,'records':transitions})+'\n')
   if a.case=='visual-transition':emit({'event':'complete','cases':len(transitions)});return
  if a.case=='regression34':
   rows=json.loads(Path('/inputs/regression-inputs.json').read_text());records=[]
   emit({'event':'ready','case':a.case,'variant':a.variant})
   for row in rows:
    before_backend=read_backend_state('before_'+row['id'])
    emit({'event':'request_begin','case':row['id']});start=time.perf_counter()
    response=model.system_one(**row['payload']);seconds=time.perf_counter()-start
    records.append({'id':row['id'],'response':response,'seconds':seconds,
                    'backend_before':before_backend,'backend_after':read_backend_state('after_'+row['id']),
                    'visual_backend_scope_counts':dict(getattr(owner,'visual_backend_scope_counts',{})),'request_backend_scope_counts':dict(getattr(owner,'request_backend_scope_counts',{}))})
    (out/'regression.json').write_text(json.dumps({'completed':False,'records':records})+'\n')
    emit({'event':'case_complete','case':row['id'],'seconds':seconds})
   (out/'regression.json').write_text(json.dumps({'completed':True,'records':records})+'\n')
   emit({'event':'complete','cases':len(records)});return
  if a.case=='visual-transition-protection-audit':
   import torch
   source={r['id']:r['payload'] for r in json.loads(Path('/inputs/regression-inputs.json').read_text())}
   paired=[];candidate_logits=model.logits
   for name in ('matched615-5','matched615-6','matched615-11'):
    for enabled in (True,False):
     with torch.backends.mkldnn.flags(enabled=enabled):
      for candidate in (False,True):
       saved=model.logits;model.logits=candidate_logits if candidate else baseline_logits
       before_backend=read_backend_state('audit_before');before_counts=copy.deepcopy(get_counts())
       emit({'event':'request_begin','case':name,'phase':'protected_backend_audit'});start=time.perf_counter()
       try:response=model.system_one(**source[name]);elapsed=time.perf_counter()-start
       finally:model.logits=saved
       paired.append({'id':name,'candidate':candidate,'backend_enabled':enabled,'response':response,'seconds':elapsed,
        'backend_before':before_backend,'backend_after':read_backend_state('audit_after'),'counts_before':before_counts,'counts_after':copy.deepcopy(get_counts())})
       (out/'protected-backend-audit.json').write_text(json.dumps({'completed':False,'records':paired})+'\n')
   (out/'protected-backend-audit.json').write_text(json.dumps({'completed':True,'records':paired})+'\n')
  if a.case in ('protection','visual-transition-protection','visual-transition-protection-audit'):
   algorithm=getattr(owner,'long_mlp',None) if a.variant in ('mlp','batch-mlp','batch-mlp-projection','batch-mlp-projection-fp32','batch-mlp-projection-fp32-attention','batch-mlp-projection-fp32-scan','batch-mlp-projection-fp32-scan-npu-attention','batch-mlp-projection-fp32-scan-npu-attention-readout','medium-npu-attention','full-batch','full-batch-precise-medium','full-batch-sparse','full-batch-sparse-cblas','full-batch-sparse-native','full-batch-sparse-fused','full-batch-sparse-flash','full-batch-dense-medium','full-batch-sparse-vector','full-batch-sparse-diagonal','full-batch-sparse-compact','full-batch-sparse-band2','full-batch-sparse-sequential','full-batch-sparse-band25','full-batch-sparse-native-scan','full-batch-sparse-visual-scope','full-batch-sparse-request-scope') else getattr(owner,'batch_planner',None)
   if algorithm is None:raise RuntimeError('Protection pairing requires a candidate')
   candidate_logits=model.logits
   records=[];emit({'event':'ready','case':a.case,'variant':a.variant})
   for name in ('protect115','protect615'):
    model.system_one(**cases[name])
    for repeat in range(5):
     for enabled in ((False,True) if repeat%2==0 else (True,False)):
      saved=model.logits;condition=operating_state()
      model.logits=candidate_logits if enabled else baseline_logits
      emit({'event':'request_begin','case':name});start=time.perf_counter()
      try:response=model.system_one(**cases[name]);seconds=time.perf_counter()-start
      finally:model.logits=saved
      records.append({'case':name,'candidate':enabled,'seconds':seconds,'response':response,'operating_state':condition})
      (out/'protection.json').write_text(json.dumps({'completed':False,'records':records})+'\n')
   (out/'protection.json').write_text(json.dumps({'completed':True,'records':records})+'\n')
   emit({'event':'complete','measurements':len(records)});return
  if a.variant=='reference':
   prepared=model.prepare(**request);responses={};tokens=0;seconds=0
   reference_engine_calls_before=owner.engine.calls
   from cpu_reference import install as install_reference
   reference_state,restore_reference=install_reference(model);restores.append(restore_reference)
   keys=prepared.runnable
   widths={key:max(len(prepared.sequences[k]) for k in keys[start:start+model.batch_size])
           for start in range(0,len(keys),model.batch_size) for key in keys[start:start+model.batch_size]}
   emit({'event':'ready','case':a.case,'variant':a.variant})
   for index,(key,question) in enumerate(request['questions'].items()):
    item=dict(request,questions={key:question});single=model.prepare(**item)
    assert prepared.sequences[key]==single.sequences[key],'Reference changed prompt tokens'
    reference_state['width']=widths[key]
    emit({'event':'request_begin','question_index':index});start=time.perf_counter()
    response=model.system_one(**item);seconds+=time.perf_counter()-start
    responses.update(response['answers']);tokens+=response['usage']['input_tokens']
    (out/'reference-progress.json').write_text(json.dumps({'completed':False,'questions':index+1,'answers':responses,'tokens':tokens,'seconds':seconds})+'\n')
    emit({'event':'question_complete','question_index':index})
   response={'model':model.model_name,'answers':responses,'usage':{'input_tokens':tokens,'output_tokens':0}}
   assert owner.engine.calls==reference_engine_calls_before,'CPU reference unexpectedly used NPU matrix calls'
   (out/'result.json').write_text(json.dumps({'completed':True,'case':a.case,'variant':a.variant,'reference_kind':'CPU dense + original current CPU kernels; original batch left-padding width preserved for each identical-token row; not a speed benchmark','reference_shapes':reference_state['records'],'npu_matrix_calls':owner.engine.calls-reference_engine_calls_before,'seconds':seconds,'response':response,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})+'\n')
   emit({'event':'complete','questions':len(responses)});return
  if a.profile:
   import torch
   if a.variant in ('full-batch','full-batch-precise-medium','full-batch-sparse','full-batch-sparse-cblas','full-batch-sparse-native','full-batch-sparse-fused','full-batch-sparse-flash','full-batch-dense-medium','full-batch-sparse-vector','full-batch-sparse-diagonal','full-batch-sparse-compact','full-batch-sparse-band2','full-batch-sparse-sequential','full-batch-sparse-band25','full-batch-sparse-native-scan','full-batch-sparse-visual-scope','full-batch-sparse-request-scope'):
    prepared=model.prepare(**request);keys=prepared.runnable;groups=owner.projections.inventory['groups']
    widths={g['parent']:sum(owner.projections.modules[g['parent']+'.'+name].out_features for name in g['members']) for g in groups}
    batches=[]
    for start in range(0,len(keys),model.batch_size):
     lengths=[len(prepared.sequences[k]) for k in keys[start:start+model.batch_size]];rows=len(lengths)*max(lengths)
     sizes={parent:rows*width*2 for parent,width in widths.items()}
     batches.append({'lengths':lengths,'rows':rows,'group_output_bytes':sizes,
                     'over_budget':[name for name,size in sizes.items() if size>owner.long_projection.max_output_bytes]})
    (out/'projection-memory-plan.json').write_text(json.dumps({'limits':owner.long_projection.counts['limits'],
      'extra_prepare_for_diagnostic_only':True,'batches':batches})+'\n')
   guarded=set(getattr(getattr(owner,'last_readout',None),'guarded_modules',[]))
   for index,(layer,_) in enumerate(model.encoder.stack):
    if layer not in guarded:wrap(layer,'forward',f'layer.{index}','layer')
   for name,module in model.backbone.named_modules():
    if module in guarded:continue
    typ=type(module).__name__
    if isinstance(module,torch.nn.Linear):wrap(module,'forward',name,'linear')
    elif typ=='Qwen3_5MLP' and a.variant not in ('mlp','batch-mlp','batch-mlp-projection','batch-mlp-projection-fp32','batch-mlp-projection-fp32-attention','batch-mlp-projection-fp32-scan','batch-mlp-projection-fp32-scan-npu-attention','batch-mlp-projection-fp32-scan-npu-attention-readout','medium-npu-attention','full-batch','full-batch-precise-medium','full-batch-sparse','full-batch-sparse-cblas','full-batch-sparse-native','full-batch-sparse-fused','full-batch-sparse-flash','full-batch-dense-medium','full-batch-sparse-vector','full-batch-sparse-diagonal','full-batch-sparse-compact','full-batch-sparse-band2','full-batch-sparse-sequential','full-batch-sparse-band25','full-batch-sparse-native-scan','full-batch-sparse-visual-scope','full-batch-sparse-request-scope'):wrap(module,'forward',name,'mlp')
    elif typ=='Qwen3_5Attention':wrap(module,'forward',name,'attention')
    elif 'GatedDeltaNet' in typ:wrap(module,'forward',name,'scan_layer')
   if hasattr(owner,'long_attention'):
    wrap(__import__('long_attention_candidate'),'kernel','long_attention_native','attention_kernel')
   if hasattr(owner,'long_scan'):
    wrap(__import__('long_scan_candidate'),'kernel','long_scan_native','scan_kernel')
   if hasattr(owner,'long_npu_attention'):
    wrap(owner.long_npu_attention,'kernel','long_npu_attention_native','attention_kernel')
   if hasattr(owner,'medium_sparse'):
    wrap(owner.medium_sparse,'correct_norm','precision.normal_read_and_repair','precision_read_normal')
    wrap(owner.medium_sparse,'correct_native','precision.native_read_and_repair','precision_read_native')
    wrap(owner.medium_sparse.repair,'repair','precision.sparse_repair','precision_repair')
   # The public attention module includes projections and gating. Time the
   # registered SDPA path separately so that those costs are not called CPU
   # attention time. This includes all six calls on this diagnostic input.
   from types import SimpleNamespace
   from transformers.models.qwen3_5 import modeling_qwen3_5 as profile_qwen
   profile_registry=profile_qwen.ALL_ATTENTION_FUNCTIONS
   previous_sdpa=profile_registry['sdpa'];profile_slot=SimpleNamespace(call=previous_sdpa)
   wrap(profile_slot,'call','registered_sdpa','sdpa')
   profiled_sdpa=profile_slot.call;profile_registry.register('sdpa',profiled_sdpa)
   def restore_profile_sdpa():
    if profile_registry['sdpa'] is profiled_sdpa:profile_registry.register('sdpa',previous_sdpa)
   restores.append(restore_profile_sdpa)
   wrap(model,'logits','logits','batch');wrap(model,'prepare','prepare','prepare');wrap(model,'respond','respond','respond')
  emit({'event':'ready','case':a.case,'profile':a.profile,'variant':a.variant})
  if not 1<=a.repeat<=5:raise ValueError('Development repeats must be between 1 and 5')
  records=[]
  for repeat in range(a.repeat):
   condition=operating_state()
   emit({'event':'request_begin','repeat':repeat});start=time.perf_counter()
   response=model.system_one(**request);seconds=time.perf_counter()-start
   record={'repeat':repeat,'seconds':seconds,'response':response,
           'engine_handles':len(owner.engine.handles),'native_mlp_handles':len(owner.native_mlp.handles),
           'native_attention_handles':len(owner.native_attention.handles),
           'medium_flash_handles':len(getattr(getattr(owner,'medium_flash',None),'handles',[])),
      'long_npu_attention_handles':len(getattr(getattr(owner,'long_npu_attention',None),'handles',[])),
           'operating_state':condition,'encoder_restored':model.encoder is owner.last_readout.encoder}
   records.append(record)
   data={'completed':repeat+1==a.repeat,'case':a.case,'profile':a.profile,'variant':a.variant,'seconds':seconds,'response':response,'records':records,'counts_before':before,'counts_after':get_counts(),'components':totals,'medium_sparse_counts':owner.medium_sparse.snapshot() if hasattr(owner,'medium_sparse') else {},'packed_cache_policy_counts':dict(getattr(getattr(owner,'packed_cache_policy',None),'counts',{})),'memory_after':memory_snapshot(),'medium_dense_counts':getattr(getattr(owner,'medium_dense',None),'counts',{}),'medium_flash_counts':getattr(getattr(owner,'medium_flash',None),'counts',{}),
       'medium_precision_counts':copy.deepcopy(getattr(getattr(owner,'medium_precision',None),'counts',{})),'visual_backend_scope_counts':dict(getattr(owner,'visual_backend_scope_counts',{})),'request_backend_scope_counts':dict(getattr(owner,'request_backend_scope_counts',{})),'candidate_counts':copy.deepcopy(getattr(getattr(owner,'batch_planner',getattr(owner,'long_mlp',None)),'counts',{})),'long_readout_counts':copy.deepcopy(getattr(getattr(owner,'long_readout',None),'counts',{})),'long_npu_attention_counts':copy.deepcopy(getattr(getattr(owner,'long_npu_attention',None),'counts',{})),'long_scan_counts':copy.deepcopy(getattr(getattr(owner,'long_scan',None),'counts',{})),'long_attention_counts':copy.deepcopy(getattr(getattr(owner,'long_attention',None),'counts',{})),'long_projection_counts':copy.deepcopy(getattr(getattr(owner,'long_projection',None),'counts',{})),'long_mlp_counts':copy.deepcopy(getattr(getattr(owner,'long_mlp',None),'counts',{})),'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
   (out/'result.json').write_text(json.dumps(data)+'\n');emit({'event':'case_complete','repeat':repeat,'seconds':seconds})
  if getattr(getattr(owner,'long_readout',None),'counts',{}).get('optimized',0):
   transitions=[]
   for name in ('protect615','protect115'):
    emit({'event':'request_begin','case':name,'phase':'after_long_transition'});start=time.perf_counter()
    protected=model.system_one(**cases[name]);elapsed=time.perf_counter()-start
    transitions.append({'id':name,'seconds':elapsed,'response':protected,
      'encoder_restored':model.encoder is owner.last_readout.encoder})
   (out/'after-long-transition.json').write_text(json.dumps({'completed':True,'records':transitions})+'\n')
  emit({'event':'complete','seconds':seconds,'tokens':response['usage']['input_tokens'],'repeats':len(records)})
 finally:
  if owner:(out/'runtime-backend-before-cleanup.json').write_text(json.dumps(read_backend_state('before_cleanup'))+'\n')
  for restore in reversed(restores):restore()
  try:
   if close:close()
  finally:
   if owner:
    (out/'lifecycle.json').write_text(json.dumps({'engine_handles':len(owner.engine.handles),
      'native_mlp_handles':len(owner.native_mlp.handles),'native_attention_handles':len(owner.native_attention.handles),
      'medium_flash_handles':len(getattr(getattr(owner,'medium_flash',None),'handles',[])),
      'long_npu_attention_handles':len(getattr(getattr(owner,'long_npu_attention',None),'handles',[]))})+'\n')
   trace.close()
if __name__=='__main__':main()
