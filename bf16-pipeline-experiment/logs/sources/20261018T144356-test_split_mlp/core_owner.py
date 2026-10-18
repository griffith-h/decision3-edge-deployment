"""Shared minimal experiment core: only adopted components, no service integration.

The old Owner owns the model, original graph banks, native engine and its pool.
This Owner owns only new attention contexts, optional640 graphs, and forward
restores. Private native work is joined before any fallback or destruction.
"""
import sys
sys.path.insert(0,'/tail')
from runner import Owner as PreviousOwner,state,stats,P,json,time,copy,torch,detailed
from pathlib import Path
from npu_attention import NpuAttention
from precise_mlp import PreciseMlp
import experiment,cpu_backend,long_kernels as lk
import ctypes,ast
class Owner(PreviousOwner):
 bridge_path="/experiment/libwide_bridge.so"
 def __init__(self):
  self.execution=json.loads(Path('/experiment/execution.json').read_text());assert self.execution['mode'] in ('wide','four','neon');assert type(self.execution['grouped']) is bool
  self.native_attention=None;self.original_attention_object=None;self.grouped_attention=None;self.full_mlp=None;self.precise_mlp=None;self.row_original=None;self.scan_original=None;self.scan_options={};self.wide_enabled=False;self.npu_calls=0
  load=experiment.load_baseline;get=cpu_backend.Backend.get
  experiment.load_baseline=lambda **kw:load(library=self.bridge_path)
  def wide_get(backend,tuned):
   parts=list(get(backend,tuned));parts[-1]=cpu_backend.clone_types(ctypes.CDLL(self.bridge_path),backend.initial_engine);return tuple(parts)
  cpu_backend.Backend.get=wide_get
  try:super().__init__()
  finally:experiment.load_baseline=load;cpu_backend.Backend.get=get
  try:
   self.base_attention=self.native_originals[0];self.scan_original=lk.scan
   self.native_attention=NpuAttention(fused=True,grouped=self.execution['grouped']);self.original_attention_object=self.native_attention
   if self.execution['grouped']:self.grouped_attention=self.native_attention
   if self.execution.get('full640',True):
    from full_mlp import FullMlp
    self.full_mlp=FullMlp(self.bank)
   self.precise_mlp=PreciseMlp(self.model,self.engine,self.candidate.table)
   selected=self.execution['mode']
   if selected in ('four','neon'):
    lib=ctypes.CDLL('/experiment/'+('libscan_four.so' if selected=='four' else 'libneon_scan.so'))
    for name in ('d3_set_sgemm','d3_set_strsm','d3_chunk_scan'):
     a=getattr(lk.scan,name);b=getattr(lib,name);b.argtypes=a.argtypes;b.restype=a.restype
    lib.d3_set_sgemm(ctypes.cast(lk.blas.scipy_cblas_sgemm64_,ctypes.c_void_p));lib.d3_set_strsm(ctypes.cast(lk.blas.scipy_cblas_strsm64_,ctypes.c_void_p));self.scan_options[selected]=lib
   self.row_original=self.projections.run
   def rows(members,x,key):
    old_width=self.projections.width;self.projections.width=640 if self.wide_enabled and tuple(x.shape) in ((1,615,1024),(1,615,2048)) else 512
    try:return self.row_original(members,x,key)
    finally:self.projections.width=old_width
   self.projections.run=rows
   prior=json.loads(Path('/tail/logs/formal.json').read_text());self.frozen_plans={}
   for case in prior['cases'].values():
    record=case['variants']['baseline']['records'][0];plans=record['mlp'].get('plans',{})
    if len(plans)!=1:continue
    plan=tuple(map(int,next(iter(plans)).split(',')));shape=ast.literal_eval(next(iter(record['mlp']['shapes'])));n=shape[0]*shape[1];self.bank.plans[n]=plan;self.frozen_plans[str(n)]=list(plan)
   assert tuple(self.bank.plans[615])==(128,128,384)
  except BaseException:self.close();raise
 def use(self,mode):
  assert mode in ('current','baseline','candidate')
  super().use('current' if mode=='current' else 'validate_mlp');candidate=mode=='candidate';self.npu_calls=0;self.wide_enabled=candidate
  if self.full_mlp:self.full_mlp.enabled=candidate and self.execution.get('full640',True)
  self.precise_mlp.enabled=candidate
  self.full_before=self.full_mlp.calls if self.full_mlp else 0;self.precise_before=self.precise_mlp.calls
  lk.scan=self.scan_options.get(self.execution['mode'],self.scan_original) if candidate else self.scan_original
  self.native_attention.residual=self.execution.get('residual',False) if candidate else False
  if candidate:
   def attention(q,k,v,mask,scale):
    if tuple(q.shape)==(1,8,615,256):
     out=self.native_attention(q,k,v,mask,scale);self.npu_calls+=1;return out
    return self.base_attention(q,k,v,mask,scale)
   lk.attention=attention
 def counts(self):
  return {**super().counts(),'attention_npu_calls':self.npu_calls,'full640_calls':self.full_mlp.calls-self.full_before if self.full_mlp else 0,'full640_errors':list(self.full_mlp.errors) if self.full_mlp else [],'precise_calls':self.precise_mlp.calls-self.precise_before,'precise_errors':list(self.precise_mlp.errors),'precise_range_fallbacks':self.precise_mlp.range_fallbacks,'wide_enabled':self.wide_enabled}
 def close(self):
  try:
   if self.row_original and hasattr(self,'projections') and self.projections:self.projections.run=self.row_original
   if self.precise_mlp:self.precise_mlp.close()
   if self.full_mlp:self.full_mlp.close()
  finally:
   try:
    if self.native_attention:self.native_attention.close()
   finally:
    if self.scan_original:lk.scan=self.scan_original
    super().close()
