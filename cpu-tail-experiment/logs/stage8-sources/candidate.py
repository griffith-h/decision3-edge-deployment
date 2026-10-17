"""Scoped candidate toggles; never replace production or mutate frozen sources."""
import torch
from lut_ops import install as lut_install
from cpu_attention import install as attention_install
from small_linear import install as small_install
from rotary import install as rotary_install
from residual import install as residual_install
from attention_prepare import install as prepare_install
from pipeline import install as pipeline_install

class Candidate:
 def __init__(self,model,engine,table):self.model=model;self.engine=engine;self.table=table;self.restores=[];self.counts={};self.initial_mkldnn=torch.backends.mkldnn.enabled
 def reset(self):
  for r in reversed(self.restores):r()
  self.restores=[];self.counts={};torch.backends.mkldnn.enabled=self.initial_mkldnn
 def use(self,cfg):
  self.reset()
  entries=[]
  if cfg.get('lut'):entries.append(('lut',lambda:lut_install(self.model,self.engine,self.table,cfg['lut'])))
  if cfg.get('attention') is not None:entries.append(('attention',lambda:attention_install(cfg['attention'])))
  if cfg.get('small') is not None:entries.append(('small',lambda:small_install(self.model,cfg['small'])))
  if cfg.get('rotary'):entries.append(('rotary',rotary_install))
  if cfg.get('residual'):entries.append(('residual',lambda:residual_install(self.model)))
  if cfg.get('attention_prepare'):entries.append(('attention_prepare',lambda:prepare_install(self.model,self.engine,self.table)))
  if cfg.get('pipeline'):entries.append(('pipeline',lambda:pipeline_install(self.model,self.engine,self.table)))
  for name,fn in entries:
   restore,c=fn();self.restores.append(restore);self.counts[name]=c
  torch.backends.mkldnn.enabled=cfg.get('mkldnn',self.initial_mkldnn)
  torch.set_num_threads(cfg.get('threads',3))
  return self.counts
