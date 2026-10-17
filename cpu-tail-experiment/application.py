"""One synchronized model owner and optional persistent graph owner bank."""
from baseline import load_baseline,state,stats,compare
from lut_ops import Tables
from candidate import Candidate
from validate_candidate import Evaluation,sources
from mlp_graph import HalfGraph,lib as graphlib
from mlp_subgraph import install as install_graph
from pathlib import Path
class Application:
 def __init__(self,profile=False):
  self.model,self.engine,self.base_counts,self.close_model=load_baseline(profile=profile,library='/work/libpipeline_profile.so' if profile else '/work/libpipeline_bridge.so')
  self.candidate=Candidate(self.model,self.engine,Tables());self.ev=Evaluation(self.model);self.graphs={};self.graph_restore=None;self.graph_counts={};self.config={}
 def use(self,cfg):
  if self.graph_restore:self.graph_restore();self.graph_restore=None
  self.candidate.use(cfg);self.graph_counts={};self.config=dict(cfg)
  if cfg.get('mlp_graph'):
   bank=self.graphs;core=cfg.get('graph_core') or 0
   class Borrowed:
    def __init__(self,path):
     key=(str(path),core)
     if key not in bank:
      g=HalfGraph(path)
      if graphlib.mlp_graph_set_core(g.handle,core):g.close();raise RuntimeError('Graph core mask rejected '+str(core))
      bank[key]=g
     self.g=bank[key]
    def run_bf16(self,x):return self.g.run_bf16(x)
    def close(self):pass
   self.graph_restore,self.graph_counts=install_graph(self.model,Borrowed)
 def invoke(self,c=None):return self.ev.invoke(c)
 def check(self,r,c=None):return self.ev.check(r,c)
 def counts(self):return {'cpu':{k:dict(v) for k,v in self.candidate.counts.items()},'graphs':dict(self.graph_counts)}
 def close(self):
  if self.graph_restore:self.graph_restore();self.graph_restore=None
  self.candidate.reset()
  for g in self.graphs.values():g.close()
  self.graphs.clear();self.close_model()
