"""Static row buckets; zero-padding stays strictly inside each token-wise MLP."""
from pathlib import Path
from row_mlp import RowBank
from bucket_graph import HalfGraph,lib
class BucketBank(RowBank):
 def __init__(self,model,root=Path('/experiment'),factory=HalfGraph,base_factory=None):
  self.extra=[];self.by_id={}
  super().__init__(model,**({'factory':base_factory} if base_factory is not None else {}))
  try:
   for i,g in enumerate(self.graphs):
    bank={}
    for rows in (128,384):
     h=factory(root/f'mlp-{i}-rows{rows}.rknn');self.extra.append(h)
     if h.rows!=rows:raise ValueError('Wrong bucket model shape')
     if factory is HalfGraph and lib.row_graph_set_core(h.handle,0):raise RuntimeError('Core selection failed')
     bank[rows]=h
    self.by_id[id(g)]=bank
  except BaseException:self.close();raise
 def run_rows(self,g,x):
  n=x.shape[0]*x.shape[1]
  if self.mode!='bucket' or n<=115:return super().run_rows(g,x)
  width=128 if n<=128 else 384
  self.counts.setdefault('bucket_calls',{});key=str(width);self.counts['bucket_calls'][key]=self.counts['bucket_calls'].get(key,0)+(n+width-1)//width
  return self.run_packed(self.by_id[id(g)][width],x,width)
 def close(self):
  if getattr(self,'closed',False):return
  for g in self.extra:g.close()
  super().close()
